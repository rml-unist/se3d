"""Checkpointing and exact-update execution shared by transfer experiments.

No scheduler submission or background watcher lives here. A job wrapper may
resume an exit status of 75 with the identical command in another allocation.
"""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from se3d.backends import backend_state

try:
    from .protocol_utils import atomic_json, json_digest, sha256
except ImportError:
    from protocol_utils import atomic_json, json_digest, sha256

SCHEMA = 'dsec-exact-update-v1'
RESUME_EXIT_CODE = 75


class Preempted(Exception):
    """A requested, resumable interruption at a safe execution boundary."""


def atomic_torch(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_pickle(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=4)
    temporary.replace(path)


def tensor_state_sha256(state):
    """Hash tensor values, not torch.save's serialization/container bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def code_fingerprint(root):
    root = Path(root)
    files = {}
    for folder in ('se3d', 'emod', 'dsgn_event', 'transfer'):
        for path in sorted((root / folder).rglob('*')):
            if not path.is_file() or path.suffix not in ('.py', '.yaml', '.yml', '.cpp', '.cu', '.h'):
                continue
            if {'tests', '__pycache__', '.git', 'build'}.intersection(path.relative_to(root).parts):
                continue
            files[str(path.relative_to(root))] = sha256(path)
    return dict(files=files, sha256=json_digest(files))


def environment():
    packages = {}
    for name in ('numpy', 'torch', 'torchvision', 'opencv-python', 'h5py', 'PyYAML', 'easydict'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, packages=packages, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), **backend_state())


def metrics_environment(interpreter):
    if not interpreter:
        return None
    script = ("import importlib.metadata as m,json,sys; "
              "print(json.dumps({'python':sys.version,'packages':"
              "{n:m.version(n) for n in ('numpy','tensorflow','waymo-open-dataset-tf-2-12-0')}}))")
    result = subprocess.run([str(interpreter), '-c', script], check=True, capture_output=True,
                            text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=60)
    return json.loads(result.stdout)


def capture_rng():
    return dict(torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng=random.getstate(), numpy_rng=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng']:
        if not torch.cuda.is_available():
            raise ValueError('Cannot resume a CUDA RNG state on a CPU-only runtime')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


@contextlib.contextmanager
def preserved_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


@contextlib.contextmanager
def run_lock(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns this output directory: ' + str(run)) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Budget:
    """SIGUSR1/SIGTERM, elapsed-time limit, and an optional Slurm end time.

    SLURM_JOB_END_TIME must be Unix seconds if supplied by a launch wrapper.
    The margin covers model/optimizer and partial-prediction checkpoint I/O.
    """
    def __init__(self, max_seconds=0, margin_seconds=600):
        if max_seconds < 0 or margin_seconds < 0 or (max_seconds and max_seconds <= margin_seconds):
            raise ValueError('Time limit must exceed the nonnegative checkpoint margin')
        self.started = time.monotonic()
        self.deadline = self.started + max_seconds - margin_seconds if max_seconds else float('inf')
        end = os.environ.get('SLURM_JOB_END_TIME')
        if end:
            self.deadline = min(self.deadline, self.started + float(end) - time.time() - margin_seconds)
        self.reason = None
        self.handlers = {}

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            self.handlers[signum] = signal.signal(signum, self._signal)
        return self

    def __exit__(self, *unused):
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)

    def _signal(self, signum, frame):
        self.reason = 'received ' + signal.Signals(signum).name

    def check(self):
        if self.reason or time.monotonic() >= self.deadline:
            raise Preempted(self.reason or 'allocation checkpoint margin reached')


def ensure_identity(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to reuse an artifact with different provenance: ' + str(path))
    else:
        atomic_json(value, path)


class EvaluationProgress:
    """A contiguous prefix of evaluation frames, bound to weights and inputs."""
    def __init__(self, path, binding, frames, detection):
        self.path = Path(path)
        self.binding, self.frames, self.detection = binding, frames, detection
        self.cursor = 0
        self.sums = np.zeros(5, dtype=np.float64)
        self.gt, self.pred = [], []
        if self.path.exists():
            value = pickle.loads(self.path.read_bytes())
            if value.get('schema') != SCHEMA or value['binding'] != binding:
                raise ValueError('Evaluation cache does not match checkpoint/data/code: ' + str(self.path))
            self.cursor, self.sums = value['cursor'], np.asarray(value['sums'], dtype=np.float64)
            self.gt, self.pred = value['gt'], value['predictions']
            if (value['frames'] != frames or value['detection'] != detection
                    or not 0 <= self.cursor <= frames or self.sums.shape != (5,)
                    or not np.isfinite(self.sums).all()
                    or (detection and (len(self.gt) != self.cursor or len(self.pred) != self.cursor))
                    or (not detection and (self.gt or self.pred))):
                raise ValueError('Malformed partial evaluation cache: ' + str(self.path))

    def append(self, index, sums, gt=None, prediction=None):
        if index != self.cursor or index >= self.frames:
            raise ValueError('Evaluation frames must be appended once, in protocol order')
        sums = np.asarray(sums, dtype=np.float64)
        if sums.shape != (5,) or not np.isfinite(sums).all() or (sums < 0).any():
            raise ValueError('Invalid disparity error sums')
        if self.detection:
            if gt is None or prediction is None:
                raise ValueError('Missing detection frame')
            self.gt.append(gt)
            self.pred.append(prediction)
        self.sums += sums
        self.cursor += 1

    def save(self):
        atomic_pickle(dict(schema=SCHEMA, binding=self.binding, frames=self.frames, detection=self.detection,
                           cursor=self.cursor, sums=self.sums, gt=self.gt, predictions=self.pred), self.path)


def validation_steps(updates, every):
    if updates <= 0 or every <= 0:
        raise ValueError('Update budget and validation interval must be positive')
    steps = list(range(every, updates + 1, every))
    if not steps or steps[-1] != updates:
        steps.append(updates)
    return steps


def train_loop(model, optimizer, run, config, anchors, initialization, batches, compute_loss, validate,
               check_stop=lambda: None, save_every=200, keep_updates=(), allocation_updates=0):
    """Cycle seeded permutations; validate at fixed updates even inside a cycle.

    Callbacks use the actual model and optimizer; CPU regression tests run this
    same loop with a tiny stochastic network. Datasets must have no stochastic
    transforms (the transfer protocol has no augmentation).
    """
    run = Path(run)
    updates, every = config['optimizer_updates'], config['validate_every']
    candidates = validation_steps(updates, every)
    size, seed = config['training_frames'], config['seed']
    limit = config.get('limit_train')
    cycle_size = min(size, limit) if limit else size
    if cycle_size <= 0 or save_every <= 0 or allocation_updates < 0:
        raise ValueError('Invalid training/checkpoint budget')
    if any(step <= 0 or step > updates for step in keep_updates):
        raise ValueError('A retained update is outside the training budget')
    epoch = cursor = step = last_validated_step = validations = 0
    best, best_step = -float('inf'), None
    last = run / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state.get('schema') != SCHEMA or state['config'] != config:
            raise ValueError('last.pth belongs to a different runtime/configuration')
        if state['anchors'] != anchors or state['initialization'] != initialization:
            raise ValueError('Checkpoint anchor/initialization provenance differs')
        epoch, cursor, step = state['epoch'], state['cursor'], state['step']
        best, best_step = state['best'], state['best_step']
        last_validated_step, validations = state['last_validated_step'], state['validations']
        if (not 0 <= step <= updates or not 0 <= cursor <= cycle_size or epoch < 0
                or step != epoch * cycle_size + cursor
                or last_validated_step not in [0] + candidates
                or last_validated_step > step
                or validations != len([s for s in candidates if s <= last_validated_step])):
            raise ValueError('Invalid training/sampler/validation cursor in checkpoint')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state)
        print(json.dumps(dict(status='resumed', step=step, data_epoch=epoch, cursor=cursor)), flush=True)
    initial_step = step

    def weights():
        return dict(schema=SCHEMA, model=model.state_dict(), epoch=epoch + bool(cursor), cursor=cursor,
                    step=step, config=config, anchors=anchors, initialization=initialization)

    def save():
        state = weights()
        state.update(optimizer=optimizer.state_dict(), epoch=epoch, best=best, best_step=best_step,
                     last_validated_step=last_validated_step, validations=validations, **capture_rng())
        atomic_torch(state, last)

    def stop():
        check_stop()
        if allocation_updates and step - initial_step >= allocation_updates:
            raise Preempted('requested allocation update limit reached')

    try:
        while True:
            stop()
            if step in candidates and step > last_validated_step:
                save()  # A resumed allocation must finish this validation before another update.
                with preserved_rng():
                    result, score = validate(step, stop)
                if not np.isfinite(score):
                    raise ValueError('Undefined model selection at update %d' % step)
                result = dict(result, step=step, data_epoch=epoch, cursor=cursor, selection_score=float(score))
                atomic_json(result, run / ('validation_step_%07d_metrics.json' % step))
                if score > best:  # Strict comparison deliberately retains the earliest exact tie.
                    best, best_step = float(score), step
                    state = weights()
                    state.update(validation_score=best, selected_step=best_step)
                    atomic_torch(state, run / 'best.pth')
                last_validated_step, validations = step, validations + 1
                save()
            if step == updates:
                break
            if cursor == cycle_size:
                epoch, cursor = epoch + 1, 0
            order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()[:cycle_size]
            model.train()
            progressed = False
            for batch in batches(order[cursor:], epoch):
                stop()
                optimizer.zero_grad(set_to_none=True)
                loss, record = compute_loss(batch)
                if loss.numel() != 1 or not torch.isfinite(loss).all():
                    raise ValueError('Nonfinite/non-scalar target loss at update %d' % step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                cursor, step, progressed = cursor + 1, step + 1, True
                if step % 20 == 0:
                    record = dict(record, epoch=epoch + 1, cursor=cursor, total=cycle_size,
                                  step=step, loss=float(loss.detach()))
                    print(json.dumps(record), flush=True)
                    with (run / 'training.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                if step in keep_updates:
                    atomic_torch(weights(), run / ('step_%07d.pth' % step))
                if step % save_every == 0:
                    save()
                if step in candidates:
                    break
            if not progressed:
                raise ValueError('Training loader ended before the saved sampler cursor')
    except Preempted as exc:
        save()
        receipt = dict(status='checkpointed_for_resume', reason=str(exc), step=step,
                       data_epoch=epoch, cursor=cursor, last_validated_step=last_validated_step)
        atomic_json(receipt, run / 'allocation_exit.json')
        print(json.dumps(receipt), flush=True)
        return RESUME_EXIT_CODE

    if validations != len(candidates) or best_step is None:
        raise RuntimeError('Incomplete fixed-step validation schedule')
    atomic_torch(weights(), run / 'final.pth')
    save()
    summary = dict(status='complete', schema=SCHEMA, steps=step, data_epoch=epoch, cursor=cursor,
                   best_step=best_step, best_validation_score=best, validations=validations,
                   validation_steps=candidates, config_sha256=json_digest(config),
                   best_checkpoint_sha256=sha256(run / 'best.pth'),
                   final_checkpoint_sha256=sha256(run / 'final.pth'))
    atomic_json(summary, run / 'training_complete.json')
    atomic_json(dict(status='complete', step=step), run / 'allocation_exit.json')
    return 0
