"""Train a baseline with explicit annotation and checkpoint-selection rules.

The release defaults use label/ for training and validation, and select the
highest validation Moderate 3D mAP40 (earliest epoch on ties). Validation uses
the full, fixed validation split, including for a sunny-training-only model.
See docs/EXPERIMENTS.md for the historical paper protocol and new run matrix.

Repeating the same command resumes optimizer, RNG, epoch and cursor from
last.pth. A time limit or termination signal saves progress and returns 75;
a scheduler may requeue that run. Other errors are not treated as preemption.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from se3d import CONDITIONS, REPO_ROOT, SPLITS_FILE
from se3d.data import SE3DFrames, model_args
from se3d.engine import evaluate, loader
from se3d.models import ANCHORS, MODELS, anchor_name, build_model, configure_classes
from se3d.protocol import LABELS, add_labels_argument, label_name


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(obj, path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def annotation_fingerprint(dataset):
    digest = hashlib.sha256()
    for name, sequence in zip(dataset.sequence_names, dataset.datasets):
        paths = sequence.labels_dataset.timestamp_to_labels_path['labels']
        for timestamp in sequence.timestamps:
            digest.update(('%s/%d\0' % (name, timestamp)).encode())
            digest.update(Path(paths[timestamp]).read_bytes())
            digest.update(b'\0')
    return digest.hexdigest()


def code_fingerprint():
    digest = hashlib.sha256()
    for folder in ('se3d', 'emod', 'dsgn_event'):
        for path in sorted((REPO_ROOT / folder).rglob('*')):
            if path.suffix not in ('.py', '.yaml') or '__pycache__' in path.parts:
                continue
            digest.update(path.relative_to(REPO_ROOT).as_posix().encode() + b'\0')
            digest.update(path.read_bytes())
    for name in ('train.py', 'test.py', 'evaluate.py', 'compute_anchors.py'):
        path = REPO_ROOT / 'tools' / name
        digest.update(('tools/' + name).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


class TrainingInterrupted(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data-root', required=True, help='SE3D root containing calib.txt and map*/')
    parser.add_argument('--cache-root', help='writable event cache root, separate from the dataset')
    parser.add_argument('--cache-time-bounds', help='verified time-bound metadata for cache-only input')
    parser.add_argument('--output', required=True, help='run directory')
    add_labels_argument(parser)
    parser.add_argument('--validation-labels', type=label_name, choices=LABELS, default='label',
                        help='validation annotations; default label for every training arm')
    parser.add_argument('--selection-metric', choices=('ap40', 'ap11'), default='ap40',
                        help='Moderate validation 3D mAP; ap11 reproduces the historical paper rule')
    parser.add_argument('--anchors', type=anchor_name, choices=sorted(ANCHORS), default=None,
                        help='defaults to training-label anchors; sunny-only training uses sunny anchors')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=None,
                        help='restrict training conditions only')
    parser.add_argument('--validation-conditions', nargs='+', choices=CONDITIONS, default=None,
                        help='restrict validation conditions (default: the full validation split)')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--epochs', type=int, default=40)
    budget.add_argument('--updates', type=int, help='exact optimizer-update budget, including a partial last epoch')
    parser.add_argument('--validate-every', type=int,
                        help='validation interval in updates (default: one training epoch)')
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--keep-epochs', nargs='*', type=int, default=[],
                        help='also save model weights at these fixed epochs; final epoch is always kept')
    parser.add_argument('--keep-updates', nargs='*', type=int, default=[214368],
                        help='save fixed-update weights independently of checkpoint selection')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='save and return 75 after this allocation time; continue with the same command')
    parser.add_argument('--allocation-updates', type=int,
                        help='save and return 75 after this many new updates, useful for resume checks')
    parser.add_argument('--strict-determinism', action='store_true',
                        help='error on operations without deterministic CUDA implementations')
    parser.add_argument('--limit-train', type=int, default=None, help='debug: use only the first N training frames')
    parser.add_argument('--limit-val', type=int, default=None, help='debug: validation frames')
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0 or any(x is not None and x < 1 for x in
            (args.limit_train, args.limit_val, args.max_seconds, args.updates, args.validate_every, args.allocation_updates)):
        parser.error('epochs, time and frame limits must be positive; workers must be nonnegative')

    started = time.monotonic()
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_stop)

    def check_stop():
        if (stop or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)
                or (args.allocation_updates is not None and step - initial_step >= args.allocation_updates)):
            raise TrainingInterrupted()

    sunny = args.conditions is not None and set(args.conditions) == {'day_sunny', 'night_sunny'}
    anchors = ANCHORS[args.anchors or (args.labels + ('_sunny' if sunny else ''))]
    configure_classes(anchors)
    train = SE3DFrames(args.data_root, 'train', label_dir=args.labels, generate_target=True,
                       conditions=args.conditions, cache_root=args.cache_root, cache_time_bounds=args.cache_time_bounds,
                       max_frames=args.limit_train)
    validation = SE3DFrames(args.data_root, 'val', label_dir=args.validation_labels, generate_target=False,
                            conditions=args.validation_conditions, cache_root=args.cache_root,
                            cache_time_bounds=args.cache_time_bounds, max_frames=args.limit_val)
    validation_hash = annotation_fingerprint(validation)
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    train_frames = min(args.limit_train or len(train), len(train))
    updates = args.updates or args.epochs * train_frames
    validation_interval = args.validate_every or train_frames
    epochs = math.ceil(updates / train_frames)
    metric = '3d_mAP' + args.selection_metric[2:]
    config = dict(protocol='se3d-release-20261010', model=args.model, labels=args.labels,
                  validation_labels=args.validation_labels, anchors=anchors.name, anchors_sha256=sha256(anchors),
                  conditions=args.conditions or 'all', validation_conditions=args.validation_conditions or 'all',
                  epochs=epochs, updates=updates, validation_interval_updates=validation_interval,
                  seed=args.seed, batch_size=1, workers=args.workers,
                  learning_rate=1e-4, optimizer='Adam', weight_decay=1e-4, loss_weights=[.5, .5],
                  augmentation='none', initialization='scratch', selection_metric=metric,
                  selection='highest validation Moderate 3D mAP; GT-present classes; earliest epoch breaks ties',
                  keep_epochs=sorted(set(args.keep_epochs + [epochs])),
                  keep_updates=sorted(set(args.keep_updates + [updates])),
                  train_frames=train_frames, validation_frames=len(validation),
                  train_annotations_sha256=annotation_fingerprint(train), validation_annotations_sha256=validation_hash,
                  splits_sha256=sha256(SPLITS_FILE), code_sha256=code_fingerprint(),
                  limit_train=args.limit_train, limit_val=args.limit_val, torch=torch.__version__,
                  cudnn_deterministic=True, cudnn_benchmark=False, allow_tf32=False,
                  strict_determinism=args.strict_determinism)
    config['cache_time_bounds_sha256'] = sha256(args.cache_time_bounds) if args.cache_time_bounds else None
    config_path = run / 'effective_config.json'
    text = json.dumps(config, indent=2) + '\n'
    if config_path.exists() and config_path.read_text() != text:
        raise RuntimeError('%s holds a run with a different configuration or input/code fingerprint' % run)
    config_path.write_text(text)
    if not (run / 'environment.json').exists():
        packages = {}
        for name in ('torch', 'torchvision', 'numpy', 'numba', 'scipy', 'timm', 'h5py', 'hdf5plugin', 'tqdm'):
            packages[name] = importlib.metadata.version(name)
        (run / 'environment.json').write_text(json.dumps(dict(python=sys.version, packages=packages,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name()), indent=2) + '\n')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(args.strict_determinism)
    model = build_model(args.model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epoch = cursor = step = last_validated_step = validations = 0
    best = -float('inf')
    last = run / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cuda', weights_only=False)
        if state['config'] != config:
            raise RuntimeError('Checkpoint configuration differs from effective_config.json')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, step, best = state['epoch'], state['cursor'], state['step'], state['best']
        last_validated_step, validations = state['last_validated_step'], state['validations']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        print('Resumed at epoch %d, step %d' % (epoch + 1, step), flush=True)
    initial_step = step

    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, cursor=cursor,
                         step=step, best=best, last_validated_step=last_validated_step, validations=validations,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), config=config), last)

    def validate():
        nonlocal best, last_validated_step, validations
        check_stop()
        validation_started = time.monotonic()
        result, _ = evaluate(model, validation, args.workers, check_stop=check_stop)
        score = result[metric][1]
        if score is None or not np.isfinite(score):
            raise ValueError('Undefined validation mAP at update %d' % step)
        result.update(epoch=epoch + 1, optimizer_steps=step, labels=args.validation_labels,
                      selection_metric=metric, validation_seconds=time.monotonic() - validation_started)
        result['selection_classes'] = [name for name, values in result['per_class'].items()
                                       if values['valid_gt'][1] > 0]
        (run / ('validation_step_%07d.json' % step)).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(epoch=epoch + 1, step=step, selection_metric=metric,
                              validation_score=score, best=max(best, score))), flush=True)
        if score > best:
            best = score
            atomic_save(dict(model=model.state_dict(), epoch=epoch + 1, step=step,
                             validation_score=score, config=config), run / 'best.pth')
        last_validated_step = step
        validations += 1
        save()
        model.train()

    try:
        while epoch < epochs:
            check_stop()
            if step > last_validated_step and (step % validation_interval == 0 or step == updates):
                validate()  # a requeued allocation may have stopped during validation
            model.train()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            if args.limit_train:
                order = order[:args.limit_train]
            for batch in loader(train, order[cursor:], args.workers, args.seed + epoch):
                if step >= updates:
                    break
                check_stop()
                batch_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                _, _, depth_loss, detection_loss = model(**model_args(batch))
                loss = .5 * depth_loss.mean() + .5 * detection_loss
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss at step %d' % step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                cursor += 1
                step += 1
                if step % 20 == 0:
                    record = dict(epoch=epoch + 1, cursor=cursor, total=len(order), step=step,
                                  loss=float(loss.detach()), depth_loss=float(depth_loss.mean().detach()),
                                  detection_loss=float(detection_loss.detach()),
                                  batch_seconds=time.monotonic() - batch_started,
                                  allocation_seconds=time.monotonic() - started,
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(record), flush=True)
                    with (run / 'training.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                if step % 200 == 0:
                    save()
                if step in config['keep_updates']:
                    atomic_save(dict(model=model.state_dict(), epoch=epoch + 1, step=step, config=config),
                                run / ('step_%07d.pth' % step))
                if step % validation_interval == 0 or step == updates:
                    save()
                    validate()
            save()
            if epoch + 1 in config['keep_epochs']:
                weights = dict(model=model.state_dict(), epoch=epoch + 1, step=step, config=config)
                atomic_save(weights, run / ('epoch_%03d.pth' % (epoch + 1)))
            epoch += 1
            cursor = 0
            save()
    except (TrainingInterrupted, KeyboardInterrupt):
        save()
        print(json.dumps(dict(status='checkpointed_for_resume', epoch=epoch, cursor=cursor, step=step)), flush=True)
        return 75
    (run / 'training_complete.json').write_text(json.dumps(dict(epochs=epoch, steps=step,
        validations=validations, last_epoch_complete=(updates % train_frames == 0),
        selection_metric=metric, best_validation_score=best, config_sha256=sha256(config_path)), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
