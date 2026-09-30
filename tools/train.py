"""Train EMOD or DSGN-event on SE3D and select the checkpoint on the validation split.

Protocol of the paper: batch size 1, Adam (lr 1e-4, weight decay 1e-4), no
augmentation, loss = 0.5 * depth + 0.5 * detection, validation after every
epoch, and the checkpoint with the highest validation Moderate 3D mAP over the
seven classes (11 recall positions) is kept as best.pth for both tasks.

Reproduce the reported 40-epoch models (trained on the annotations before
deduplication):
    python tools/train.py --model emod --data-root /data/SE3D --output runs/emod --labels original
    python tools/train.py --model dsgn_event --data-root /data/SE3D --output runs/dsgn_event --labels original
Sunny-only EMOD (same number of updates as 40 all-condition epochs):
    python tools/train.py --model emod --data-root /data/SE3D --output runs/emod_sunny --labels original \
        --anchors original_sunny --conditions day_sunny night_sunny --epochs 105

The run can be interrupted and restarted with the same command; it resumes from last.pth.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from se3d import CONDITIONS
from se3d.data import SE3DFrames, model_args
from se3d.engine import evaluate, loader
from se3d.models import ANCHORS, MODELS, build_model, configure_classes

LABEL_DIRS = {'original': 'label_original', 'corrected': 'label'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(obj, path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data-root', required=True, help='SE3D root containing calib.txt and map*/')
    parser.add_argument('--output', required=True, help='run directory')
    parser.add_argument('--labels', choices=sorted(LABEL_DIRS), default='corrected',
                        help='training/validation annotations (the reported models use "original")')
    parser.add_argument('--anchors', choices=sorted(ANCHORS), default=None,
                        help='anchor sizes; defaults to the ones computed from --labels')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=None,
                        help='restrict training and validation to these conditions')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-train', type=int, default=None, help='debug: frames per epoch')
    parser.add_argument('--limit-val', type=int, default=None, help='debug: validation frames')
    args = parser.parse_args()

    anchors = ANCHORS[args.anchors or args.labels]
    label_dir = LABEL_DIRS[args.labels]
    configure_classes(anchors)
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, labels=label_dir, anchors=anchors.name, anchors_sha256=sha256(anchors),
                  conditions=args.conditions or 'all', epochs=args.epochs, seed=args.seed, batch_size=1,
                  learning_rate=1e-4, optimizer='Adam', weight_decay=1e-4, loss_weights=[.5, .5],
                  augmentation='none', initialization='scratch',
                  selection='highest validation Moderate 3D mAP (AP11), earliest epoch breaks ties',
                  limit_train=args.limit_train, limit_val=args.limit_val, torch=torch.__version__)
    config_path = run / 'effective_config.json'
    text = json.dumps(config, indent=2) + '\n'
    if config_path.exists() and config_path.read_text() != text:
        raise RuntimeError('%s holds a run with a different configuration' % run)
    config_path.write_text(text)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(args.model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epoch = cursor = step = 0
    best = -float('inf')
    last = run / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, step, best = state['epoch'], state['cursor'], state['step'], state['best']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        print('Resumed at epoch %d, step %d' % (epoch + 1, step), flush=True)

    train = SE3DFrames(args.data_root, 'train', label_dir=label_dir, generate_target=True, conditions=args.conditions)
    validation = SE3DFrames(args.data_root, 'val', label_dir=label_dir, generate_target=False, conditions=args.conditions)
    if args.limit_val:
        validation = torch.utils.data.Subset(validation, range(min(args.limit_val, len(validation))))
        validation.collate_fn = validation.dataset.collate_fn

    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, cursor=cursor,
                         step=step, best=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), config=config), last)

    while epoch < args.epochs:
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        if args.limit_train:
            order = order[:args.limit_train]
        for batch in loader(train, order[cursor:], args.workers, args.seed + epoch):
            started = time.monotonic()
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
                record = dict(epoch=epoch + 1, cursor=cursor, total=len(order), step=step, loss=float(loss.detach()),
                              depth_loss=float(depth_loss.mean().detach()), detection_loss=float(detection_loss.detach()),
                              batch_seconds=time.monotonic() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                print(json.dumps(record), flush=True)
                with (run / 'training.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
            if step % 200 == 0:
                save()
        save()
        result, _ = evaluate(model, validation, args.workers)
        score = result['3d_mAP11'][1]
        if score is None or not np.isfinite(score):
            raise ValueError('Undefined validation mAP after epoch %d' % (epoch + 1))
        result['epoch'] = epoch + 1
        result['optimizer_steps'] = step
        (run / ('validation_epoch_%03d.json' % (epoch + 1))).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(epoch=epoch + 1, validation_3d_mAP11_moderate=score, best=max(best, score))), flush=True)
        if score > best:
            best = score
            atomic_save(dict(model=model.state_dict(), epoch=epoch + 1, step=step, validation_score=best,
                             config=config), run / 'best.pth')
        epoch += 1
        cursor = 0
        save()
    (run / 'training_complete.json').write_text(
        json.dumps(dict(epochs=epoch, steps=step, best_validation_3d_mAP11_moderate=best), indent=2) + '\n')


if __name__ == '__main__':
    main()
