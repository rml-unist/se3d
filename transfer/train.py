"""Paired target training on DSEC-3DOD: scratch versus SE3D initialization.

Both arms of a pair use the same architecture, target data, 16 epochs
(62,496 updates), batch size 1, Adam (lr 1e-4, weight decay 1e-4), no
augmentation and the same seed. The SE3D arm copies every tensor of the SE3D
checkpoint except the detection prediction layers, which keep their seeded
initialization. After every epoch the model is scored on the 434 internal
validation keyframes; best.pth keeps the epoch with the highest V/P Level-2 AP
(lowest disparity MAE for SE-CFF).

    # DSGN-event, seed 20260909 (the paper also uses 20260910 and 20260911)
    python transfer/train.py --model dsgn_event --init scratch --seed 20260909 --output runs/dsec_dsgn_scratch_s0909 ...
    python transfer/train.py --model dsgn_event --init se3d --source weights/se3d_dsgn_event_8ep.pth \
        --seed 20260909 --output runs/dsec_dsgn_se3d_s0909 ...
    # EMOD and SE-CFF (both initialized from the 8-epoch EMOD checkpoint)
    python transfer/train.py --model emod   --init se3d --source weights/se3d_emod_8ep.pth --output ... ...
    python transfer/train.py --model se_cff --init se3d --source weights/se3d_emod_8ep.pth --output ... ...

Common arguments: --dsec-root <DSEC>/train --labels-root <DSEC-3DOD> [--cache-root <cache>]
--metrics-python <python of the metrics environment> (not needed for se_cff).
An interrupted run restarts from last.pth with the same command.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def atomic_save(obj, path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=common.MODELS, required=True)
    parser.add_argument('--init', choices=['scratch', 'se3d'], required=True)
    parser.add_argument('--source', help='SE3D checkpoint for --init se3d')
    parser.add_argument('--dsec-root', required=True, help='DSEC train/ directory (events, disparity, calibration)')
    parser.add_argument('--labels-root', required=True, help='DSEC-3DOD directory with <chunk>/<chunk>_fov_bbox_lidar_check.pkl')
    parser.add_argument('--cache-root', default=None, help='event stacks from prepare_cache.py (optional)')
    parser.add_argument('--metrics-python', default=None, help='python of the metrics environment')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-train', type=int, default=None, help='debug: keyframes per epoch')
    parser.add_argument('--limit-val', type=int, default=None, help='debug: validation keyframes')
    args = parser.parse_args()
    if args.init == 'se3d' and not args.source:
        parser.error('--init se3d requires --source')
    detection = args.model != 'se_cff'
    if detection and not args.metrics_python:
        parser.error('--metrics-python is required to select detection checkpoints')

    common.configure()
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, init=args.init, epochs=args.epochs, seed=args.seed, batch_size=1, optimizer='Adam',
                  learning_rate=1e-4, weight_decay=1e-4, augmentation='none',
                  loss_weights=[.5, .5] if detection else 'SE-CFF weighted pyramid SmoothL1, valid-pixel mean',
                  deform_offset_learning_rate=None if detection else 1e-5,
                  training_pool_frames=3906, validation_frames=434, test_frames=1178,
                  selection='Max validation mean Vehicle/Pedestrian official L2 AP; earliest tie' if detection
                  else 'Minimum validation disparity MAE; earliest tie',
                  protocol_sha256=common.sha256(common.PROTOCOL), anchors_sha256=common.sha256(common.ANCHORS),
                  pretrained_checkpoint_sha256=common.sha256(args.source) if args.init == 'se3d' else None,
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
    model = common.build_model(args.model).cuda()
    last = run / 'last.pth'
    if args.init == 'se3d' and not last.exists():
        copied = common.initialize_from_se3d(model, args.model, args.source)
        (run / 'transfer_tensors.json').write_text(json.dumps(copied, indent=2) + '\n')
    if detection:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:  # SE-CFF: lr 1e-4, and 1e-5 for the deformable-convolution offsets
        optimizer = torch.optim.Adam(model.get_params_group(1e-4), weight_decay=1e-4)
    epoch = cursor = step = 0
    best = -float('inf')
    if last.exists():
        state = torch.load(last, map_location='cuda', weights_only=False)
        if state['config'] != config:
            raise RuntimeError('last.pth belongs to a different configuration')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, step, best = state['epoch'], state['cursor'], state['step'], state['best']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
    roots = dict(dsec_root=args.dsec_root, labels_root=args.labels_root, cache_root=args.cache_root)
    train = common.dataset('train', generate_target=detection, **roots)
    validation = common.dataset('validation', generate_target=False, **roots)
    if args.limit_val:
        validation.rows = validation.rows[:args.limit_val]

    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, cursor=cursor,
                         step=step, best=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), config=config), last)

    from se3d.data import model_args
    from se3d.engine import loader
    while epoch < args.epochs:
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        if args.limit_train:
            order = order[:args.limit_train]
        for batch in loader(train, order[cursor:], args.workers, args.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            if detection:
                _, _, depth_loss, detection_loss = model(**model_args(batch))
                loss = .5 * depth_loss.mean() + .5 * detection_loss
            else:
                _, loss_vector = model(**common.depth_args(batch))
                loss = loss_vector.mean() if loss_vector.numel() else loss_vector.sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite target training loss at step %d' % step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            cursor += 1
            step += 1
            if step % 20 == 0:
                record = dict(epoch=epoch + 1, cursor=cursor, total=len(train), step=step, loss=float(loss.detach()))
                if detection:
                    record.update(depth_loss=float(depth_loss.mean().detach()), detection_loss=float(detection_loss.detach()))
                print(json.dumps(record), flush=True)
                with (run / 'training.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
            if step % 200 == 0:
                save()
        save()
        tag = 'validation_epoch_%03d' % (epoch + 1)
        if detection:
            result = common.evaluate_detection(model, validation, args.workers, run, tag, args.metrics_python)
            score = result['selection_vehicle_pedestrian_L2_AP']
        else:
            result = common.evaluate_depth(model, validation, args.workers, run, tag)
            score = -result['depth']['MAE']
        if not np.isfinite(score):
            raise ValueError('Undefined model selection after epoch %d' % (epoch + 1))
        result.update(epoch=epoch + 1, step=step)
        (run / ('%s_metrics.json' % tag)).write_text(json.dumps(result, indent=2) + '\n')
        if score > best:
            best = score
            atomic_save(dict(model=model.state_dict(), epoch=epoch + 1, validation_score=best, config=config),
                        run / 'best.pth')
        epoch += 1
        cursor = 0
        save()
    summary = dict(epochs=epoch, steps=step)
    summary['best_validation_L2_AP' if detection else 'best_validation_MAE'] = best if detection else -best
    (run / 'training_complete.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
