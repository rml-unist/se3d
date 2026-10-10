"""Evaluate a checkpoint on an SE3D split (default: the 3,991 test frames).

Writes metrics.json (overall and per condition) and predictions.pkl to --output.
With --save-kitti, it also writes one KITTI-format file per frame under
<output>/kitti/<map>/<sequence>/<frame>.txt, the format read by tools/evaluate.py.

    python tools/test.py --model dsgn_event --checkpoint weights/se3d_dsgn_event_40ep.pth \
        --data-root /data/SE3D --output results/dsgn_event
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from se3d.backends import restore_checkpoint_backend
from se3d.data import SE3DFrames
from se3d.engine import report_by_condition, predict
from se3d.models import ANCHORS, MODELS, anchor_name, build_model, configure_classes
from se3d.protocol import add_labels_argument


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The released checkpoints record the sha256 of the statistics file their anchors came from.
KNOWN_ANCHORS = {
    '6e95f124a010e5f618c85b36dc8f783c6d2dd565c1fb013ea2129873260717ba': 'label_original',
    'aee2ec2d0351af6e1cafdd10c7a463f1804f8daf990d4ad26d0f63a8e5d003b8': 'label_original_sunny',
    '8253bd7cd354664ec323788609ebc3d166ac476caa9356a5dddec3a74ef8e5d2': 'label',
}


def anchors_for(checkpoint, requested):
    if requested:
        return ANCHORS[requested]
    config = checkpoint.get('config', {})
    by_name = {path.name: path for path in ANCHORS.values()}
    if config.get('anchors') in by_name:
        path = by_name[config['anchors']]
        if config.get('anchors_sha256') and sha256(path) != config['anchors_sha256']:
            raise ValueError('Checkpoint anchor hash does not match %s' % path)
        return path
    if config.get('anchors_sha256') in KNOWN_ANCHORS:
        return ANCHORS[KNOWN_ANCHORS[config['anchors_sha256']]]
    raise ValueError('Cannot tell which anchors this checkpoint was trained with; pass --anchors')


def write_kitti(prediction, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(len(prediction['name'])):
        l, h, w = prediction['dimensions'][i]
        x, y, z = prediction['location'][i]
        x1, y1, x2, y2 = prediction['bbox'][i]
        lines.append('%s -1 -1 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' % (
            prediction['name'][i], prediction['alpha'][i], x1, y1, x2, y2, h, w, l, x, y, z,
            prediction['rotation_y'][i], prediction['score'][i]))
    path.write_text('\n'.join(lines) + ('\n' if lines else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='test', choices=['test', 'val', 'test_car_filtered'])
    add_labels_argument(parser)
    parser.add_argument('--anchors', type=anchor_name, choices=sorted(ANCHORS), default=None,
                        help='anchors used in training (detected automatically for the released checkpoints)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cache-root', help='writable event cache root, separate from the dataset')
    parser.add_argument('--cache-time-bounds', help='verified time-bound metadata for cache-only input')
    parser.add_argument('--save-kitti', action='store_true')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    backend, backend_source = restore_checkpoint_backend(checkpoint.get('config', {}))
    configure_classes(anchors_for(checkpoint, args.anchors))
    config = checkpoint.get('config', {})
    seed = int(config.get('seed', 20260909))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = build_model(args.model).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    dataset = SE3DFrames(args.data_root, args.split, label_dir=args.labels, generate_target=False,
                         cache_root=args.cache_root, cache_time_bounds=args.cache_time_bounds)
    out = predict(model, dataset, args.workers)
    report = report_by_condition(out['gt'], out['predictions'], out['depth_sums'],
                                 [m['sequence'] for m in out['metadata']])
    report.update(split=args.split, labels=args.labels, checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha256(args.checkpoint), selected_epoch=checkpoint.get('epoch'),
                  backend=backend, backend_source=backend_source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'predictions.pkl').open('wb') as f:
        pickle.dump(dict(metadata=out['metadata'], predictions=out['predictions'], depth_sums=out['depth_sums']), f,
                    protocol=4)
    if args.save_kitti:
        for meta, prediction in zip(out['metadata'], out['predictions']):
            write_kitti(prediction, output / 'kitti' / meta['sequence'] / ('%06d.txt' % meta['frame']))
    print(json.dumps({'frames': report['frames'], '3d_mAP40_moderate': report['3d_mAP40'][1],
                      'disparity_MAE': report['depth']['MAE']}))


if __name__ == '__main__':
    main()
