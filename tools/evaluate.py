"""Score 3D detections and disparity maps on an SE3D split.

Detections: one KITTI-format file per frame with a score in the 16th column,
    <kitti-dir>/<map>/<sequence>/<frame>.txt   e.g. map1/map1_night_sunny_moving/001657.txt
Boxes are in the left event-camera frame (P2 of calib.txt); the 2D box is in
the 640x480 event image. Disparity (optional): one float32 array of shape
480x640 per frame, in pixels of the left event camera,
    <disparity-dir>/<map>/<sequence>/<frame>.npy
A predictions.pkl written by tools/test.py can be scored instead.

Detection uses Moderate AP with 40 recall positions (AP11 is also reported),
IoU 0.7 for Car, Truck, Van and Bus and 0.5 for Pedestrian, Bicycle and
Motorcycle, and the SE3D difficulty levels (2D height > 40/25/15 px,
occlusion <= 0/1/2). Disparity errors are computed over pixels with a valid
ground truth.

    python tools/evaluate.py --data-root /data/SE3D --kitti-dir my_method/kitti \
        --disparity-dir my_method/disparity --output my_method/metrics.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from se3d.protocol import add_labels_argument, load_splits
from se3d.models import ANCHORS, configure_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='test', choices=['test', 'val', 'test_car_filtered'])
    add_labels_argument(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--kitti-dir', help='per-frame KITTI detection files')
    source.add_argument('--predictions', help='predictions.pkl from tools/test.py')
    parser.add_argument('--disparity-dir', help='per-frame disparity .npy files (with --kitti-dir)')
    parser.add_argument('--output', required=True, help='metrics JSON')
    args = parser.parse_args()

    configure_classes(ANCHORS['label'])  # sets the seven evaluation classes; anchors are unused here
    from se3d.engine import disparity_sums, report_by_condition
    from utils.kitti_common import get_label_anno

    root = Path(args.data_root)
    frames = [(name, t, f) for name, items in sorted(load_splits()['frames'][args.split].items()) for t, f in items]
    gt = [get_label_anno(str(root / name / args.labels / ('%06d.txt' % f))) for name, _, f in frames]
    if args.predictions:
        saved = pickle.loads(Path(args.predictions).read_bytes())
        by_key = {(m['sequence'], m['timestamp']): i for i, m in enumerate(saved['metadata'])}
        missing = [(n, t) for n, t, _ in frames if (n, t) not in by_key]
        if missing:
            raise ValueError('%d frames of the split are not in %s, e.g. %s' % (len(missing), args.predictions, missing[:3]))
        predictions = [saved['predictions'][by_key[(n, t)]] for n, t, _ in frames]
        depth = [saved['depth_sums'][by_key[(n, t)]] for n, t, _ in frames]
    else:
        predictions, depth = [], []
        for name, _, f in frames:
            path = Path(args.kitti_dir) / name / ('%06d.txt' % f)
            if not path.is_file():
                raise FileNotFoundError('Missing detection file %s (write an empty file for no detections)' % path)
            predictions.append(get_label_anno(str(path)))
            if args.disparity_dir:
                prediction = np.load(Path(args.disparity_dir) / name / ('%06d.npy' % f)).astype(np.float32)
                target = np.load(root / name / 'disparity' / 'event' / ('%06d.npy' % f)).astype(np.float32)
                if prediction.shape != target.shape:
                    raise ValueError('Disparity of %s/%06d has shape %s, expected %s' % (name, f, prediction.shape, target.shape))
                depth.append(disparity_sums(torch.from_numpy(prediction), torch.from_numpy(target)))
            else:
                depth.append([0, 0, 0, 0, 0])
    report = report_by_condition(gt, predictions, depth, [name for name, _, _ in frames])
    report.update(split=args.split, labels=args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    per_class = {name: values['3d_AP40'][1] for name, values in report['per_class'].items()}
    print(json.dumps({'frames': report['frames'], '3d_mAP40_moderate': report['3d_mAP40'][1],
                      'per_class_3d_AP40_moderate': per_class, 'disparity_MAE': report['depth']['MAE']}, indent=1))


if __name__ == '__main__':
    main()
