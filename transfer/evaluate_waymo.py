"""Waymo-style 3D AP on DSEC-3DOD predictions written by transfer/train.py or transfer/test.py.

Run this in the metrics environment (TensorFlow 2.12 and waymo-open-dataset,
see requirements-metrics.txt), separate from the PyTorch environment:

    python transfer/evaluate_waymo.py runs/dsgn_event_se3d/fixed_test_predictions.pkl metrics.json

The input holds per-keyframe ground truth (Vehicle, Pedestrian, Cyclist boxes
in the DSEC-3DOD LiDAR frame with point counts), predictions in the same
frame, and the disparity error summary. Level 2 AP uses IoU 0.7 for Vehicle
and 0.5 for Pedestrian and Cyclist within 100 m. The reported detection score
is the mean of Vehicle and Pedestrian Level 2 AP (V/P AP).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendor.waymo_eval_detection import WaymoDetectionMetricsEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions')
    parser.add_argument('output')
    args = parser.parse_args()
    data = pickle.loads(Path(args.predictions).read_bytes())
    metrics = WaymoDetectionMetricsEstimator().waymo_evaluation(
        data['predictions'], data['gt'], ['Vehicle', 'Pedestrian', 'Cyclist'], fake_gt_infos=False)
    result = {k: float(np.asarray(v).reshape(-1)[0]) * 100 for k, v in metrics.items()}
    keys = ['OBJECT_TYPE_TYPE_%s_LEVEL_2/AP' % name for name in ('VEHICLE', 'PEDESTRIAN')]
    result = {'metrics_percent': result,
              'selection_vehicle_pedestrian_L2_AP': float(np.mean([result[k] for k in keys])),
              'frames': len(data['gt']), 'depth': data['depth'],
              'definition': 'Official Waymo 3D AP/APH, native annotation LiDAR frame, IoU Vehicle .7 '
                            'Pedestrian/Cyclist .5, keyframes only, range100m, no blind-time benchmark'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
