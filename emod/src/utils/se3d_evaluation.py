"""Revision metrics built on the transferred KITTI-style matching implementation.

Difficulty thresholds: 2D heights 40/25/15 px (strict greater-than for GT),
occlusion 0/1/2, truncation <=1 (no limit). AP40 is the reported metric; AP11
from the same 41-position precision envelope selects checkpoints on validation.
These are SE3D criteria, not the official KITTI difficulty definition.
"""
import numpy as np
from configs.od_cfg import cfg
from configs.se3d_revision import CLASSES
from . import evalod

IOU = [0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7]


def evaluate_se3d(gt, predictions, classes=CLASSES, legacy_aliases=False):
    if len(gt) != len(predictions):
        raise ValueError('Ground truth and prediction frame counts differ')
    previous = getattr(cfg, 'evaluation_class_names', None)
    previous_aliases = getattr(cfg, 'evaluation_ignore_aliases', None)
    cfg.evaluation_class_names = list(classes)
    cfg.evaluation_ignore_aliases = legacy_aliases
    try:
        thresholds = np.asarray([IOU[CLASSES.index(c)] for c in classes])
        overlaps = np.broadcast_to(thresholds, (1, 3, len(classes))).copy()
        metrics = evalod.do_eval_v3(gt, predictions, list(range(len(classes))),
                                   overlaps, compute_aos=True, difficultys=(0, 1, 2))
        result = dict(protocol='SE3D-custom-height40-25-15-truncation1-AP11+AP40-v1',
                      classes=list(classes), iou=thresholds.tolist(), frames=len(gt), legacy_ignore_aliases=legacy_aliases, per_class={})
        for i, name in enumerate(classes):
            counts = [sum(evalod.clean_data(a, b, i, d)[0] for a, b in zip(gt, predictions))
                      for d in range(3)]
            values = dict(valid_gt=counts)
            for metric in ('bbox', 'bev', '3d'):
                precision = metrics[metric]['precision'][i, :, 0]
                for ap, indices in (('AP11', list(range(0, 41, 4))), ('AP40', list(range(1, 41)))):
                    scores = precision[:, indices].mean(axis=1) * 100
                    values[metric + '_' + ap] = [float(v) if counts[d] else None for d, v in enumerate(scores)]
            result['per_class'][name] = values
        for metric in ('bbox', 'bev', '3d'):
            for ap in ('AP11', 'AP40'):
                result[metric + '_m' + ap] = []
                for difficulty in range(3):
                    scores = [v[metric + '_' + ap][difficulty] for v in result['per_class'].values()
                              if v[metric + '_' + ap][difficulty] is not None]
                    result[metric + '_m' + ap].append(float(np.mean(scores)) if scores else None)
        return result
    finally:
        if previous_aliases is None:
            del cfg['evaluation_ignore_aliases']
        else:
            cfg.evaluation_ignore_aliases = previous_aliases
        if previous is None:
            del cfg['evaluation_class_names']
        else:
            cfg.evaluation_class_names = previous

