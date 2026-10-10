"""Inference over a split and the SE3D metrics (detection AP and disparity errors)."""
import json

import numpy as np
import torch

from . import CONDITIONS
from .data import condition_of, model_args
from configs.od_cfg import cfg
from lib.dsgn.utils.inference3d import make_fcos3d_postprocessor
from utils.evalod import get_dimensions
from utils.kitti_common import get_label_anno
from utils.se3d_evaluation import evaluate_se3d


def loader(dataset, indices, workers, seed):
    return torch.utils.data.DataLoader(dataset, batch_size=1, sampler=indices, num_workers=workers,
                                       collate_fn=dataset.collate_fn, pin_memory=True,
                                       generator=torch.Generator().manual_seed(seed))


def prediction_annotation(boxlist, class_names=None):
    """Convert post-processed boxes to a KITTI-style annotation dictionary."""
    class_names = cfg.class_names if class_names is None else class_names
    corners = boxlist.get_field('box_corner3d').detach().cpu()
    centers = corners.mean(dim=1)
    dimensions, rotations = [], []
    for cor, center in zip(corners, centers):
        h, w, l, ry = get_dimensions((cor - center).T)
        dimensions.append([l, h, w])
        rotations.append(ry)
    dimensions = np.array(dimensions, dtype=np.float64).reshape(-1, 3)
    centers = centers.numpy().reshape(-1, 3)
    if len(centers):
        centers[:, 1] += dimensions[:, 1] / 2
    rotations = np.asarray(rotations, dtype=np.float64)
    return dict(name=np.array([class_names[int(i) - 1] for i in boxlist.get_field('labels').cpu()]),
                bbox=boxlist.bbox.detach().cpu().numpy().reshape(-1, 4), dimensions=dimensions,
                location=centers, rotation_y=rotations, alpha=rotations - np.arctan2(centers[:, 0], centers[:, 2]),
                score=boxlist.get_field('scores').detach().cpu().numpy())


def disparity_sums(disparity, gt_disparity):
    """[sum |e|, sum e^2, #|e|>1, #|e|>2, #valid] over pixels with a valid ground truth."""
    valid = (gt_disparity > 0) & torch.isfinite(gt_disparity)
    error = (disparity - gt_disparity).abs()[valid].double()
    if not torch.isfinite(error).all():
        raise ValueError('Non-finite disparity prediction')
    return [error.sum().item(), (error * error).sum().item(), (error > 1).sum().item(),
            (error > 2).sum().item(), error.numel()]


def summarize(gt, predictions, depth_sums, indices):
    """Detection AP (all classes) and disparity errors over the frames in indices."""
    report = evaluate_se3d([gt[i] for i in indices], [predictions[i] for i in indices])
    absolute, square, over1, over2, count = np.sum([depth_sums[i] for i in indices], axis=0)
    report['frames'] = len(indices)
    report['depth'] = dict(valid_pixels=int(count),
                           MAE=float(absolute / count) if count else None,
                           RMSE=float(np.sqrt(square / count)) if count else None,
                           one_pixel_error_percent=float(100 * over1 / count) if count else None,
                           two_pixel_error_percent=float(100 * over2 / count) if count else None)
    return report


def report_by_condition(gt, predictions, depth_sums, sequences):
    groups = {}
    for index, sequence in enumerate(sequences):
        groups.setdefault(condition_of(sequence), []).append(index)
    report = summarize(gt, predictions, depth_sums, list(range(len(gt))))
    report['by_condition'] = {name: summarize(gt, predictions, depth_sums, indices)
                              for name, indices in sorted(groups.items())}
    report['absent_conditions'] = sorted(set(CONDITIONS) - set(groups))
    return report


@torch.no_grad()
def predict(model, dataset, workers=2, log_every=100, check_stop=None):
    """Run a model over every frame; returns predictions, ground truth and disparity sums."""
    model.eval()
    processor = make_fcos3d_postprocessor(cfg)
    out = dict(metadata=[], predictions=[], gt=[], depth_sums=[])
    for index, batch in enumerate(loader(dataset, range(len(dataset)), workers, 1)):
        if check_stop is not None:
            check_stop()
        args = model_args(batch, training=False)
        disparity, detection, _, _ = model(**args)
        if not all(torch.isfinite(v).all() for v in detection.values()):
            raise ValueError('Non-finite detection output')
        boxes = processor(detection['bbox_cls'], detection['bbox_reg'], detection['bbox_centerness'],
                          image_sizes=(480, 640), calibs_Proj=args['calibs_Proj'])[0][0]
        out['predictions'].append(prediction_annotation(boxes))
        out['gt'].append(get_label_anno(batch['metadata'][0]['label_path']))
        out['depth_sums'].append(disparity_sums(disparity, args['gt_disparity']))
        out['metadata'].append(batch['metadata'][0])
        if log_every and (index + 1) % log_every == 0:
            print(json.dumps(dict(frames=index + 1, total=len(dataset))), flush=True)
    return out


def evaluate(model, dataset, workers=2, check_stop=None):
    out = predict(model, dataset, workers, check_stop=check_stop)
    return report_by_condition(out['gt'], out['predictions'], out['depth_sums'],
                               [m['sequence'] for m in out['metadata']]), out
