"""Shared pieces of the DSEC-3DOD transfer experiments (paper Section VI).

Keyframes, the internal model-selection split and the target anchors are fixed
by protocol/dsec_paired_v2.json and protocol/dsec_training_anchors_v2.json.
The protocol stores the cluster paths it was built from; the scripts replace
them with --dsec-root and --labels-root.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

TRANSFER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRANSFER_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
import se3d  # noqa: E402,F401  (puts emod/ and dsgn_event/ on sys.path)
sys.path.insert(0, str(TRANSFER_ROOT / 'se_cff'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402

PROTOCOL = TRANSFER_ROOT / 'protocol' / 'dsec_paired_v2.json'
ANCHORS = TRANSFER_ROOT / 'protocol' / 'dsec_training_anchors_v2.json'
MODELS = ('emod', 'dsgn_event', 'se_cff')
SPLIT_SIZES = {'train': 3906, 'validation': 434, 'test': 1178}
# Tensors not copied from the SE3D checkpoint: the class, box and centerness
# prediction layers of the detectors, and the whole detection branch of EMOD
# when its stereo network initializes SE-CFF.
RESET_PREFIXES = {
    'emod': tuple('object_detection_net.' + n for n in ('bbox_cls.', 'bbox_reg.', 'bbox_centerness.')),
    'dsgn_event': tuple('network.' + n for n in ('bbox_cls.', 'bbox_reg.', 'bbox_centerness.')),
    'se_cff': ('object_detection_net.',),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    """Vehicle/Pedestrian/Cyclist with anchors from the target training labels; call before anything else."""
    from configs.se3d_revision import configure_dsec_classes
    return configure_dsec_classes(ANCHORS)


def dataset(split, dsec_root, labels_root, cache_root=None, generate_target=False):
    from lib.datasets.dsec_original import DSECKeyframeDataset
    ds = DSECKeyframeDataset(PROTOCOL, split, generate_target, cache_root)
    ds.original_root = Path(dsec_root)
    ds.labels_root = Path(labels_root)
    if len(ds) != SPLIT_SIZES[split]:
        raise ValueError('Unexpected %s size %d' % (split, len(ds)))
    return ds


def build_model(name):
    if name in ('emod', 'dsgn_event'):
        from se3d.models import build_model as build_joint
        return build_joint(name)
    if name == 'se_cff':
        from components.models import EventStereoMatchingNetwork
        conf = EasyDict(yaml.safe_load((TRANSFER_ROOT / 'se_cff' / 'config.yaml').read_text()))
        return EventStereoMatchingNetwork(**conf.MODEL.PARAMS)
    raise ValueError(name)


def initialize_from_se3d(model, name, checkpoint_path):
    """Copy every tensor of the SE3D checkpoint except the reset prefixes; returns the copied names."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    target = model.state_dict()
    prefixes = RESET_PREFIXES[name]
    transfer = {k: v for k, v in state.items() if not k.startswith(prefixes)}
    if any(k not in target or target[k].shape != v.shape for k, v in transfer.items()):
        raise ValueError('The SE3D checkpoint does not match the %s architecture' % name)
    if set(transfer) != {k for k in target if not k.startswith(prefixes)}:
        raise ValueError('The SE3D checkpoint lacks shared tensors of %s' % name)
    target.update(transfer)
    model.load_state_dict(target, strict=True)
    return sorted(transfer)


def depth_args(batch, training=True, device='cuda'):
    # EMOD layout (b, s, t, h, w, c) -> SE-CFF layout (b, c, h, w, t, s); values unchanged.
    return dict(left_event=batch['event']['left'].permute(0, 5, 3, 4, 2, 1).to(device),
                right_event=batch['event']['right'].permute(0, 5, 3, 4, 2, 1).to(device),
                gt_disparity=batch['disparity'].to(device) if training else None)


def camera_annotation_to_lidar(annotation, event_to_lidar):
    """Predicted KITTI-style boxes (left event camera) -> DSEC-3DOD LiDAR boxes (x, y, z, l, w, h, heading)."""
    t = np.asarray(event_to_lidar)
    dims = np.asarray(annotation['dimensions']).reshape(-1, 3)  # l, h, w
    centers = np.asarray(annotation['location']).reshape(-1, 3).copy()
    centers[:, 1] -= dims[:, 1] / 2
    centers = centers @ t[:3, :3].T + t[:3, 3]
    ry = np.asarray(annotation['rotation_y'])
    direction = np.stack((np.cos(ry), np.zeros_like(ry), -np.sin(ry)), axis=1) @ t[:3, :3].T
    heading = np.arctan2(direction[:, 1], direction[:, 0])
    boxes = np.concatenate((centers, dims[:, [0, 2, 1]], heading[:, None]), axis=1)
    if not np.isfinite(boxes).all():
        raise ValueError('Nonfinite prediction geometry')
    return dict(name=annotation['name'], boxes_lidar=boxes, score=annotation['score'])


def _depth_summary(sums):
    absolute, square, one, two, count = sums
    return dict(valid_pixels=int(count), MAE=float(absolute / count) if count else None,
                RMSE=float(np.sqrt(square / count)) if count else None,
                one_pixel_error_percent=float(100 * one / count) if count else None,
                two_pixel_error_percent=float(100 * two / count) if count else None)


@torch.no_grad()
def evaluate_detection(model, ds, workers, run, tag, metrics_python):
    """Predict every keyframe, write <tag>_predictions.pkl and score it with evaluate_waymo.py."""
    from configs.od_cfg import cfg
    from lib.dsgn.utils.inference3d import make_fcos3d_postprocessor
    from se3d.data import model_args
    from se3d.engine import loader, prediction_annotation
    model.eval()
    gt, pred, annotations = [], [], {}
    sums = np.zeros(5, dtype=np.float64)
    processor = make_fcos3d_postprocessor(cfg)
    for i, batch in enumerate(loader(ds, range(len(ds)), workers, 1)):
        args = model_args(batch, training=False)
        disp, det, _, _ = model(**args)
        if not torch.isfinite(disp).all() or not all(torch.isfinite(v).all() for v in det.values()):
            raise ValueError('Nonfinite evaluation prediction')
        boxes = processor(det['bbox_cls'], det['bbox_reg'], det['bbox_centerness'], image_sizes=(480, 640),
                          calibs_Proj=args['calibs_Proj'])[0][0]
        row = batch['metadata'][0]
        path = ds.labels_root / row['annotation_path']
        if str(path) not in annotations:
            annotations[str(path)] = pickle.loads(path.read_bytes())
        a = annotations[str(path)][row['frame']]
        if int(a['time_stamp']) != row['timestamp_us']:
            raise ValueError('GT timestamp mismatch')
        gt.append({k: a['annos'][k].copy() for k in ('name', 'gt_boxes_lidar', 'num_points_in_gt')})
        pred.append(camera_annotation_to_lidar(prediction_annotation(boxes), a['image']['image_0_extrinsic']))
        valid = (args['gt_disparity'] > 0) & torch.isfinite(args['gt_disparity'])
        err = (disp - args['gt_disparity']).abs()[valid].double()
        sums += np.array([err.sum().item(), err.square().sum().item(), (err > 1).sum().item(),
                          (err > 2).sum().item(), err.numel()])
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(evaluation=tag, frames=i + 1, total=len(ds))), flush=True)
    inputs = Path(run) / ('%s_predictions.pkl' % tag)
    inputs.write_bytes(pickle.dumps(dict(gt=gt, predictions=pred, depth=_depth_summary(sums)), protocol=4))
    output = Path(run) / ('%s_metrics.json' % tag)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='2')
    subprocess.run([str(metrics_python), str(TRANSFER_ROOT / 'evaluate_waymo.py'), str(inputs), str(output)],
                   check=True, env=env, timeout=1800)
    return json.loads(output.read_text())


@torch.no_grad()
def evaluate_depth(model, ds, workers, run, tag):
    from se3d.engine import loader
    model.eval()
    sums = np.zeros(5, dtype=np.float64)
    for i, batch in enumerate(loader(ds, range(len(ds)), workers, 1)):
        disp, _ = model(**depth_args(batch, False))
        gt = batch['disparity'].to(disp.device)
        if not torch.isfinite(disp).all():
            raise ValueError('Nonfinite depth evaluation')
        valid = (gt > 0) & torch.isfinite(gt)
        err = (disp - gt).abs()[valid].double()
        sums += np.array([err.sum().item(), err.square().sum().item(), (err > 1).sum().item(),
                          (err > 2).sum().item(), err.numel()])
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(evaluation=tag, frames=i + 1, total=len(ds))), flush=True)
    if not sums[4]:
        raise ValueError('No valid depth GT')
    result = dict(frames=len(ds), depth=_depth_summary(sums))
    (Path(run) / ('%s_metrics.json' % tag)).write_text(json.dumps(result, indent=2) + '\n')
    return result
