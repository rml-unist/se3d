"""Shared pieces of the DSEC-3DOD transfer experiments (paper Section VI).

Keyframes, the internal model-selection split and the target anchors are fixed
by protocol/dsec_paired_v2.json and protocol/dsec_training_anchors_v2.json.
The protocol stores the cluster paths it was built from; the scripts replace
them with --dsec-root and --labels-root.
"""
import json
import os
import pickle
import subprocess
import sys
import time
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
try:
    from . import protocol_utils, runtime
except ImportError:
    import protocol_utils
    import runtime

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


sha256 = protocol_utils.sha256


def configure(anchors=ANCHORS):
    """Use validated target anchors, including a checkpoint's embedded payload."""
    from configs.od_cfg import cfg
    payload = anchors if isinstance(anchors, dict) else json.loads(Path(anchors).read_text())
    values = protocol_utils.validate_anchor_dimensions(payload)
    cfg.class_names = list(protocol_utils.CLASSES)
    cfg.num_classes = len(cfg.class_names)
    cfg.valid_classes = list(range(1, cfg.num_classes + 1))
    for key, field in [('ANCHORS_HEIGHT', 'height'), ('ANCHORS_WIDTH', 'width'),
                       ('ANCHORS_LENGTH', 'length'), ('ANCHORS_Y', 'center_y')]:
        setattr(cfg.RPN3D, key, [values[name][field] for name in cfg.class_names])
    return cfg


def dataset(split, dsec_root, labels_root, cache_root=None, generate_target=False,
            protocol_path=PROTOCOL, train_rows=None):
    from lib.datasets.dsec_original import DSECKeyframeDataset
    ds = DSECKeyframeDataset(protocol_path, split, generate_target, cache_root)
    ds.original_root = Path(dsec_root)
    ds.labels_root = Path(labels_root)
    if len(ds) != SPLIT_SIZES[split]:
        raise ValueError('Unexpected %s size %d' % (split, len(ds)))
    if train_rows is not None:
        if split != 'train' or not train_rows:
            raise ValueError('Only the training split can be restricted to a nonempty subset')
        pool = {protocol_utils.frame_key(row): row for row in ds.rows}
        if any(pool.get(protocol_utils.frame_key(row)) != row for row in train_rows):
            raise ValueError('Subset rows are not in the protocol training pool')
        # The constructor reads no annotations. Restrict before the first __getitem__.
        ds.rows = list(train_rows)
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


def initialize_from_se3d(model, name, checkpoint_path=None, source_state=None):
    """Copy every tensor of the SE3D checkpoint except the reset prefixes; returns the copied names."""
    state = (torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
             if source_state is None else source_state)
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


def _depth_sums(disp, gt):
    if not torch.isfinite(disp).all():
        raise ValueError('Nonfinite depth evaluation')
    valid = (gt > 0) & torch.isfinite(gt)
    err = (disp - gt).abs()[valid].double()
    return np.array([err.sum().item(), err.square().sum().item(), (err > 1).sum().item(),
                     (err > 2).sum().item(), err.numel()], dtype=np.float64)


def evaluation_loader(ds, indices, workers):
    from se3d.engine import loader
    return loader(ds, indices, workers, 1)


def _evaluation_binding(model, ds, context):
    from configs.od_cfg import cfg
    return dict(model_tensors_sha256=runtime.tensor_state_sha256(model.state_dict()),
                rows_sha256=protocol_utils.json_digest(ds.rows),
                anchor_dimensions={key: list(getattr(cfg.RPN3D, key)) for key in
                                   ('ANCHORS_HEIGHT', 'ANCHORS_WIDTH', 'ANCHORS_LENGTH', 'ANCHORS_Y')},
                context=context if context is not None else
                dict(code_sha256=runtime.code_fingerprint(REPO_ROOT)['sha256']))


def _collect(model, ds, workers, run, tag, detection, predict_frame, context, check_stop):
    model.eval()
    binding = _evaluation_binding(model, ds, context)
    progress = runtime.EvaluationProgress(Path(run) / (tag + '_progress.pkl'), binding, len(ds), detection)
    try:
        check_stop()
        for i, batch in enumerate(evaluation_loader(ds, range(progress.cursor, len(ds)), workers), progress.cursor):
            check_stop()
            if batch['metadata'][0] != ds.rows[i]:
                raise ValueError('Evaluation loader order differs from the protocol')
            sums, gt, pred = predict_frame(batch)
            progress.append(i, sums, gt, pred)
            if progress.cursor % 50 == 0:
                progress.save()
            if progress.cursor % 100 == 0:
                print(json.dumps(dict(evaluation=tag, frames=progress.cursor, total=len(ds))), flush=True)
            check_stop()
    except runtime.Preempted:
        progress.save()
        raise
    if progress.cursor != len(ds) or not progress.sums[4]:
        raise ValueError('Incomplete evaluation or no valid depth GT')
    progress.save()
    return progress


def _score_predictions(inputs, output, metrics_python, binding, check_stop):
    prediction_hash = sha256(inputs)
    if output.exists():
        result = json.loads(output.read_text())
        if result.get('predictions_sha256') != prediction_hash or result.get('binding') != binding:
            raise ValueError('Metric output is stale or belongs to different predictions')
        return result
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='2')
    check_stop()
    process = subprocess.Popen([str(metrics_python), str(TRANSFER_ROOT / 'evaluate_waymo.py'),
                                str(inputs), str(output)], env=env)
    started = time.monotonic()
    try:
        while process.poll() is None:
            check_stop()
            if time.monotonic() - started > 1800:
                raise TimeoutError('Waymo CPU evaluation exceeded 1800 seconds')
            time.sleep(.25)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    result = json.loads(output.read_text())
    if result.get('predictions_sha256') != prediction_hash or result.get('binding') != binding:
        raise ValueError('Waymo metric result has unexpected input provenance')
    return result


@torch.no_grad()
def evaluate_detection(model, ds, workers, run, tag, metrics_python, *, context=None,
                       check_stop=lambda: None, device='cuda'):
    """Resume a contiguous prediction prefix, then run official CPU metrics."""
    from configs.od_cfg import cfg
    from lib.dsgn.utils.inference3d import make_fcos3d_postprocessor
    from se3d.data import model_args
    from se3d.engine import prediction_annotation
    processor = make_fcos3d_postprocessor(cfg)
    annotations = {}

    def predict(batch):
        args = model_args(batch, training=False, device=device)
        disp, det, _, _ = model(**args)
        if not torch.isfinite(disp).all() or not all(torch.isfinite(v).all() for v in det.values()):
            raise ValueError('Nonfinite evaluation prediction')
        boxes = processor(det['bbox_cls'], det['bbox_reg'], det['bbox_centerness'], image_sizes=(480, 640),
                          calibs_Proj=args['calibs_Proj'])[0][0]
        row = batch['metadata'][0]
        path = ds.labels_root / row['annotation_path']
        if str(path) not in annotations:
            annotations[str(path)] = pickle.loads(path.read_bytes())
        annotation = annotations[str(path)][row['frame']]
        if int(annotation['time_stamp']) != row['timestamp_us']:
            raise ValueError('GT timestamp mismatch')
        gt = {key: annotation['annos'][key].copy() for key in ('name', 'gt_boxes_lidar', 'num_points_in_gt')}
        pred = camera_annotation_to_lidar(prediction_annotation(boxes), annotation['image']['image_0_extrinsic'])
        return _depth_sums(disp, args['gt_disparity']), gt, pred

    progress = _collect(model, ds, workers, run, tag, True, predict, context, check_stop)
    inputs = Path(run) / (tag + '_predictions.pkl')
    if inputs.exists():
        saved = pickle.loads(inputs.read_bytes())
        if (saved.get('binding') != progress.binding or len(saved['gt']) != len(ds)
                or len(saved['predictions']) != len(ds)):
            raise ValueError('Complete predictions belong to a different evaluation')
    else:
        runtime.atomic_pickle(dict(gt=progress.gt, predictions=progress.pred,
                                   depth=_depth_summary(progress.sums), binding=progress.binding), inputs)
    return _score_predictions(inputs, Path(run) / (tag + '_metrics.json'), metrics_python,
                              progress.binding, check_stop)


@torch.no_grad()
def evaluate_depth(model, ds, workers, run, tag, *, context=None, check_stop=lambda: None, device='cuda'):
    def predict(batch):
        disp, _ = model(**depth_args(batch, False, device=device))
        return _depth_sums(disp, batch['disparity'].to(disp.device)), None, None

    progress = _collect(model, ds, workers, run, tag, False, predict, context, check_stop)
    result = dict(frames=len(ds), depth=_depth_summary(progress.sums), binding=progress.binding)
    protocol_utils.atomic_json(result, Path(run) / (tag + '_metrics.json'))
    return result
