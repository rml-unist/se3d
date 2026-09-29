"""The two joint baselines: EMOD and DSGN-event.

Both take the left/right 10-stack event tensors and return
(disparity, detection outputs, depth loss, detection loss).
"""
import json
from pathlib import Path

import torch
import yaml
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

from . import ANCHORS_ROOT, CLASSES, EMOD_ROOT
from configs.od_cfg import cfg
from lib.dsgn.loss3d import RPN3DLoss

MODELS = ('emod', 'dsgn_event')
ANCHORS = {
    'original': ANCHORS_ROOT / 'se3d_anchors_label_original.json',
    'corrected': ANCHORS_ROOT / 'se3d_anchors_label.json',
    'original_sunny': ANCHORS_ROOT / 'se3d_anchors_label_original_sunny.json',
}


def configure_classes(anchors_path):
    """Set the seven classes and their training-set anchor sizes in the shared cfg.

    Call this before building datasets or models: target generation and the
    detection heads both read cfg.
    """
    anchors = json.loads(Path(anchors_path).read_text())['training_anchor_dimensions']
    cfg.class_names = list(CLASSES)
    cfg.num_classes = len(CLASSES)
    cfg.valid_classes = list(range(1, len(CLASSES) + 1))
    for key, field in [('ANCHORS_HEIGHT', 'height'), ('ANCHORS_WIDTH', 'width'),
                       ('ANCHORS_LENGTH', 'length'), ('ANCHORS_Y', 'center_y')]:
        setattr(cfg.RPN3D, key, [anchors[name][field] for name in CLASSES])
    return cfg


class DSGNEvent(nn.Module):
    """The DSGN stereo network and 3D head with a 10-channel event input.

    DSGN predicts metric depth; disparity is derived from it with f*B/Z. The
    depth loss is Smooth L1 in meters over the DSGN depth range.
    """

    def __init__(self, **unused):
        super().__init__()
        from dsgn.models import StereoNet
        bc = cfg.clone()
        bc.event_input_channels = 10
        bc.downsample_disp = 4
        bc.input_size = [480, 656]
        bc.output_size = [120, 164]
        bc.CV_X_MAX = 655.
        bc.CV_INPUT_WIDTH = 656
        bc.CV_GRID_SIZE = [bc.CV_INPUT_DEPTH, bc.CV_INPUT_HEIGHT, 656]
        self.network = StereoNet(cfg=bc)
        self.loss_function = RPN3DLoss(cfg)
        self.baseline_cfg = bc

    def forward(self, left_event, right_event, gt_disparity=None, calibs_Proj=None, calibs_Proj_R=None,
                targets=None, ious=None, labels_map=None, is_test=False, **unused):
        b, s, t, h, w, c = left_event.shape
        if (s, t, c) != (10, 1, 1):
            raise ValueError('Expected a past-only 10-stack event input')

        def flatten(x):
            return F.pad(x.permute(0, 5, 1, 2, 3, 4).reshape(b, c * s * t, h, w), (0, 656 - w))

        left, right = flatten(left_event), flatten(right_event)
        pl = calibs_Proj.to(left.device).float()
        pr = calibs_Proj_R.to(left.device).float()
        fu = pl[:, 0, 0]
        baseline = (pl[:, 0, 3] - pr[:, 0, 3]).abs() / fu
        fb = fu * baseline
        outputs = self.network(left, right, fu, baseline, pl, pr)
        depths = outputs['depth_preds']
        depth = (depths[-1] if isinstance(depths, list) else depths)[:, :h, :w]
        disparity = fb[:, None, None] / depth.clamp_min(1e-6)
        detection = {k: outputs[k] for k in ('bbox_cls', 'bbox_reg', 'bbox_centerness')}
        depth_loss = depth.sum() * 0
        if gt_disparity is not None:
            gt_depth = torch.where(gt_disparity > 0, fb[:, None, None] / gt_disparity.clamp_min(1e-6),
                                   torch.zeros_like(gt_disparity))
            mask = (gt_disparity > 0) & (gt_depth > self.baseline_cfg.min_depth) & (gt_depth <= self.baseline_cfg.max_depth)
            if mask.any():
                depth_loss = F.smooth_l1_loss(depth[mask], gt_depth[mask])
        detection_loss = depth.sum() * 0
        if not is_test:
            for target in targets:
                target.bbox = target.bbox.to(left.device)
                target.box3d = target.box3d.to(left.device)
            detection_loss = self.loss_function(detection['bbox_cls'], detection['bbox_reg'],
                                                detection['bbox_centerness'], targets, calibs_Proj,
                                                calibs_Proj_R, ious=ious, labels_map=labels_map)[0]
        return disparity, detection, depth_loss, detection_loss


def build_model(name):
    """Build a baseline after configure_classes() has been called."""
    conf = EasyDict(yaml.safe_load((EMOD_ROOT / 'configs' / 'config.yaml').read_text()))
    if name == 'emod':
        from lib.net import MyModel
        return MyModel(**conf.MODEL.PARAMS)
    if name == 'dsgn_event':
        return DSGNEvent(**conf.MODEL.PARAMS)
    raise ValueError('Unknown model: %s (choose from %s)' % (name, ', '.join(MODELS)))
