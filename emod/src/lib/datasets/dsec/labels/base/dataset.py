"""SE3D labels aligned explicitly with disparity/event timestamps.

Camera geometry is in the 640x480 rectified event-camera frame. The RGB image
is resized only for visualization/legacy return compatibility; it is not an EMOD
input. Original files and their coordinates are never rewritten.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image
import numpy as np
from scipy import sparse
import torch

from .utils.KITTILoader3D import get_kitti_annos
from .utils import kitti_util
from ...frames import timestamp_file_map
from lib.dsgn.utils.bounding_box import Box3DList
from lib.dsgn.utils.numpy_utils import clip_boxes
from lib.dsgn.utils.torch_utils import compute_locations_bev
from configs.od_cfg import cfg


class LabelsDataset(torch.utils.data.Dataset):
    NO_VALUE = 0.0

    def __init__(self, root, freeze_mode=None, training=True, generate_target=True,
                 label_directory='label_3',
                 timestamp_file='disparity/timestamps_with_label.txt',
                 calibration_path=None, min_box_height=0.0):
        self.root = Path(root).parent / label_directory
        self.sequence_root = self.root.parent
        self.training = training
        self.generate_target = generate_target
        self.freeze_mode = freeze_mode
        self.cfg = cfg
        self.min_box_height = min_box_height
        self.valid_classes = list(cfg.valid_classes)
        self.class_mapping = ({name: i + 1 for i, name in enumerate(cfg.class_names)}
                              if getattr(cfg, 'class_names', None) else None)
        self.timestamp_to_labels_path = {'labels': timestamp_file_map(
            self.sequence_root, timestamp_file, label_directory, '.txt')}
        images = timestamp_file_map(self.sequence_root, timestamp_file, 'image_2', '.png')
        labels = self.timestamp_to_labels_path['labels']
        if set(images) != set(labels):
            raise ValueError(f"Image/label timestamp mismatch: {self.root}")
        for timestamp in labels:
            if Path(labels[timestamp]).stem != Path(images[timestamp]).stem:
                raise ValueError(f"Image/label frame ID mismatch: {self.root}/{timestamp}")
        self.timestamp_to_labels_path['left_img'] = images
        self.timestamps = np.asarray(list(labels), dtype=np.int64)
        self.timestamp_to_index = {t: int(Path(p).stem) for t, p in labels.items()}
        if calibration_path is None:
            # SE3D root / map / sequence / label directory.
            calibration_path = self.sequence_root.parent.parent / 'calib.txt'
        self.calibration_path = Path(calibration_path)
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"Explicit calibration required: {self.calibration_path}")
        self.calib = kitti_util.Calibration.fromfile(str(self.calibration_path))
        self.calib_R = kitti_util.Calibration.fromrightfile(str(self.calibration_path))
        self._anchors = {}

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        return self.load_labels(self.timestamp_to_labels_path['labels'][timestamp])

    @staticmethod
    def collate_fn(batch):
        return batch

    def _locations_and_corners(self, cls):
        if cls not in self._anchors:
            locations = compute_locations_bev(
                self.cfg.Z_MIN, self.cfg.Z_MAX, self.cfg.VOXEL_Z_SIZE,
                self.cfg.X_MIN, self.cfg.X_MAX, self.cfg.VOXEL_X_SIZE, torch.device('cpu'))
            xs, zs = locations[:, 0], locations[:, 1]
            ys = torch.zeros_like(xs) + self.cfg.RPN3D.ANCHORS_Y[cls - 1]
            centers = torch.stack((xs, ys, zs), dim=1)[:, None].repeat(1, self.cfg.num_angles, 1)
            sizes = torch.tensor([self.cfg.RPN3D.ANCHORS_HEIGHT[cls - 1],
                                  self.cfg.RPN3D.ANCHORS_WIDTH[cls - 1],
                                  self.cfg.RPN3D.ANCHORS_LENGTH[cls - 1]])
            sizes = sizes[None, None].repeat(len(centers), self.cfg.num_angles, 1)
            angles = torch.tensor(self.cfg.ANCHOR_ANGLES)[None].repeat(len(centers), 1)
            boxes = torch.cat((sizes, centers, angles[:, :, None]), dim=2)
            boxes[:, :, 4] += boxes[:, :, 0] / 2
            boxes = boxes.reshape(-1, 7)
            target = Box3DList(torch.zeros(len(boxes), 4),
                               tuple(reversed(self.cfg.input_size)), mode='xyxy', box3d=boxes,
                               Proj=self.calib.P, Proj_R=self.calib_R.P)
            corners = target.box_corners() + target.box3d[:, None, 3:6]
            self._anchors[cls] = corners[:, :4, [0, 2]]
        return self._anchors[cls]

    def load_labels(self, path):
        path = Path(path)
        image_index = int(path.stem)
        image_path = self.sequence_root / 'image_2' / (path.stem + '.png')
        with Image.open(image_path) as source:
            left_img = source.convert('L').resize(tuple(reversed(self.cfg.input_size)), Image.Resampling.BILINEAR)
        size = left_img.size
        labels = kitti_util.read_label(str(path))
        labels = [label for label in labels if
                  label.ymax - label.ymin >= self.min_box_height and
                  min(label.h, label.w, label.l, label.cz) > 0]
        boxes, box3ds, classes = get_kitti_annos(
            labels, valid_classes=self.valid_classes, class_mapping=self.class_mapping)
        if len(boxes):
            boxes[:, 2:4] += boxes[:, :2]
            boxes = clip_boxes(boxes, size, remove_empty=False)
            # Target columns and precomputed distances share this exact order:
            # class first, then far-to-near within each class.
            order = np.lexsort((-box3ds[:, 5], classes))
            boxes, box3ds, classes = boxes[order], box3ds[order], classes[order]
        box3ds = torch.as_tensor(box3ds).reshape(-1, 7)
        if self.cfg.learn_viewpoint and len(box3ds):
            box3ds[:, 6] += torch.atan2(box3ds[:, 5], box3ds[:, 3]) - np.pi / 2
        target = Box3DList(torch.as_tensor(boxes).reshape(-1, 4), size,
                           mode='xyxy', box3d=box3ds, Proj=self.calib.P, Proj_R=self.calib_R.P)
        target.add_field('labels', torch.as_tensor(classes, dtype=torch.long))
        iou, label_map = self.target_maps(target)
        return [np.asarray(left_img)[None].astype(np.float32) / 255,
                image_index, size, iou, label_map, self.calib, self.calib_R, target]

    def target_maps(self, target):
        box3ds = target.box3d
        distances, assigned = [], []
        if self.generate_target:
            corners = target.box_corners() + target.box3d[:, None, 3:6]
            for cls in self.valid_classes:
                mask = target.get_field('labels') == cls
                anchors = self._locations_and_corners(cls)
                target_corners = corners[mask][:, :4, [0, 2]]
                distance = torch.norm(anchors[:, None] - target_corners[None], dim=-1).mean(dim=-1)
                distance = distance.clamp(max=5.0)
                labels_map = torch.zeros(distance.shape, dtype=torch.uint8)
                for i, box in enumerate(box3ds[mask]):
                    pixels = float(box[1] * box[2]) / abs(self.cfg.VOXEL_X_SIZE * self.cfg.VOXEL_Z_SIZE)
                    # Preserve legacy rules only for the legacy class mapping.
                    if self.class_mapping is None:
                        if (cls == 2 and getattr(self.cfg, 'less_car_pos', False)) or (
                            cls in (1, 3) and getattr(self.cfg, 'less_human_pos', False)):
                            pixels /= 4
                    k = min(len(distance), max(1, int(abs(pixels))))
                    values, indices = torch.topk(distance[:, i], k, largest=False, sorted=False)
                    labels_map[indices[values < 5.0], i] = cls
                distances.append(distance)
                assigned.append(labels_map)
            iou = sparse.csr_matrix(torch.cat(distances, dim=1).numpy())
            label_map = sparse.csr_matrix(torch.cat(assigned, dim=1).numpy())
        else:
            iou, label_map = None, None
        return iou, label_map
