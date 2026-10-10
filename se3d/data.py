"""SE3D frames of one split, read with the EMOD sequence loaders.

The frame lists in splits/se3d_splits.json fix which frames belong to each
split and in which order they are read. Each entry is [timestamp, frame], where
frame is the file number of the label, image and disparity files.
"""
import bisect
import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

from . import CONDITIONS, EMOD_ROOT, SPLITS_FILE
from .protocol import condition_of, label_name, load_splits
from lib.datasets.dsec.sequence import SequenceDataset

# Crop width 648 pads the 640-pixel event frames as in the original EMOD setup.
HEIGHT, WIDTH, CROP_WIDTH = 480, 640, 648


class SE3DFrames(torch.utils.data.Dataset):
    """Frames of one split ('train', 'val', 'test' or 'test_car_filtered').

    label_dir selects the annotation version: 'label' (deduplicated, used for
    all reported evaluations) or 'label_original' (used to train the reported
    40-epoch models). Event stacks are computed from events.h5 on first access
    and cached under <sequence>/events/sbn_5000000_MixedDensityEventStacking_10_0/.
    """

    def __init__(self, data_root, split, label_dir='label', generate_target=True,
                 conditions=None, splits_file=SPLITS_FILE, cache_root=None, cache_time_bounds=None,
                 sequences=None, max_frames=None):
        label_dir = label_name(label_dir)
        splits = load_splits(splits_file)
        if split not in splits['frames']:
            raise ValueError('Unknown split: %s' % split)
        conf = EasyDict(yaml.safe_load((EMOD_ROOT / 'configs' / 'config.yaml').read_text()))
        training = split == 'train'
        self.data_root = Path(data_root)
        self.split = split
        self.datasets, self.sequence_names, self.ends = [], [], []
        remaining = max_frames
        for name, frames in sorted(splits['frames'][split].items()):
            if sequences is not None and name not in sequences:
                continue
            if conditions and condition_of(name) not in conditions:
                continue
            if remaining is not None:
                if remaining <= 0:
                    break
                frames = frames[:remaining]
                remaining -= len(frames)
            event_cfg = copy.deepcopy(conf.DATASET.TRAIN.PARAMS.event_cfg)
            if cache_root is not None:
                event_cfg.PARAMS.cache_root = str(Path(cache_root) / name / 'events')
            if cache_time_bounds is not None:
                event_cfg.PARAMS.cache_only = True
                event_cfg.PARAMS.cache_time_bounds = str(cache_time_bounds)
            labels_cfg = copy.deepcopy(conf.DATASET.TRAIN.PARAMS.labels_cfg)
            labels_cfg.PARAMS.label_directory = label_dir
            labels_cfg.PARAMS.generate_target = generate_target
            labels_cfg.PARAMS.training = training
            labels_cfg.PARAMS.min_box_height = 15 if training else 0
            labels_cfg.PARAMS.timestamp_file = 'timestamps.txt'
            disparity_cfg = copy.deepcopy(conf.DATASET.TRAIN.PARAMS.disparity_cfg)
            if split.startswith('test'):
                disparity_cfg.PARAMS.timestamp_file = 'timestamps.txt'
            # 'val' only selects the evaluation transforms; the frames come from the split file.
            ds = SequenceDataset(root=str(self.data_root / name), freeze_mode=None,
                                 split='train' if training else 'val', sampling_ratio=1,
                                 event_cfg=event_cfg, disparity_cfg=disparity_cfg, labels_cfg=labels_cfg,
                                 crop_height=HEIGHT, crop_width=CROP_WIDTH, validate_cache=False)
            ds.timestamps = np.asarray([t for t, _ in frames], dtype=np.int64)
            ds.timestamp_to_index = {int(t): int(f) for t, f in frames}
            for t, f in frames:
                for mapping in (ds.disparity_dataset.timestamp_to_index, ds.labels_dataset.timestamp_to_index):
                    if int(mapping.get(t, -1)) != f:
                        raise ValueError('%s: timestamp %d is not frame %06d in the data' % (name, t, f))
            ds.event_dataset.validate_cache_paths(ds.timestamps)
            self.datasets.append(ds)
            self.sequence_names.append(name)
            self.ends.append((self.ends[-1] if self.ends else 0) + len(ds))
        if not self.ends:
            raise ValueError('No frames selected for split %s' % split)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        sequence = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[sequence - 1] if sequence else 0)
        ds = self.datasets[sequence]
        sample = ds[local]
        timestamp = ds.timestamps[local]
        sample['metadata'] = dict(sequence=self.sequence_names[sequence], timestamp=int(timestamp),
                                  frame=int(sample['file_index']),
                                  label_path=str(ds.labels_dataset.timestamp_to_labels_path['labels'][timestamp]))
        return sample

    @staticmethod
    def collate_fn(batch):
        return dict(event={s: torch.stack([b['event'][s] for b in batch]) for s in ('left', 'right')},
                    disparity=torch.stack([b['disparity'] for b in batch]),
                    labels=[b['labels'] for b in batch],
                    metadata=[b['metadata'] for b in batch],
                    file_index=torch.tensor([b['file_index'] for b in batch]))


def model_args(batch, device='cuda', training=True):
    """Keyword arguments shared by both baselines' forward()."""
    labels = batch['labels']
    return dict(left_event=batch['event']['left'].to(device), right_event=batch['event']['right'].to(device),
                gt_disparity=batch['disparity'].to(device),
                calibs_Proj=torch.as_tensor(np.stack([l[5].P for l in labels])),
                calibs_Proj_R=torch.as_tensor(np.stack([l[6].P for l in labels])),
                targets=[l[7] for l in labels] if training else None,
                ious=[l[3] for l in labels], labels_map=[l[4] for l in labels], is_test=not training)
