import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from . import labels
from . import transforms


class SequenceDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'event': 'events',
        'disparity': 'disparity',
        'labels': 'label_2'
    }
    HEIGHT = 384
    WIDTH = 1248

    def __init__(self, root, freeze_mode, split, sampling_ratio, event_cfg, disparity_cfg, labels_cfg,
                 crop_height, crop_width, num_workers=0):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.labels_cfg = labels_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        self.freeze_mode = freeze_mode

        self.sequence_name = root.split('/')[-2:]

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        event_root = os.path.join(root, self._PATH_DICT['event'])
        self.event_dataset = event_module.EventDataset(root=event_root, **event_cfg.PARAMS)

        # Disparity Dataset
        disparity_module = getattr(disparity, disparity_cfg.NAME)
        disparity_root = os.path.join(root, self._PATH_DICT['disparity'])
        self.disparity_dataset = disparity_module.DisparityDataset(root=disparity_root, freeze_mode=freeze_mode, **disparity_cfg.PARAMS)

        # Labels Dataset
        labels_module = getattr(labels, labels_cfg.NAME)
        labels_root = os.path.join(root, self._PATH_DICT['labels'])
        self.labels_dataset = labels_module.LabelsDataset(root=labels_root, freeze_mode=freeze_mode, **labels_cfg.PARAMS)



        # Timestamps
        if split in ['train', 'validation', 'trainval']:
            self.timestamps = copy.copy(self.disparity_dataset.timestamps)
            if event_cfg.NAME != 'none':
                minimum_timestamp = max(self.event_dataset.event_slicer['left'].t_offset,
                                        self.event_dataset.event_slicer['right'].t_offset)
                maximum_timestamp = None
                if event_cfg.NAME == 'base':
                    minimum_timestamp += self.event_dataset.time_interval
                elif event_cfg.NAME == 'sbn':
                    minimum_timestamp = max(self.event_dataset.event_slicer['left'].min_time,
                                            self.event_dataset.event_slicer['right'].min_time,)
                    maximum_timestamp = min(self.event_dataset.event_slicer['left'].max_time,
                                            self.event_dataset.event_slicer['right'].max_time)
                self.timestamps = self.timestamps[self.timestamps >= minimum_timestamp]
                if maximum_timestamp is not None:
                    self.timestamps = self.timestamps[self.timestamps <= maximum_timestamp]
            self.timestamp_to_index = copy.copy(self.disparity_dataset.timestamp_to_index)
        elif split in ['test']:
            self.timestamps, self.timestamp_to_index = read_csv(os.path.join(root, 'timestamps_idx.csv'))
        else:
            raise NotImplementedError

        self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]

        # Transforms
        if split in ['train', 'trainval']:
            self.transforms = transforms.Compose([
                transforms.ToTensor(event_module=event_module,
                                    disparity_module=disparity_module,)
            ])
        elif split in ['validation', 'test']:
            self.transforms = transforms.Compose([
                transforms.ToTensor(event_module=event_module,
                                    disparity_module=disparity_module, ),
            ])
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        data = self.load_data(idx)

        return data

    def collate_fn(self, batch):
        output = {}
        # Event
        domain = 'event'
        if domain in batch[0].keys():
            output[domain] = self.event_dataset.collate_fn([sample[domain] for sample in batch])

        # Disparity
        domain = 'disparity'
        if domain in batch[0].keys():
            output[domain] = self.disparity_dataset.collate_fn([sample[domain] for sample in batch])

        # Labels
        domain = 'labels'
        if domain in batch[0].keys():
            output[domain] = self.labels_dataset.collate_fn([sample[domain] for sample in batch])

        # Others
        for key in batch[0].keys():
            if key not in ['event', 'disparity', 'labels']:
                output[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch])

        return output

    def load_data(self, idx):
        timestamp = self.timestamps[idx]
        data = {}
        event_data, separated_event = self.event_dataset[timestamp]
        disparity_data = self.disparity_dataset[timestamp]
        labels_data = self.labels_dataset[timestamp]
        data['file_index'] = self.timestamp_to_index[timestamp]

        if event_data is not None:
            data['event'] = event_data
        if disparity_data is not None:
            data['disparity'] = disparity_data
        if labels_data is not None:
            data['labels'] = labels_data
        if separated_event is not None:
            data['separated_event'] = separated_event

        return data


def read_csv(csv_file):
    timestamps = []
    timestamp_to_index = {}
    with open(csv_file) as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            assert row[0] not in timestamps
            if row[0].isnumeric():
                timestamps.append(int(row[0]))
                timestamp_to_index[int(row[0])] = int(row[1])

    return np.asarray(timestamps), timestamp_to_index
