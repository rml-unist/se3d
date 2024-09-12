import os
from PIL import Image

import numpy as np

import torch.utils.data


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps_with_label.txt',
        'event': 'event',
    }
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    def __init__(self, root, freeze_mode):
        self.root = root
        self.freeze_mode = freeze_mode
        if self.freeze_mode == 'disparity': # object detection task
            self._PATH_DICT['timestamp'] = 'timestamps_with_label.txt'

        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))

        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        for domain in self._DOMAIN:
            self.timestamp_to_disparity_path[domain] = {}
            self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))
            for timestamp, filepath in zip(self.timestamps, self.disparity_path_list[domain]):
                if timestamp == "":
                    continue
                else:
                    self.timestamp_to_disparity_path[domain][int(timestamp)] = filepath 
        self.timestamps = np.array([int(timestamp)for timestamp in self.timestamps if timestamp != ""])
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }


    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        return load_disparity(self.timestamp_to_disparity_path['event'][timestamp])

    @staticmethod
    def collate_fn(batch):

        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch


def load_timestamp(root):
    with open(root, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = lines[i].replace("\n", "")
    return lines


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root):
    disparity = np.load(root).astype(np.float32)
    return disparity
