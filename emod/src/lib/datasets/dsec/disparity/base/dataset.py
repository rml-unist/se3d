import os
from PIL import Image

import numpy as np

import torch.utils.data
from ...frames import timestamp_file_map


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps_with_label.txt',
        'event': 'event',
    }
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    def __init__(self, root, freeze_mode=None,
                 timestamp_file='disparity/timestamps_with_label.txt'):
        self.root = root
        self.freeze_mode = freeze_mode
        mapping = timestamp_file_map(os.path.dirname(root), timestamp_file, 'disparity/event', '.npy')
        self.timestamp_to_disparity_path = {'event': mapping}
        self.timestamps = np.asarray(list(mapping), dtype=np.int64)
        self.timestamp_to_index = {t: int(os.path.splitext(os.path.basename(p))[0]) for t, p in mapping.items()}

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
    # return np.loadtxt(root, dtype='int64')


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root):
    disparity = np.load(root).astype(np.float32)
    return disparity
