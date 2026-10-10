import os
import numpy as np
import torch.utils.data

from .cache import load_time_bounds, load_trusted_cache
from . import stack, constant


class EventDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps_with_label.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None

    def __init__(self, root, num_of_event, stack_method, stack_size,
                 num_of_future_event=0, use_preprocessed_image=False,
                 cache_only=False, cache_time_bounds=None, cache_root=None, **kwargs):
        self.root = root
        self.cache_root = root if cache_root is None else os.fspath(cache_root)
        self.num_of_event = num_of_event
        self.stack_method = stack_method
        self.stack_size = stack_size
        self.num_of_future_event = num_of_future_event
        self.use_preprocessed_image = use_preprocessed_image
            
        self.cache_only = cache_only
        if cache_only and not use_preprocessed_image:
            raise ValueError("cache_only requires use_preprocessed_image=True")
        self.cache_name = 'sbn_%d_%s_%d_%d' % (
            num_of_event, stack_method, stack_size, num_of_future_event)
        if cache_only:
            self.event_slicer = load_time_bounds(
                cache_time_bounds, root, num_of_event, stack_method,
                stack_size, num_of_future_event)
        else:
            from .slice import EventSlicer
            self.event_slicer = {}
            for location in self._LOCATION:
                event_path = os.path.join(root, location, 'events.h5')
                rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
                self.event_slicer[location] = EventSlicer(
                    event_path, rectify_map_path, num_of_event, num_of_future_event)

        self.stack_function = getattr(stack, stack_method)(stack_size, num_of_event,
                                                           constant.EVENT_HEIGHT, constant.EVENT_WIDTH, **kwargs)
        self.NO_VALUE = self.stack_function.NO_VALUE

    def __len__(self):
        return 0

    def __getitem__(self, timestamp):
        if self.use_preprocessed_image:
            save_path = os.path.join(self.cache_root, self.cache_name, '%ld.npy' % timestamp)
            if self.cache_only or os.path.exists(save_path):
                event_data = load_trusted_cache(
                    save_path, self.stack_size, constant.EVENT_HEIGHT,
                    constant.EVENT_WIDTH, self.num_of_future_event)
            else:
                event_data = self._pre_load_event_data(timestamp=timestamp)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                temporary = '%s.%d.tmp' % (save_path, os.getpid())
                torch.save(event_data, temporary)
                os.replace(temporary, save_path)
                event_data = self._past_only(event_data)
        else:
            event_data = self._past_only(self._pre_load_event_data(timestamp=timestamp))

        event_data = self._post_load_event_data(event_data)

        return event_data

    def _past_only(self, event_data):
        # Same rule as load_trusted_cache: without future events, only the past stack is used.
        if self.num_of_future_event == 0:
            event_data = {side: stacks[:1] for side, stacks in event_data.items()}
        return event_data

    def validate_cache_paths(self, timestamps):
        if not self.cache_only:
            return
        missing = [int(t) for t in timestamps if not os.path.isfile(
            os.path.join(self.cache_root, self.cache_name, '%ld.npy' % t))]
        if missing:
            raise FileNotFoundError(
                f"{self.root}: {len(missing)} required caches missing; "
                f"first timestamps={missing[:10]}. Split is not changed.")

    def _pre_load_event_data(self, timestamp):
        event_data = {}

        minimum_time, maximum_time = -float('inf'), float('inf')
        for location in self._LOCATION:
            event_data[location] = self.event_slicer[location][timestamp]
            minimum_time = max(minimum_time, event_data[location]['t'].min())
            maximum_time = min(maximum_time, event_data[location]['t'].max())

        for location in self._LOCATION:
            mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
            for data_type in ['x', 'y', 't', 'p']:
                event_data[location][data_type] = event_data[location][data_type][mask]

        for location in self._LOCATION:
            event_data[location] = self.stack_function.pre_stack(event_data[location], timestamp)

        return event_data

    def _post_load_event_data(self, event_data):
        for location in self._LOCATION:
            loc = 'l' if location=='left' else 'r'
            event_data[location] = self.stack_function.post_stack(event_data[location])

        return event_data

    def collate_fn(self, batch):
        batch = self.stack_function.collate_fn(batch)

        return batch

