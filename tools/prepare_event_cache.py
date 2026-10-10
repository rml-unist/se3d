"""Precompute the event stacks of the split frames (optional).

Training and testing compute a missing stack from events.h5 on first access and
store it; running this script first makes the first epoch as fast as the rest.
Stacks go to <sequence>/events/sbn_5000000_MixedDensityEventStacking_10_0/<timestamp>.npy
(about 3 MB per frame).

    python tools/prepare_event_cache.py --data-root /data/SE3D --splits train val test --processes 8
"""
import argparse
import copy
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from easydict import EasyDict

from se3d import EMOD_ROOT
from se3d.protocol import load_splits


def prepare(job):
    data_root, cache_root, sequence, timestamps = job
    from lib.datasets.dsec.event.sbn.dataset import EventDataset
    conf = EasyDict(yaml.safe_load((EMOD_ROOT / 'configs' / 'config.yaml').read_text()))
    params = copy.deepcopy(dict(conf.DATASET.TRAIN.PARAMS.event_cfg.PARAMS))
    params['use_preprocessed_image'] = True
    if cache_root is not None:
        params['cache_root'] = str(Path(cache_root) / sequence / 'events')
    dataset = EventDataset(root=str(Path(data_root) / sequence / 'events'), **params)
    for timestamp in timestamps:
        dataset[timestamp]
    return sequence, len(timestamps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-root', help='writable event cache root, separate from the dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test', 'test_car_filtered'])
    parser.add_argument('--processes', type=int, default=4)
    args = parser.parse_args()
    frames = {}
    for split in args.splits:
        for sequence, items in load_splits()['frames'][split].items():
            frames.setdefault(sequence, set()).update(t for t, _ in items)
    jobs = [(args.data_root, args.cache_root, sequence, sorted(timestamps)) for sequence, timestamps in sorted(frames.items())]
    with Pool(args.processes) as pool:
        for done, (sequence, count) in enumerate(pool.imap_unordered(prepare, jobs), 1):
            print('[%d/%d] %s: %d frames' % (done, len(jobs), sequence, count), flush=True)


if __name__ == '__main__':
    main()
