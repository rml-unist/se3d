"""Precompute the causal stereo event stacks of all protocol keyframes (optional).

Each keyframe uses the last 5,000,000 raw events before its timestamp in each
camera, rectified and cut to the common time window, stacked into 10
mixed-density stacks. Training reads identical tensors with or without this
cache; the cache only avoids re-reading events.h5 every epoch.

    python transfer/prepare_cache.py --dsec-root <DSEC>/train --cache-root <cache dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402


def validate(path, timestamp, n, stack):
    with np.load(path, allow_pickle=False) as f:
        assert int(f['timestamp_us']) == timestamp and int(f['num_events']) == n and int(f['stack_size']) == stack
        for side in ('left', 'right'):
            a = f[side]
            assert a.shape == (stack, 1, 480, 640, 1) and a.dtype == np.int8 and np.isin(a, [-1, 0, 1]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dsec-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    args = parser.parse_args()
    from lib.datasets.dsec_original import OriginalStereoReader
    manifest = json.loads(common.PROTOCOL.read_text())
    rows = [row for split in args.splits for row in manifest['splits'][split]]
    n, stack = manifest['num_past_raw_events'], manifest['stack_size']
    cache, started, readers, generated = Path(args.cache_root), time.monotonic(), {}, 0
    for i, row in enumerate(rows):
        seq, timestamp = row['sequence'], row['timestamp_us']
        out = cache / seq / ('%d.npz' % timestamp)
        if out.exists():
            validate(out, timestamp, n, stack)
            continue
        if seq not in readers:
            readers = {seq: OriginalStereoReader(Path(args.dsec_root) / seq, n, stack)}
        events, window = readers[seq].read(timestamp)
        for a in events.values():
            assert np.isfinite(a).all() and np.isin(a, [-1, 0, 1]).all()
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix('.%d.tmp' % os.getpid())
        with temporary.open('wb') as f:
            np.savez_compressed(f, **{s: a.astype(np.int8) for s, a in events.items()}, timestamp_us=timestamp,
                                num_events=n, stack_size=stack, common_start_us=window['common_start_us'],
                                common_last_us=window['common_last_us'])
        validate(temporary, timestamp, n, stack)
        with np.load(temporary, allow_pickle=False) as f:
            for side in events:
                np.testing.assert_array_equal(f[side], events[side])
        temporary.replace(out)
        generated += 1
        if generated % 100 == 0:
            print(json.dumps(dict(progress=i + 1, total=len(rows), seconds=round(time.monotonic() - started))), flush=True)
    print(json.dumps(dict(keyframes=len(rows), new_caches=generated, seconds=round(time.monotonic() - started))))


if __name__ == '__main__':
    main()
