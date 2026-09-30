"""Compute per-class anchor sizes from the training frames.

Rule used for the files in anchors/: over the training boxes with a 2D height
of at least 15 px and positive height, width, length and depth, take per class
the median height, width and length and the median of (y - height/2).

    python tools/compute_anchors.py --data-root /data/SE3D --labels label --output anchors/mine.json
    python tools/compute_anchors.py --data-root /data/SE3D --labels label_original \
        --conditions day_sunny night_sunny --output anchors/sunny.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from se3d import CLASSES, CONDITIONS
from se3d.data import condition_of, load_splits

RULE = ('Per class, the median height, width and length, and the median of (y - height/2), over training '
        'boxes with a 2D height of at least 15 px and positive height, width, length and depth.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--labels', default='label')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    samples = {name: [] for name in CLASSES}
    for sequence, frames in sorted(load_splits()['frames']['train'].items()):
        if args.conditions and condition_of(sequence) not in args.conditions:
            continue
        for _, frame in frames:
            path = Path(args.data_root) / sequence / args.labels / ('%06d.txt' % frame)
            for line in path.read_text().splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 15 or fields[0] not in samples:
                    raise ValueError('Unexpected label line in %s: %s' % (path, line))
                if float(fields[7]) - float(fields[5]) >= 15 and all(float(fields[k]) > 0 for k in (8, 9, 10, 13)):
                    samples[fields[0]].append([float(fields[k]) for k in (8, 9, 10, 12)])
    anchors = {}
    for name, values in samples.items():
        if not values:
            raise ValueError('No eligible training boxes for %s' % name)
        a = np.asarray(values)
        anchors[name] = dict(samples=len(a), height=float(np.median(a[:, 0])), width=float(np.median(a[:, 1])),
                             length=float(np.median(a[:, 2])), center_y=float(np.median(a[:, 3] - a[:, 0] / 2)))
    result = dict(labels=args.labels, split='train',
                  conditions=', '.join(args.conditions) if args.conditions else 'all', rule=RULE,
                  training_anchor_dimensions=anchors)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['samples'] for k, v in anchors.items()}))


if __name__ == '__main__':
    main()
