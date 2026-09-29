"""Check a downloaded SE3D copy against the split file and the published counts.

    python tools/check_dataset.py --data-root /data/SE3D
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from se3d.data import load_splits

EXPECTED = dict(frames=48304, annotated_frames=46229, label=175123, label_original=184884)
REQUIRED = ['timestamps.txt', 'disparity/timestamps_with_label.txt', 'events/left/events.h5',
            'events/left/rectify_map.h5', 'events/right/events.h5', 'events/right/rectify_map.h5']
PER_FRAME = [('label', '.txt'), ('label_original', '.txt'), ('image_2', '.png'), ('disparity/event', '.npy')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--sequences', nargs='*', help='check only these sequences (skips the dataset totals)')
    args = parser.parse_args()
    root = Path(args.data_root)
    splits = load_splits()
    problems = []
    totals = dict(frames=0, annotated_frames=0, label=0, label_original=0)
    if not (root / 'calib.txt').is_file():
        problems.append('missing calib.txt')
    selected = sorted(args.sequences or splits['sequences'])
    for sequence in selected:
        base = root / sequence
        missing = [name for name in REQUIRED if not (base / name).is_file()]
        if missing:
            problems.append('%s: missing %s' % (sequence, ', '.join(missing)))
            continue
        count = len((base / 'timestamps.txt').read_text().splitlines())
        totals['frames'] += count
        for folder, suffix in PER_FRAME:
            found = len(list((base / folder).glob('*' + suffix)))
            if found != count:
                problems.append('%s/%s: %d files for %d timestamps' % (sequence, folder, found, count))
        for version in ('label', 'label_original'):
            for path in (base / version).glob('*.txt'):
                rows = sum(1 for line in path.read_text().splitlines() if line.strip())
                totals[version] += rows
                if version == 'label':
                    totals['annotated_frames'] += rows > 0
    for split, items in splits['frames'].items():
        for sequence, frames in items.items():
            if sequence not in selected:
                continue
            for _, frame in frames:
                if not (root / sequence / 'label' / ('%06d.txt' % frame)).is_file():
                    problems.append('%s: %s frame %06d missing' % (split, sequence, frame))
                    break
    for key, value in EXPECTED.items():
        if not args.sequences and totals[key] != value:
            problems.append('%s: found %d, expected %d' % (key, totals[key], value))
    print(json.dumps(dict(totals=totals, problems=problems[:50], number_of_problems=len(problems)), indent=1))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
