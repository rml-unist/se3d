"""Check release split and anchor invariants without downloading data or PyTorch."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from se3d import ANCHORS_ROOT, CLASSES, REPO_ROOT
from se3d.protocol import condition_of, load_splits


def check():
    data = load_splits()
    expected = {'train': 26796, 'val': 6709, 'test': 3991, 'test_car_filtered': 2731}
    for name, count in expected.items():
        frames = data['frames'][name]
        if sum(len(v) for v in frames.values()) != count:
            raise ValueError('Wrong split size: ' + name)
        for sequence, rows in frames.items():
            condition_of(sequence)
            if rows != sorted(rows) or len({t for t, _ in rows}) != len(rows):
                raise ValueError('Unordered/duplicate timestamps: ' + sequence)
            if len({f for _, f in rows}) != len(rows):
                raise ValueError('Duplicate frame indices: ' + sequence)
    train, val, test = [set(data['frames'][s]) for s in ('train', 'val', 'test')]
    if train & val or train & test or val & test:
        raise ValueError('A recording belongs to multiple splits')
    for sequence, rows in data['frames']['test_car_filtered'].items():
        if not set(map(tuple, rows)) <= set(map(tuple, data['frames']['test'][sequence])):
            raise ValueError('Historical test is not a subset of the full test')
    anchors = sorted(ANCHORS_ROOT.glob('*.json'))
    for path in anchors:
        values = json.loads(path.read_text())['training_anchor_dimensions']
        if set(values) != set(CLASSES):
            raise ValueError('Anchor classes differ: ' + path.name)
        for name, row in values.items():
            if not all(math.isfinite(row[k]) for k in ('height', 'width', 'length', 'center_y')):
                raise ValueError('Nonfinite anchor: ' + name)
            if min(row[k] for k in ('height', 'width', 'length')) <= 0:
                raise ValueError('Nonpositive anchor dimensions: ' + name)
    if 'MIT License' not in (REPO_ROOT / 'docs' / 'DATASET_LICENSE.md').read_text():
        raise ValueError('Dataset license is missing')
    return dict(passed=True, split_frames=expected, sequences=len(train | val | test),
                anchor_files=len(anchors), dataset_license='MIT')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
