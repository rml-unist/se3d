"""Check a local DSEC + DSEC-3DOD copy against the frozen transfer protocol.

Recomputes, from the downloaded files, what protocol/dsec_paired_v2.json and
protocol/dsec_training_anchors_v2.json fix:
  * the official split lists (train.txt, val.txt) and every annotation pickle and
    calibration file, by sha256;
  * the keyframe list of each split (3,906 train, 434 internal validation,
    1,178 test), including the causal 5M-event history and disparity files;
  * the target anchors (per-class medians of the training boxes).

    python transfer/check_data.py --dsec-root <DSEC>/train --labels-root <DSEC-3DOD>
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dsec-root', required=True)
    parser.add_argument('--labels-root', required=True)
    args = parser.parse_args()
    data, original = Path(args.labels_root), Path(args.dsec_root)
    protocol = json.loads(common.PROTOCOL.read_text())
    problems = []

    for name in ('train', 'val'):
        if common.sha256(data / (name + '.txt')) != protocol['source_split_sha256'][name]:
            problems.append('%s.txt differs from the DSEC-3DOD release used in the paper' % name)
    expected_hashes = {}
    for key, value in protocol['source_hashes'].items():
        # Calibration keys were stored as absolute cluster paths; keep <sequence>/calibration/<file>.
        expected_hashes['/'.join(Path(key).parts[-3:]) if 'calibration' in key else key] = value

    train_chunks = [line.split()[0] for line in (data / 'train.txt').read_text().splitlines()]
    test_chunks = [line.split()[0] for line in (data / 'val.txt').read_text().splitlines()]
    validation = set(sorted(train_chunks, key=lambda s: hashlib.sha256(
        ('se3d-dsec-model-selection-v1/' + s).encode()).hexdigest())[:14])
    splits = {s: [] for s in ('train', 'validation', 'test')}
    bounds, checked = {}, set()
    for original_split, chunks in (('train', train_chunks), ('val', test_chunks)):
        for chunk in chunks:
            split = 'test' if original_split == 'val' else 'validation' if chunk in validation else 'train'
            seq = chunk.rsplit('_', 1)[0]
            path = data / chunk / (chunk + '_fov_bbox_lidar_check.pkl')
            key = str(path.relative_to(data))
            if common.sha256(path) != expected_hashes.get(key):
                problems.append('annotation differs: ' + key)
            annotations = pickle.loads(path.read_bytes())
            if seq not in bounds:
                low, high = [], []
                for side in ('left', 'right'):
                    with h5py.File(original / seq / 'events' / side / 'events.h5', 'r') as f:
                        offset = int(f['t_offset'][()])
                        low.append(int(f['events/t'][4999999]) + offset)
                        high.append(int(f['events/t'][-1]) + offset)
                bounds[seq] = (max(low), min(high))
                for filename in ('cam_to_cam.yaml', 'cam_to_lidar.yaml'):
                    rel = '%s/calibration/%s' % (seq, filename)
                    if common.sha256(original / rel) != expected_hashes.get(rel):
                        problems.append('calibration differs: ' + rel)
            times = np.loadtxt(original / seq / 'images' / 'timestamps.txt', dtype=np.int64)
            for i, a in enumerate(annotations):
                timestamp = int(a['time_stamp'])
                event_id = int(Path(a['image']['event_0_path']).stem)
                if not bounds[seq][0] < timestamp <= bounds[seq][1]:
                    problems.append('insufficient event history: %s/%d' % (chunk, i))
                if int(times[event_id]) != timestamp:
                    problems.append('image timestamp mismatch: %s/%d' % (chunk, i))
                disparity = original / seq / 'disparity' / 'event' / ('%06d.png' % event_id)
                if not disparity.is_file():
                    problems.append('missing disparity: %s' % disparity)
                splits[split].append(dict(sequence=seq, chunk=chunk, frame=i, timestamp_us=timestamp,
                                          original_frame_id=event_id, annotation_path=key,
                                          disparity_path=str(disparity.relative_to(original))))
            checked.add(chunk)
    for split, rows in splits.items():
        if rows != protocol['splits'][split]:
            problems.append('%s keyframes differ from the protocol (%d vs %d)' % (split, len(rows), len(protocol['splits'][split])))

    # Target anchors from the training keyframes only.
    common.configure()
    from lib.datasets.dsec_original import load_calibration, lidar_boxes_to_camera
    pool = {n: [] for n in ('Vehicle', 'Pedestrian', 'Cyclist')}
    annotations, calibrations = {}, {}
    for row in protocol['splits']['train']:
        chunk, seq = row['chunk'], row['sequence']
        if chunk not in annotations:
            annotations[chunk] = pickle.loads((data / row['annotation_path']).read_bytes())
        if seq not in calibrations:
            calibrations[seq] = load_calibration(original / seq, annotations[chunk][row['frame']]['image']['image_0_extrinsic'])
        a = annotations[chunk][row['frame']]['annos']
        boxes, _ = lidar_boxes_to_camera(a['gt_boxes_lidar'], calibrations[seq])
        for name, box in zip(a['name'], boxes):
            pool[str(name)].append([*box[:3], box[4] - box[0] / 2])
    anchors = {}
    for name, rows in pool.items():
        median = np.median(rows, axis=0)
        anchors[name] = dict(zip(('height', 'width', 'length', 'center_y'), median.tolist()))
        anchors[name]['annotations'] = len(rows)
    if dict(training_anchor_dimensions=anchors) != json.loads(common.ANCHORS.read_text()):
        problems.append('target anchors differ from protocol/dsec_training_anchors_v2.json')

    report = dict(chunks=len(checked), keyframes={s: len(r) for s, r in splits.items()}, problems=problems[:50],
                  problem_count=len(problems), matches_protocol=not problems)
    print(json.dumps(report, indent=1))
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
