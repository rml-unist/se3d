"""Transfer split, input and anchor provenance; no model or CUDA imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'protocol' / 'dsec_paired_v2.json'
CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a dataset-relative path: %r' % value)
    return path


def frame_key(row):
    return (row['chunk'], int(row['frame']), int(row['timestamp_us']))


def load_protocol(path=PROTOCOL):
    protocol = json.loads(Path(path).read_text())
    splits = protocol['splits']
    seen_frames, seen_chunks = set(), set()
    for split in ('train', 'validation', 'test'):
        rows = splits[split]
        keys = [frame_key(row) for row in rows]
        chunks = {row['chunk'] for row in rows}
        if not rows or len(keys) != len(set(keys)) or seen_frames.intersection(keys):
            raise ValueError('Empty, duplicate or overlapping frames in ' + split)
        if seen_chunks.intersection(chunks):
            raise ValueError('A chunk appears in multiple splits')
        seen_frames.update(keys)
        seen_chunks.update(chunks)
        for row in rows:
            safe_relative(row['annotation_path'])
            safe_relative(row['disparity_path'])
            safe_relative(row['sequence'])
    return protocol


def expected_calibration_hash(protocol, sequence, filename):
    suffix = '/'.join((sequence, 'calibration', filename))
    found = [value for key, value in protocol['source_hashes'].items()
             if key == suffix or key.endswith('/' + suffix)]
    if len(found) != 1:
        raise ValueError('Missing/ambiguous calibration provenance: ' + suffix)
    return found[0]


def audit_input_files(rows, protocol, dsec_root, labels_root):
    """Hash the supplied frames' annotations, calibration and depth GT.

    Event HDF5/cache bytes are deliberately not rehashed on every allocation.
    Their content/cache equivalence belongs to the separate data preparation gate.
    """
    dsec_root, labels_root = Path(dsec_root), Path(labels_root)
    files = {}
    for row in rows:
        annotation = str(safe_relative(row['annotation_path']))
        key = 'labels/' + annotation
        if key not in files:
            value = sha256(labels_root / annotation)
            if value != protocol['source_hashes'].get(annotation):
                raise ValueError('Annotation differs from frozen protocol: ' + annotation)
            files[key] = value
        for filename in ('cam_to_cam.yaml', 'cam_to_lidar.yaml'):
            rel = str(safe_relative(row['sequence']) / 'calibration' / filename)
            key = 'dsec/' + rel
            if key not in files:
                value = sha256(dsec_root / rel)
                if value != expected_calibration_hash(protocol, row['sequence'], filename):
                    raise ValueError('Calibration differs from frozen protocol: ' + rel)
                files[key] = value
        rel = str(safe_relative(row['disparity_path']))
        key = 'dsec/' + rel
        if key not in files:
            files[key] = sha256(dsec_root / rel)
    return dict(frames=len(rows), rows_sha256=json_digest(rows), files=files,
                files_sha256=json_digest(files),
                scope='selected annotation/calibration/depth bytes; event/cache contents require separate gate')


def validate_anchor_dimensions(payload):
    anchors = payload['training_anchor_dimensions']
    if set(anchors) != set(CLASSES):
        raise ValueError('Target anchors must include Vehicle, Pedestrian and Cyclist')
    for name in CLASSES:
        row = anchors[name]
        for key in ('height', 'width', 'length', 'center_y'):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Nonfinite target anchor: %s/%s' % (name, key))
            if key != 'center_y' and value <= 0:
                raise ValueError('Nonpositive target anchor: %s/%s' % (name, key))
        if 'annotations' in row and (not isinstance(row['annotations'], int) or row['annotations'] < 0):
            raise ValueError('Invalid anchor annotation count')
    return anchors
