"""Transfer split, subset and anchor provenance; no model or CUDA imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'protocol' / 'dsec_paired_v2.json'
FALLBACK = ROOT / 'protocol' / 'absent_class_anchors_v1.json'
CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')
SUBSET_SCHEMA = 'dsec-nested-train-chunks-v1'
ANCHOR_SCHEMA = 'dsec-subset-anchors-v1'
SUBSET_SALT = 'se3d-target-label-efficiency-chunks-v1'


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


def make_subset(protocol, protocol_sha256, nominal_percent, subset_seed=20261010):
    """Hash whole TRAIN chunks; never rank by annotations or held-out performance."""
    if not 0 < nominal_percent < 100:
        raise ValueError('Subset percentage must be strictly between 0 and 100')
    nominal_percent = int(nominal_percent) if float(nominal_percent).is_integer() else float(nominal_percent)
    train = protocol['splits']['train']
    salt = '%s/%d/' % (SUBSET_SALT, subset_seed)
    chunks = sorted({row['chunk'] for row in train},
                    key=lambda c: (hashlib.sha256((salt + c).encode()).hexdigest(), c))
    count = int(math.ceil(nominal_percent * len(chunks) / 100))
    selected_chunks = chunks[:count]
    chosen = set(selected_chunks)
    indices = [i for i, row in enumerate(train) if row['chunk'] in chosen]
    rows = [train[i] for i in indices]
    return dict(schema=SUBSET_SCHEMA, protocol_sha256=protocol_sha256, subset_seed=int(subset_seed),
                nominal_percent=nominal_percent, actual_percent=100 * len(rows) / len(train),
                selection_unit='whole train chunks', hash_salt=salt, chunks=selected_chunks,
                training_pool_frames=len(train), training_pool_chunks=len(chunks),
                selected_frames=len(rows), selected_chunks=len(chosen), train_indices=indices,
                selected_rows_sha256=json_digest(rows),
                supervision='box and disparity labels from selected frames only',
                subset_changes_with_target_seed=False)


def validate_subset(subset, protocol, protocol_sha256):
    """Require the exact prespecified hash prefix, not just a plausible frame count."""
    if subset.get('schema') != SUBSET_SCHEMA:
        raise ValueError('Unsupported subset schema')
    expected = make_subset(protocol, protocol_sha256, subset['nominal_percent'], subset['subset_seed'])
    if subset != expected:
        raise ValueError('Subset differs from its protocol/hash-prefix definition')
    rows = [protocol['splits']['train'][i] for i in subset['train_indices']]
    heldout = {row['chunk'] for split in ('validation', 'test')
               for row in protocol['splits'][split]}
    if heldout.intersection(subset['chunks']):
        raise ValueError('Subset includes a held-out chunk')
    return rows


def select_train_rows(protocol_path=PROTOCOL, subset_path=None):
    protocol = load_protocol(protocol_path)
    subset = json.loads(Path(subset_path).read_text()) if subset_path else None
    rows = (validate_subset(subset, protocol, sha256(protocol_path)) if subset is not None
            else list(protocol['splits']['train']))
    return protocol, subset, rows


def expected_calibration_hash(protocol, sequence, filename):
    suffix = '/'.join((sequence, 'calibration', filename))
    found = [value for key, value in protocol['source_hashes'].items()
             if key == suffix or key.endswith('/' + suffix)]
    if len(found) != 1:
        raise ValueError('Missing/ambiguous calibration provenance: ' + suffix)
    return found[0]


def audit_input_files(rows, protocol, dsec_root, labels_root, include_disparity=True):
    """Hash only selected annotations/calibration and optional selected depth GT.

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
        if include_disparity:
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


def derive_subset_anchors(protocol, protocol_sha256, subset, subset_sha256,
                          dsec_root, labels_root, fallback_path=FALLBACK):
    """Read only selected train chunks; medians follow the historical target rule."""
    import pickle
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT.parent))
    import se3d  # noqa: F401 (legacy model-package import paths)
    from lib.datasets.dsec_original import load_calibration, lidar_boxes_to_camera

    rows = validate_subset(subset, protocol, protocol_sha256)
    audit = audit_input_files(rows, protocol, dsec_root, labels_root, include_disparity=False)
    fallback = json.loads(Path(fallback_path).read_text())
    validate_anchor_dimensions(fallback)
    annotations, calibrations = {}, {}
    pool = {name: [] for name in CLASSES}
    frame_counts = {name: 0 for name in CLASSES}
    class_chunks = {name: set() for name in CLASSES}
    class_sequences = {name: set() for name in CLASSES}
    empty_frames = 0
    for row in rows:
        rel, sequence = row['annotation_path'], row['sequence']
        if rel not in annotations:
            annotations[rel] = pickle.loads((Path(labels_root) / rel).read_bytes())
        annotation = annotations[rel][row['frame']]
        if int(annotation['time_stamp']) != row['timestamp_us']:
            raise ValueError('Selected training timestamp differs from protocol')
        if sequence not in calibrations:
            calibrations[sequence] = load_calibration(Path(dsec_root) / sequence,
                                                      annotation['image']['image_0_extrinsic'])
        calibration = calibrations[sequence]
        if not np.allclose(calibration['event_to_lidar'], annotation['image']['image_0_extrinsic'],
                           atol=1e-10, rtol=0):
            raise ValueError('Annotation extrinsic varies within selected sequence')
        boxes, _ = lidar_boxes_to_camera(annotation['annos']['gt_boxes_lidar'], calibration)
        names = annotation['annos']['name']
        if len(boxes) != len(names):
            raise ValueError('Training annotation class/box count mismatch')
        if not len(names):
            empty_frames += 1
        for name in set(map(str, names)):
            if name not in pool:
                raise ValueError('Unexpected target training class: ' + name)
            frame_counts[name] += 1
            class_chunks[name].add(row['chunk'])
            class_sequences[name].add(sequence)
        for name, box in zip(names, boxes):
            name = str(name)
            if name not in pool:
                raise ValueError('Unexpected target training class: ' + name)
            values = [*box[:3], box[4] - box[0] / 2]
            if not np.isfinite(values).all() or not (np.asarray(values[:3]) > 0).all():
                raise ValueError('Invalid dimensions in selected training annotation')
            pool[name].append(values)
    anchors, absent = {}, []
    for name in CLASSES:
        if pool[name]:
            median = np.median(pool[name], axis=0)
            anchors[name] = dict(zip(('height', 'width', 'length', 'center_y'), median.tolist()))
            anchors[name]['annotations'] = len(pool[name])
        else:
            absent.append(name)
            anchors[name] = {key: fallback['training_anchor_dimensions'][name][key]
                             for key in ('height', 'width', 'length', 'center_y')}
            anchors[name]['annotations'] = 0
    payload = dict(schema=ANCHOR_SCHEMA, training_anchor_dimensions=anchors,
                   training_audit=dict(frames=len(rows), chunks=len({row['chunk'] for row in rows}),
                                       recordings=len({row['sequence'] for row in rows}), empty_frames=empty_frames,
                                       per_class={name: dict(boxes=len(pool[name]), frames=frame_counts[name],
                                                             chunks=len(class_chunks[name]),
                                                             recordings=len(class_sequences[name]))
                                                  for name in CLASSES}),
                   provenance=dict(protocol_sha256=protocol_sha256, subset_sha256=subset_sha256,
                                   selected_rows_sha256=json_digest(rows),
                                   selected_annotation_and_calibration_files=audit['files'],
                                   selected_files_sha256=audit['files_sha256'],
                                   fallback_sha256=sha256(fallback_path),
                                   fallback_classes=absent,
                                   computation='per-class median h,w,l and camera center_y of selected raw train boxes',
                                   excluded_annotations_read=False))
    validate_anchor_dimensions(payload)
    return payload
