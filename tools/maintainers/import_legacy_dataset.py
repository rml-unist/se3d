"""Convert the archived internal layout into the public SE3D layout (maintainers only).

Annotations are copied into label/ and label_original/. Large sensor files and
existing event caches are linked, so the original capture is never renamed.
The verified annotation archive is the only place this importer reads label_5.
Use package_dataset.py to create self-contained archives from the resulting tree.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from se3d import CLASSES, REPO_ROOT
from se3d.protocol import load_splits

ANNOTATIONS_SHA256 = '1c542a13d3574f11700f6448334fcda5fc9ec45357ff17e0709a9edc6525eb7f'
CACHE_NAME = 'sbn_5000000_MixedDensityEventStacking_10_0'
FILES = ('timestamps.txt', 'coords.txt', 'speeds.txt')
FOLDERS = ('disparity', 'image_2', 'image_3', 'depth_map', 'dvs_2', 'dvs_3', 'velodyne')
EXPECTED = {'files': 48304, 'label': 175123, 'label_original': 184884}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.absolute()
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError('Existing link points elsewhere: %s' % target)
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(source)


def write_verified(target, content):
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError('Existing annotation differs: %s' % target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='archived internal capture root')
    parser.add_argument('--annotations-archive', required=True, help='verified deduplicated annotation ZIP')
    parser.add_argument('--output', required=True, help='new public-layout dataset root')
    parser.add_argument('--cache-overlay', help='additional prepared event caches, indexed by map/sequence/events')
    parser.add_argument('--cache-time-bounds', help='copy verified metadata for cache-only input')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('source and output must be separate trees')
    if sha256(args.annotations_archive) != ANNOTATIONS_SHA256:
        raise ValueError('Annotation archive checksum mismatch')
    output.mkdir(parents=True, exist_ok=True)
    totals = {name: Counter() for name in ('label', 'label_original')}
    digests = {name: hashlib.sha256() for name in totals}
    counts, cached = {}, 0
    with zipfile.ZipFile(args.annotations_archive) as archive:
        for sequence in sorted(load_splits()['sequences']):
            src, dst = source / sequence, output / sequence
            dst.mkdir(parents=True, exist_ok=True)
            for name in FILES + FOLDERS:
                if (src / name).exists():
                    link(src / name, dst / name)
            for side in ('left', 'right'):
                if (src / 'events' / side).exists():
                    link(src / 'events' / side, dst / 'events' / side)
            cache_sources = [src / 'events' / CACHE_NAME]
            if args.cache_overlay:
                cache_sources.append(Path(args.cache_overlay) / sequence / 'events' / CACHE_NAME)
            for cache in cache_sources:
                for item in sorted(cache.glob('*.npy')):
                    target = dst / 'events' / CACHE_NAME / item.name
                    if not target.is_symlink() and not target.exists():
                        link(item, target)
                        cached += 1
            frames = sorted((src / 'label_5').glob('*.txt'))
            if not frames:
                raise ValueError('No archived annotations: %s' % src)
            expected_names = {'labels/%s/label_5/%s' % (sequence, frame.name) for frame in frames}
            present_names = {n for n in archive.namelist() if n.startswith('labels/%s/label_5/' % sequence)
                             and n.endswith('.txt')}
            if expected_names != present_names:
                raise ValueError('Original and release annotation file lists differ: %s' % sequence)
            for frame in frames:
                versions = {'label_original': frame.read_bytes(),
                            'label': archive.read('labels/%s/label_5/%s' % (sequence, frame.name))}
                for name, content in versions.items():
                    relative = '%s/%s/%s' % (sequence, name, frame.name)
                    write_verified(output / relative, content)
                    digests[name].update(relative.encode() + b'\0' + content + b'\0')
                    for line in content.decode().splitlines():
                        fields = line.split()
                        if not fields:
                            continue
                        if len(fields) != 15 or fields[0] not in CLASSES:
                            raise ValueError('Invalid annotation: ' + relative)
                        totals[name][fields[0]] += 1
            counts[sequence] = len(frames)
            print(json.dumps(dict(sequence=sequence, annotation_files=len(frames))), flush=True)
        for original, public in [('corrected_label_manifest.csv', 'label_correction_manifest.csv'),
                                 ('car_correction_decisions.json', 'car_correction_decisions.json')]:
            write_verified(output / public, archive.read(original))
    observed = {'files': sum(counts.values()), **{name: sum(value.values()) for name, value in totals.items()}}
    if observed != EXPECTED:
        raise ValueError('Annotation totals do not match the release: %r' % observed)
    link(source / 'calib.txt', output / 'calib.txt')
    shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET_LICENSE.md', output / 'LICENSE')
    shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET.md', output / 'README.md')
    if args.cache_time_bounds:
        write_verified(output / 'cache_time_bounds.json', Path(args.cache_time_bounds).read_bytes())
    report = dict(schema='se3d-public-layout-v1', annotations_archive_sha256=ANNOTATIONS_SHA256,
                  annotation_sha256={name: d.hexdigest() for name, d in digests.items()},
                  annotation_files_by_sequence=counts, objects_by_class=totals, totals=observed,
                  license='MIT', source=str(source), sensor_storage='symlinks; dereference when packaging')
    (output / 'dataset_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verified=True, totals=observed, newly_linked_cache_files=cached)), flush=True)


if __name__ == '__main__':
    main()
