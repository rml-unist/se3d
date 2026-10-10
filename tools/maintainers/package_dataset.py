"""Package an already verified public-layout SE3D tree (maintainers only).

Input is the same layout consumed by tools/train.py and tools/check_dataset.py:
label/ contains release annotations and label_original/ the historical set.
Use import_legacy_dataset.py once to convert the archived internal capture.

Each sequence archive includes the dataset MIT license. SE3D_meta.tar.gz holds
calibration, annotation provenance, license and documentation. Symlinks in a
working tree are dereferenced; released archives have no external dependency.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from se3d import REPO_ROOT
from se3d.protocol import load_splits
from tools.maintainers.import_legacy_dataset import public_annotation_metadata

FILES = ('timestamps.txt', 'coords.txt', 'speeds.txt', 'disparity/timestamps_with_label.txt',
         'events/left/events.h5', 'events/left/rectify_map.h5',
         'events/right/events.h5', 'events/right/rectify_map.h5')
FOLDERS = ('disparity/event', 'image_2', 'image_3', 'depth_map', 'dvs_2', 'dvs_3', 'velodyne',
           'label', 'label_original')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stage_sequence(source, sequence, stage):
    src, dst = source / sequence, stage / 'SE3D' / sequence
    size = 0
    for name in FILES:
        item = src / name
        if not item.is_file():
            raise FileNotFoundError(item)
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        (dst / name).symlink_to(item)
        size += item.stat().st_size
    for name in FOLDERS:
        folder = src / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        (dst / name).symlink_to(folder)
        size += sum(item.stat().st_size for item in folder.iterdir() if item.is_file())
    original = {p.name for p in (src / 'label_original').glob('*.txt')}
    release = {p.name for p in (src / 'label').glob('*.txt')}
    if not release or release != original:
        raise ValueError('Annotation file lists differ: ' + sequence)
    shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET_LICENSE.md', stage / 'SE3D' / 'LICENSE')
    return size


def archive(stage, members, output, processes):
    temporary = output.with_name(output.name + '.partial')
    with temporary.open('wb') as out:
        tar = subprocess.Popen(['tar', '-ch', '-C', str(stage), *members], stdout=subprocess.PIPE)
        try:
            subprocess.run(['pigz', '-p', str(processes)], stdin=tar.stdout, stdout=out, check=True)
        finally:
            tar.stdout.close()
            status = tar.wait()
        if status:
            raise RuntimeError('tar failed: ' + repr(members))
    temporary.replace(output)


def read_manifest(output):
    path = output / 'manifest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def record(output, name, entry):
    with (output / '.manifest.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = read_manifest(output)
        records[name] = entry
        (output / 'manifest.json').write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
        (output / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (r['sha256'], n) for n, r in sorted(records.items())))
    print(json.dumps({name: entry}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='verified public-layout SE3D root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequences', nargs='*', help='subset, e.g. map1/map1_day_rain_moving')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='validate input and report uncompressed sizes')
    parser.add_argument('--meta-only', action='store_true', help='package only metadata and the dataset license')
    parser.add_argument('--no-meta', action='store_true', help='skip metadata when packaging parallel batches')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    sequences = [] if args.meta_only else args.sequences or sorted(load_splits()['sequences'])
    source_manifest = source / 'dataset_manifest.json'
    provenance = sha256(source_manifest) if source_manifest.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        total = 0
        for sequence in sequences:
            stage = temporary / sequence.replace('/', '_')
            size = stage_sequence(source, sequence, stage)
            total += size
            name = 'SE3D_%s.tar.gz' % sequence.split('/')[-1]
            existing = read_manifest(output).get(name)
            if args.dry_run:
                print('%s: %.2f GB before compression' % (sequence, size / 1e9), flush=True)
            elif existing and (output / name).is_file():
                if existing.get('dataset_manifest_sha256') != provenance or existing.get('license') != 'MIT':
                    raise ValueError('Output holds a different release: ' + name)
                print(name + ' already packaged', flush=True)
            else:
                archive(stage, ['SE3D/' + sequence, 'SE3D/LICENSE'], output / name, args.processes)
                record(output, name, dict(sequence=sequence, bytes=(output / name).stat().st_size,
                    sha256=sha256(output / name), dataset_manifest_sha256=provenance, license='MIT'))
            shutil.rmtree(stage)
        print('total before compression: %.1f GB' % (total / 1e9), flush=True)
        if args.dry_run or args.no_meta:
            return
        meta = temporary / 'meta' / 'SE3D'
        meta.mkdir(parents=True)
        shutil.copyfile(source / 'calib.txt', meta / 'calib.txt')
        for name in ('label_correction_manifest.csv', 'car_correction_decisions.json'):
            (meta / name).write_bytes(public_annotation_metadata(name, (source / name).read_bytes()))
        if source_manifest.exists():
            public_manifest = json.loads(source_manifest.read_text())
            public_manifest.pop('source', None)
            public_manifest['sensor_storage'] = 'self-contained files in sequence archives'
            public_manifest['provenance_paths'] = {
                'path': 'public label/ annotation',
                'original_path': 'public label_original/ annotation',
            }
            (meta / 'dataset_manifest.json').write_text(json.dumps(public_manifest, indent=2) + '\n')
        shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET.md', meta / 'README.md')
        shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET_LICENSE.md', meta / 'LICENSE')
        shutil.copyfile(REPO_ROOT / 'docs' / 'DATASET_LICENSE.md', meta / 'DATASET_LICENSE.md')
        shutil.copyfile(REPO_ROOT / 'docs' / 'EXPERIMENTS.md', meta / 'EXPERIMENTS.md')
        shutil.copytree(REPO_ROOT / 'splits', meta / 'splits')
        (meta / 'benchmarks').mkdir()
        shutil.copyfile(REPO_ROOT / 'docs' / 'benchmarks' / 'epoch8_label_comparison.json',
                        meta / 'benchmarks' / 'epoch8_label_comparison.json')
        archive(temporary / 'meta', ['SE3D'], output / 'SE3D_meta.tar.gz', args.processes)
        record(output, 'SE3D_meta.tar.gz', dict(bytes=(output / 'SE3D_meta.tar.gz').stat().st_size,
               sha256=sha256(output / 'SE3D_meta.tar.gz'), dataset_manifest_sha256=provenance, license='MIT'))


if __name__ == '__main__':
    main()
