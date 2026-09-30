"""Build the SE3D release archives (maintainers only).

One archive per sequence plus SE3D_meta.tar.gz. Every archive unpacks into
SE3D/, so extracting all of them yields the layout documented in docs/DATASET.md:

    SE3D/calib.txt, README.txt, label_correction_manifest.csv, car_correction_decisions.json
    SE3D/<map>/<sequence>/
        events/{left,right}/{events.h5,rectify_map.h5}
        disparity/event/<frame>.npy, disparity/timestamps_with_label.txt
        image_2/, image_3/, depth_map/, dvs_2/, dvs_3/, velodyne/
        label/ (deduplicated), label_original/ (before deduplication)
        timestamps.txt, coords.txt, speeds.txt

    python tools/package_dataset.py --source /nas_data/SE3D/refined_new \
        --corrected-labels SE3D_labels_v3_20260925.zip --output /scratch/se3d_release --dry-run
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
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from se3d.data import load_splits

FILES = ['timestamps.txt', 'coords.txt', 'speeds.txt', 'disparity/timestamps_with_label.txt',
         'events/left/events.h5', 'events/left/rectify_map.h5', 'events/right/events.h5', 'events/right/rectify_map.h5']
FOLDERS = ['disparity/event', 'image_2', 'image_3', 'depth_map', 'dvs_2', 'dvs_3', 'velodyne']
LABELS_SHA256 = '1c542a13d3574f11700f6448334fcda5fc9ec45357ff17e0709a9edc6525eb7f'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, target)


def stage_sequence(source, labels, sequence, stage):
    """Symlink tree SE3D/<sequence>/ with the released files only."""
    src, dst = Path(source) / sequence, stage / 'SE3D' / sequence
    size = 0
    for name in FILES:
        link(src / name, dst / name)
        size += (src / name).stat().st_size
    for folder in FOLDERS:
        (dst / folder).mkdir(parents=True, exist_ok=True)
        for item in sorted((src / folder).iterdir()):
            link(item, dst / folder / item.name)
            size += item.stat().st_size
    (dst / 'label_original').mkdir(parents=True)
    for item in sorted((src / 'label_5').iterdir()):
        link(item, dst / 'label_original' / item.name)
    (dst / 'label').mkdir(parents=True)
    corrected = labels / 'labels' / sequence / 'label_5'
    for item in sorted(corrected.iterdir()):
        link(item, dst / 'label' / item.name)
    if len(list((dst / 'label').iterdir())) != len(list((dst / 'label_original').iterdir())):
        raise ValueError('%s: corrected and original label counts differ' % sequence)
    return size


def archive(stage, member, output, processes):
    with open(output, 'wb') as out:
        tar = subprocess.Popen(['tar', '-ch', '-C', str(stage), member], stdout=subprocess.PIPE)
        subprocess.run(['pigz', '-p', str(processes)], stdin=tar.stdout, stdout=out, check=True)
        if tar.wait():
            raise RuntimeError('tar failed for %s' % member)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='refined_new root')
    parser.add_argument('--corrected-labels', required=True, help='SE3D_labels_v3_20260925.zip')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequences', nargs='*', help='subset, e.g. map1/map1_day_rain_moving')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='only report the sizes')
    parser.add_argument('--no-meta', action='store_true', help='skip SE3D_meta.tar.gz (for parallel batches)')
    args = parser.parse_args()

    if sha256(args.corrected_labels) != LABELS_SHA256:
        raise ValueError('Unexpected corrected-label archive')
    sequences = args.sequences or sorted(load_splits()['sequences'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        labels = temporary / 'labels_v3'
        with zipfile.ZipFile(args.corrected_labels) as z:
            members = [m for m in z.namelist() if m.startswith('labels/') and
                       any(m.startswith('labels/%s/' % s) for s in sequences)]
            z.extractall(labels, members + ['corrected_label_manifest.csv', 'car_correction_decisions.json'])
        total = 0
        for sequence in sequences:
            stage = temporary / ('stage_' + sequence.replace('/', '_'))
            size = stage_sequence(args.source, labels, sequence, stage)
            total += size
            name = 'SE3D_%s.tar.gz' % sequence.split('/')[-1]
            if args.dry_run:
                print('%s: %.2f GB before compression' % (sequence, size / 1e9), flush=True)
            elif (output / name).exists() and name in read_manifest(output):
                print('%s already packaged' % name, flush=True)
            else:
                archive(stage, 'SE3D/' + sequence, output / name, args.processes)
                record(output, name, dict(sequence=sequence, bytes=(output / name).stat().st_size,
                                          sha256=sha256(output / name)))
            shutil.rmtree(stage)
        print('total before compression: %.1f GB' % (total / 1e9))
        if args.dry_run or args.no_meta:
            return
        meta = temporary / 'meta' / 'SE3D'
        meta.mkdir(parents=True)
        shutil.copy(Path(args.source) / 'calib.txt', meta / 'calib.txt')
        shutil.copy(labels / 'corrected_label_manifest.csv', meta / 'label_correction_manifest.csv')
        shutil.copy(labels / 'car_correction_decisions.json', meta / 'car_correction_decisions.json')
        shutil.copy(Path(__file__).resolve().parents[1] / 'docs' / 'DATASET.md', meta / 'README.md')
        archive(temporary / 'meta', 'SE3D', output / 'SE3D_meta.tar.gz', args.processes)
        record(output, 'SE3D_meta.tar.gz', dict(bytes=(output / 'SE3D_meta.tar.gz').stat().st_size,
                                                sha256=sha256(output / 'SE3D_meta.tar.gz')))


def read_manifest(output):
    path = output / 'manifest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def record(output, name, entry):
    """Add one archive to manifest.json and SHA256SUMS; safe for parallel batches."""
    with open(output / '.manifest.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = read_manifest(output)
        records[name] = entry
        (output / 'manifest.json').write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
        (output / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (r['sha256'], n) for n, r in sorted(records.items())))
    print(json.dumps({name: entry}), flush=True)


if __name__ == '__main__':
    main()
