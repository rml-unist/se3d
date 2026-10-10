"""Freeze nested train-only chunk subsets and their own target anchors.

No held-out labels are inspected. Default prefixes select 403/992/1953 frames
of the frozen 3906-frame TRAIN pool (nominal 10/25/50%; report actual fractions).
Use --manifests-only to prepare identities without reading any annotation.
"""
import argparse
import json
from pathlib import Path

try:
    from . import protocol_utils as protocol
except ImportError:
    import protocol_utils as protocol


def prepare(args):
    base = protocol.load_protocol(args.protocol)
    base_hash = protocol.sha256(args.protocol)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(protocol_sha256=base_hash, subset_seed=args.subset_seed,
                   fractions=[], prepared_without_test_labels=True,
                   anchors_prepared=not args.manifests_only)
    for fraction in sorted(set(args.fractions)):
        subset = protocol.make_subset(base, base_hash, fraction, args.subset_seed)
        folder = output / ('fraction_%g' % fraction)
        path = folder / 'subset.json'
        if path.exists() and json.loads(path.read_text()) != subset:
            raise ValueError('Refusing to replace a different frozen subset: ' + str(path))
        protocol.atomic_json(subset, path)
        record = dict(nominal_percent=fraction, actual_percent=subset['actual_percent'],
                      chunks=subset['selected_chunks'], frames=subset['selected_frames'],
                      subset=str(path.resolve()), subset_sha256=protocol.sha256(path))
        if not args.manifests_only:
            anchors = protocol.derive_subset_anchors(base, base_hash, subset, protocol.sha256(path),
                                                      args.dsec_root, args.labels_root, args.fallback)
            target = folder / 'anchors.json'
            if target.exists() and json.loads(target.read_text()) != anchors:
                raise ValueError('Refusing to replace different subset anchors: ' + str(target))
            protocol.atomic_json(anchors, target)
            record.update(anchors=str(target.resolve()), anchors_sha256=protocol.sha256(target),
                          fallback_classes=anchors['provenance']['fallback_classes'])
        receipt['fractions'].append(record)
    protocol.atomic_json(receipt, output / 'SUBSETS.json')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(protocol.PROTOCOL))
    parser.add_argument('--output', required=True)
    parser.add_argument('--fractions', nargs='+', type=float, default=[10, 25, 50])
    parser.add_argument('--subset-seed', type=int, default=20261010)
    parser.add_argument('--dsec-root')
    parser.add_argument('--labels-root')
    parser.add_argument('--fallback', default=str(protocol.FALLBACK))
    parser.add_argument('--manifests-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.manifests_only and (not args.dsec_root or not args.labels_root):
        parser.error('Anchor preparation requires --dsec-root and --labels-root')
    if any(not 0 < fraction < 100 for fraction in args.fractions):
        parser.error('Fractions must be strictly between 0 and 100')
    print(json.dumps(prepare(args), indent=2))


if __name__ == '__main__':
    main()
