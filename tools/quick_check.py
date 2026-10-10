"""Check loading and inference using one downloaded SE3D sequence and two frames.

This is a smoke check, not a benchmark score. Download SE3D_meta.tar.gz and
SE3D_map3_day_sunny_moving.tar.gz, extract both, then run this script. A released
checkpoint is optional; without one, the model uses random initialization.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from se3d.backends import configure_backend, restore_checkpoint_backend
from se3d.data import SE3DFrames
from se3d.engine import predict
from se3d.models import ANCHORS, MODELS, anchor_name, build_model, configure_classes
from se3d.protocol import add_labels_argument
from tools.test import anchors_for, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', choices=MODELS, default='dsgn_event')
    parser.add_argument('--checkpoint')
    parser.add_argument('--anchors', type=anchor_name, choices=sorted(ANCHORS))
    parser.add_argument('--sequence', default='map3/map3_day_sunny_moving')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--frames', type=int, default=2)
    parser.add_argument('--cache-root')
    parser.add_argument('--cache-time-bounds')
    parser.add_argument('--output', default='results/quick_check.json')
    add_labels_argument(parser)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    torch.manual_seed(20260909)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else None
    if checkpoint:
        backend, backend_source = restore_checkpoint_backend(checkpoint.get('config', {}))
    else:
        backend, backend_source = configure_backend('historical'), 'random_initialization_default'
    anchors = anchors_for(checkpoint, args.anchors) if checkpoint else ANCHORS[args.anchors or 'label']
    configure_classes(anchors)
    model = build_model(args.model).cuda()
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    ds = SE3DFrames(args.data_root, args.split, label_dir=args.labels, generate_target=False,
                    sequences=[args.sequence], max_frames=args.frames, cache_root=args.cache_root,
                    cache_time_bounds=args.cache_time_bounds)
    out = predict(model, ds, workers=0, log_every=1)
    report = dict(passed=True, scope='input loading and finite inference; not a benchmark', model=args.model,
                  sequence=args.sequence, split=args.split, labels=args.labels, frames=len(ds),
                  checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
                  backend=backend, backend_source=backend_source,
                  anchors_sha256=sha256(anchors), metadata=out['metadata'],
                  predictions_per_frame=[len(a['name']) for a in out['predictions']],
                  valid_disparity_pixels=[int(v[4]) for v in out['depth_sums']])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
