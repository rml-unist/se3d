"""Score a target checkpoint on the 1,178 official DSEC-3DOD validation keyframes (the test set).

    python transfer/test.py --model dsgn_event --checkpoint runs/dsec_dsgn_se3d_s0909/best.pth \
        --dsec-root <DSEC>/train --labels-root <DSEC-3DOD> --metrics-python <metrics env python> \
        --output results/dsec_dsgn_se3d_s0909

Writes fixed_test_predictions.pkl and test_metrics.json (Waymo Level-1/2 AP and
APH per class, V/P AP, disparity MAE/RMSE/1PE/2PE). SE-CFF reports disparity only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=common.MODELS, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dsec-root', required=True)
    parser.add_argument('--labels-root', required=True)
    parser.add_argument('--cache-root', default=None)
    parser.add_argument('--metrics-python', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.model != 'se_cff' and not args.metrics_python:
        parser.error('--metrics-python is required for detection metrics')

    common.configure()
    model = common.build_model(args.model).cuda()
    state = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    ds = common.dataset('test', args.dsec_root, args.labels_root, args.cache_root, generate_target=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.model == 'se_cff':
        result = common.evaluate_depth(model, ds, args.workers, out, 'fixed_test')
    else:
        result = common.evaluate_detection(model, ds, args.workers, out, 'fixed_test', args.metrics_python)
    result.update(checkpoint_sha256=common.sha256(args.checkpoint), selected_epoch=state.get('epoch'))
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = dict(frames=result['frames'], disparity_MAE=result['depth']['MAE'])
    if 'metrics_percent' in result:
        m = result['metrics_percent']
        summary.update(vehicle_L2_AP=m['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'],
                       pedestrian_L2_AP=m['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'],
                       vp_L2_AP=result['selection_vehicle_pedestrian_L2_AP'])
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
