"""Score a target checkpoint on the 1,178 fixed DSEC-3DOD test keyframes.

Uses the checkpoint's own embedded anchors (including subset-only anchors).
Writes fixed_test_predictions.pkl and test_metrics.json. Inference resumes
from a bound partial prefix after SIGUSR1/SIGTERM or the allocation margin.
Legacy checkpoints require their matching protocol and anchor files; their
missing runtime provenance is recorded explicitly in the evaluation receipt.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from . import common, protocol_utils as protocol, runtime
except ImportError:
    import common
    import protocol_utils as protocol
    import runtime

import torch  # noqa: E402
from se3d.backends import restore_checkpoint_backend  # noqa: E402


def checkpoint_anchors(state, anchors_path=None):
    config = state.get('config', {})
    if 'anchors' in state:
        anchors = state['anchors']
        if protocol.json_digest(anchors) != config.get('anchor_payload_sha256'):
            raise ValueError('Embedded anchors differ from the checkpoint configuration')
        if anchors_path and json.loads(Path(anchors_path).read_text()) != anchors:
            raise ValueError('Explicit anchors differ from the checkpoint anchors')
    else:
        path = Path(anchors_path) if anchors_path else common.ANCHORS
        anchors = json.loads(path.read_text())
        if config.get('anchors_sha256') and protocol.sha256(path) != config['anchors_sha256']:
            raise ValueError('Legacy checkpoint requires its original anchor file')
    protocol.validate_anchor_dimensions(anchors)
    return anchors


def run(args, budget):
    checkpoint_hash = protocol.sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if protocol.sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed while it was being loaded')
    config = state.get('config', {})
    backend, backend_source = restore_checkpoint_backend(config)
    if config.get('model', args.model) != args.model:
        raise ValueError('Checkpoint architecture differs from --model')
    protocol_hash = protocol.sha256(args.protocol)
    if config.get('protocol_sha256') and protocol_hash != config['protocol_sha256']:
        raise ValueError('Checkpoint requires its matching protocol file')
    code = runtime.code_fingerprint(common.REPO_ROOT)
    if config.get('code_sha256') and config['code_sha256'] != code['sha256']:
        raise ValueError('Inference code differs from the frozen training code')
    anchors = checkpoint_anchors(state, args.anchors)
    common.configure(anchors)
    base = protocol.load_protocol(args.protocol)
    test_rows = base['splits']['test'][:args.limit_test]
    inputs = protocol.audit_input_files(test_rows, base, args.dsec_root, args.labels_root)
    # A separate inference receipt permits an explicitly reported evaluation
    # environment while preventing reuse of partial results across environments.
    identity = dict(checkpoint_sha256=checkpoint_hash, selected_step=state.get('step'),
                    selected_epoch=state.get('epoch'), selection_score=state.get('validation_score'),
                    training_config_sha256=protocol.json_digest(config), code_sha256=code['sha256'],
                    protocol_sha256=protocol_hash, anchor_payload_sha256=protocol.json_digest(anchors),
                    test_inputs=dict(rows_sha256=inputs['rows_sha256'], files_sha256=inputs['files_sha256']),
                    environment=runtime.environment(), metrics_environment=runtime.metrics_environment(args.metrics_python),
                    backend=backend, backend_source=backend_source,
                    legacy_checkpoint=state.get('schema') != runtime.SCHEMA, model=args.model,
                    device=args.device, debug_limit_test=args.limit_test, frames=len(test_rows))
    output = Path(args.output)
    runtime.ensure_identity(output / 'evaluation_config.json', identity)
    runtime.ensure_identity(output / 'input_manifest.json', inputs)
    runtime.ensure_identity(output / 'code_manifest.json', code)
    budget.check()
    model = common.build_model(args.model).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    del state
    ds = common.dataset('test', args.dsec_root, args.labels_root, args.cache_root,
                        generate_target=False, protocol_path=args.protocol)
    ds.rows = ds.rows[:args.limit_test]
    if args.model == 'se_cff':
        result = common.evaluate_depth(model, ds, args.workers, output, 'fixed_test',
                                       context=identity, check_stop=budget.check, device=args.device)
    else:
        result = common.evaluate_detection(model, ds, args.workers, output, 'fixed_test', args.metrics_python,
                                           context=identity, check_stop=budget.check, device=args.device)
    result.update(checkpoint_sha256=checkpoint_hash, selected_step=identity['selected_step'],
                  selected_epoch=identity['selected_epoch'], evaluation_config=identity)
    protocol.atomic_json(result, output / 'test_metrics.json')
    protocol.atomic_json(dict(status='complete', frames=len(ds)), output / 'allocation_exit.json')
    summary = dict(frames=result['frames'], disparity_MAE=result['depth']['MAE'])
    if 'metrics_percent' in result:
        metrics = result['metrics_percent']
        summary.update(vehicle_L2_AP=metrics['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'],
                       pedestrian_L2_AP=metrics['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'],
                       vp_L2_AP=result['selection_vehicle_pedestrian_L2_AP'])
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=common.MODELS, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dsec-root', required=True)
    parser.add_argument('--labels-root', required=True)
    parser.add_argument('--cache-root')
    parser.add_argument('--metrics-python')
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default=str(protocol.PROTOCOL))
    parser.add_argument('--anchors', help='legacy checkpoint anchors, or assert equality with embedded anchors')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--limit-test', type=int, help='debug only; outputs carry the restricted frame count')
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--save-margin-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.model != 'se_cff' and not args.metrics_python:
        parser.error('--metrics-python is required for detection metrics')
    if args.workers < 0 or (args.limit_test is not None and args.limit_test <= 0):
        parser.error('Invalid workers or test-frame limit')
    with runtime.Budget(args.max_seconds, args.save_margin_seconds) as budget, runtime.run_lock(args.output):
        try:
            return run(args, budget)
        except runtime.Preempted as exc:
            receipt = dict(status='checkpointed_for_resume', reason=str(exc))
            protocol.atomic_json(receipt, Path(args.output) / 'allocation_exit.json')
            print(json.dumps(receipt), flush=True)
            return runtime.RESUME_EXIT_CODE


if __name__ == '__main__':
    sys.exit(main())
