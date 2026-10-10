"""Paired DSEC-3DOD transfer with a fixed optimizer-update/validation budget.

Default: 62,496 updates, validation every 3,906 updates (16 candidates),
batch size 1, Adam 1e-4/weight decay 1e-4, no augmentation. Smaller labeled
subsets cycle their own seeded permutations under the identical update budget.
Only validation selects best.pth; exact ties retain the earlier candidate.

Resume with the same command. SIGUSR1/SIGTERM or an allocation time/update
limit saves last.pth and partial validation, then exits 75. A job wrapper may
resubmit this status. final.pth and training_complete.json mark full completion.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from . import common, protocol_utils as protocol, runtime
except ImportError:
    import common
    import protocol_utils as protocol
    import runtime

import numpy as np  # noqa: E402
import torch  # noqa: E402
from se3d.backends import add_backend_arguments, configure_backend  # noqa: E402


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=common.MODELS, required=True)
    parser.add_argument('--init', choices=['scratch', 'se3d'], required=True)
    parser.add_argument('--source', help='SE3D checkpoint for --init se3d')
    parser.add_argument('--source-sha256', help='require this immutable source checkpoint hash')
    parser.add_argument('--source-seed', type=int, help='require this source training seed')
    parser.add_argument('--source-step', type=int, help='require this source optimizer step (new matrix: 214368)')
    parser.add_argument('--dsec-root', required=True, help='DSEC train/ directory')
    parser.add_argument('--labels-root', required=True, help='DSEC-3DOD annotation directory')
    parser.add_argument('--cache-root', help='event stacks from prepare_cache.py')
    parser.add_argument('--metrics-python', help='interpreter with official Waymo CPU metrics')
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default=str(protocol.PROTOCOL))
    parser.add_argument('--subset', help='frozen subset.json from prepare_subsets.py')
    parser.add_argument('--anchors', help='subset anchors.json; required with --subset')
    parser.add_argument('--fallback', default=str(protocol.FALLBACK))
    parser.add_argument('--epochs', type=int, help='legacy budget shorthand: epochs times the full train pool')
    parser.add_argument('--updates', type=int, help='exact optimizer budget; default 62496')
    parser.add_argument('--validate-every', type=int, help='optimizer updates between validations; default 3906')
    parser.add_argument('--seed', type=int, default=20260909, help='target optimization seed')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU is for small runtime gates')
    add_backend_arguments(parser)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--keep-updates', nargs='*', type=int, default=[])
    parser.add_argument('--max-seconds', type=float, default=0, help='allocation wall budget, including startup; 0 disables')
    parser.add_argument('--save-margin-seconds', type=float, default=600)
    parser.add_argument('--allocation-updates', type=int, default=0, help='checkpoint and exit after N more updates; 0 disables')
    parser.add_argument('--limit-train', type=int, help='debug: cap each selected-pool permutation')
    parser.add_argument('--limit-val', type=int, help='debug: restrict validation keyframes')
    args = parser.parse_args(argv)
    if args.init == 'se3d' and not args.source:
        parser.error('--init se3d requires --source')
    if args.init == 'scratch' and any(value is not None for value in
                                    (args.source, args.source_sha256, args.source_seed, args.source_step)):
        parser.error('Scratch runs cannot specify source checkpoint arguments')
    if args.model != 'se_cff' and not args.metrics_python:
        parser.error('--metrics-python is required to select detection checkpoints')
    if args.subset and not args.anchors:
        parser.error('--subset requires its own --anchors receipt')
    if args.updates is not None and args.epochs is not None:
        parser.error('Specify --updates or the legacy --epochs shorthand, not both')
    for field in ('updates', 'epochs', 'validate_every', 'limit_train', 'limit_val', 'source_step'):
        if getattr(args, field) is not None and getattr(args, field) <= 0:
            parser.error('--' + field.replace('_', '-') + ' must be positive')
    if args.workers < 0 or args.save_every <= 0 or args.allocation_updates < 0:
        parser.error('Invalid worker/checkpoint/update-limit argument')
    if args.strict_determinism and args.backend_profile == 'historical':
        parser.error('--strict-determinism cannot be combined with --backend-profile historical')
    return args


def source_checkpoint(args):
    if args.init == 'scratch':
        return None, None
    digest = protocol.sha256(args.source)
    if args.source_sha256 and digest != args.source_sha256:
        raise ValueError('Source checkpoint SHA256 differs from the frozen source')
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    if protocol.sha256(args.source) != digest:
        raise ValueError('Source checkpoint changed while it was being loaded')
    config = state.get('config', {})
    seed, step = config.get('seed'), state.get('step')
    if args.source_seed is not None and seed != args.source_seed:
        raise ValueError('Source seed mismatch (or missing source seed metadata)')
    if args.source_step is not None and step != args.source_step:
        raise ValueError('Source update mismatch (or missing source step metadata)')
    expected_model = 'emod' if args.model == 'se_cff' else args.model
    # Historical checkpoints used descriptive labels. The complete shared-key
    # and shape check in initialize_from_se3d still verifies the architecture.
    declared_model = config.get('model', expected_model)
    aliases = {'DSGN-event adaptation': 'dsgn_event', 'DSGN-event': 'dsgn_event', 'EMOD': 'emod'}
    if aliases.get(declared_model, declared_model) != expected_model:
        raise ValueError('Source checkpoint model differs from the target architecture')
    return state, dict(checkpoint_sha256=digest, source_seed=seed, source_step=step,
                       source_epoch=state.get('epoch'), source_config=config,
                       source_config_sha256=protocol.json_digest(config))


def prepare_inputs(args):
    base, subset, rows = protocol.select_train_rows(args.protocol, args.subset)
    for split, expected in common.SPLIT_SIZES.items():
        if len(base['splits'][split]) != expected:
            raise ValueError('Transfer protocol has a different ' + split + ' size')
    anchors_path = Path(args.anchors) if args.anchors else common.ANCHORS
    anchors = json.loads(anchors_path.read_text())
    protocol.validate_anchor_dimensions(anchors)
    if subset is not None:
        expected = protocol.derive_subset_anchors(base, protocol.sha256(args.protocol), subset,
                                                  protocol.sha256(args.subset), args.dsec_root,
                                                  args.labels_root, args.fallback)
        if anchors != expected:
            raise ValueError('Subset anchors do not match selected-only training labels and frozen fallback')
    elif anchors != json.loads(common.ANCHORS.read_text()):
        raise ValueError('The full-pool protocol uses the frozen full-pool target anchors')
    val_rows = base['splits']['validation'][:args.limit_val]
    inputs = {name: protocol.audit_input_files(values, base, args.dsec_root, args.labels_root)
              for name, values in [('train', rows), ('validation', val_rows)]}
    return base, subset, rows, anchors_path, anchors, inputs


def run(args, budget):
    output = Path(args.output)
    backend = configure_backend(args.backend_profile, args.strict_determinism)
    base, subset, rows, anchors_path, anchors, inputs = prepare_inputs(args)
    common.configure(anchors)
    source, source_info = source_checkpoint(args)
    code = runtime.code_fingerprint(common.REPO_ROOT)
    train_size = len(base['splits']['train'])
    legacy_cycle = min(train_size, args.limit_train) if args.limit_train else train_size
    # An explicit --epochs retains short historical debug budgets. The new matrix
    # always passes --updates/--validate-every, including for smaller label subsets.
    updates = args.updates if args.updates is not None else (args.epochs * legacy_cycle if args.epochs else 62496)
    interval = args.validate_every if args.validate_every is not None else (legacy_cycle if args.epochs else 3906)
    candidates = runtime.validation_steps(updates, interval)
    detection = args.model != 'se_cff'
    config = dict(schema=runtime.SCHEMA, model=args.model, init=args.init, seed=args.seed, batch_size=1,
                  optimizer='Adam', learning_rate=1e-4, weight_decay=1e-4, augmentation='none',
                  loss_weights=[.5, .5] if detection else 'SE-CFF weighted pyramid SmoothL1, valid-pixel mean',
                  deform_offset_learning_rate=None if detection else 1e-5,
                  optimizer_updates=updates, validate_every=interval, validation_steps=candidates,
                  training_pool_frames=train_size, training_frames=len(rows),
                  validation_frames=len(base['splits']['validation'][:args.limit_val]), test_frames=len(base['splits']['test']),
                  selection='Max validation mean Vehicle/Pedestrian official L2 AP; earliest tie' if detection
                  else 'Minimum validation disparity MAE; earliest tie',
                  sampler='torch.randperm(selected_frames, seed + data_epoch), cycle until update limit',
                  protocol_sha256=protocol.sha256(args.protocol),
                  subset_sha256=protocol.sha256(args.subset) if args.subset else None,
                  subset=subset, anchors_sha256=protocol.sha256(anchors_path),
                  anchor_payload_sha256=protocol.json_digest(anchors), source=source_info,
                  input_hashes={name: dict(rows_sha256=value['rows_sha256'], files_sha256=value['files_sha256'])
                                for name, value in inputs.items()}, code_sha256=code['sha256'],
                  environment=runtime.environment(), metrics_environment=runtime.metrics_environment(args.metrics_python),
                  device=args.device, workers=args.workers, backend=backend,
                  strict_determinism=args.strict_determinism,
                  limit_train=args.limit_train, limit_val=args.limit_val,
                  keep_updates=sorted(set(args.keep_updates)))
    runtime.ensure_identity(output / 'effective_config.json', config)
    runtime.ensure_identity(output / 'code_manifest.json', code)
    runtime.ensure_identity(output / 'input_manifest.json', inputs)
    runtime.ensure_identity(output / 'anchors.json', anchors)
    budget.check()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = common.build_model(args.model).to(args.device)
    initial = model.state_dict()
    prefixes = common.RESET_PREFIXES[args.model]
    reset = {name: value for name, value in initial.items() if name.startswith(prefixes)}
    initialization = dict(seed=args.seed, reset_prefixes=list(prefixes), reset_tensors=sorted(reset),
                          seeded_model_sha256=runtime.tensor_state_sha256(initial),
                          seeded_reset_tensors_sha256=runtime.tensor_state_sha256(reset))
    copied = []
    if source is not None:
        copied = common.initialize_from_se3d(model, args.model, source_state=source['model'])
        after = {name: value for name, value in model.state_dict().items() if name.startswith(prefixes)}
        if runtime.tensor_state_sha256(after) != initialization['seeded_reset_tensors_sha256']:
            raise ValueError('Source initialization changed a seeded target prediction head')
    initialization.update(copied_tensors=copied, initialized_model_sha256=runtime.tensor_state_sha256(model.state_dict()),
                          source_checkpoint_sha256=source_info['checkpoint_sha256'] if source_info else None)
    del source, initial, reset
    runtime.ensure_identity(output / 'initialization.json', initialization)
    runtime.ensure_identity(output / 'transfer_tensors.json', copied)
    optimizer = torch.optim.Adam(model.parameters() if detection else model.get_params_group(1e-4),
                                 lr=1e-4, weight_decay=1e-4)
    roots = dict(dsec_root=args.dsec_root, labels_root=args.labels_root, cache_root=args.cache_root,
                 protocol_path=args.protocol)
    train = common.dataset('train', generate_target=detection, train_rows=rows, **roots)
    validation = common.dataset('validation', generate_target=False, **roots)
    validation.rows = validation.rows[:args.limit_val]
    from se3d.data import model_args
    from se3d.engine import loader

    def compute_loss(batch):
        if detection:
            _, _, depth_loss, detection_loss = model(**model_args(batch, device=args.device))
            loss = .5 * depth_loss.mean() + .5 * detection_loss
            return loss, dict(depth_loss=float(depth_loss.mean().detach()), detection_loss=float(detection_loss.detach()))
        _, loss_vector = model(**common.depth_args(batch, device=args.device))
        return (loss_vector.mean() if loss_vector.numel() else loss_vector.sum()), {}

    def validate(step, check_stop):
        tag = 'validation_step_%07d' % step
        context = dict(config_sha256=protocol.json_digest(config), validation_inputs=config['input_hashes']['validation'],
                       step=step, code_sha256=config['code_sha256'])
        if detection:
            result = common.evaluate_detection(model, validation, args.workers, output, tag, args.metrics_python,
                                               context=context, check_stop=check_stop, device=args.device)
            score = result['selection_vehicle_pedestrian_L2_AP']
        else:
            result = common.evaluate_depth(model, validation, args.workers, output, tag, context=context,
                                           check_stop=check_stop, device=args.device)
            score = -result['depth']['MAE'] if result['depth']['MAE'] is not None else float('nan')
        return result, score

    return runtime.train_loop(model, optimizer, output, config, anchors, initialization,
                              lambda indices, epoch: loader(train, indices, args.workers, args.seed + epoch),
                              compute_loss, validate, budget.check, args.save_every, args.keep_updates,
                              args.allocation_updates)


def main(argv=None):
    args = arguments(argv)
    with runtime.Budget(args.max_seconds, args.save_margin_seconds) as budget, runtime.run_lock(args.output):
        try:
            return run(args, budget)
        except runtime.Preempted as exc:
            # Startup has not modified training state; any prior last.pth is intact.
            print(json.dumps(dict(status='startup_interrupted_for_resume', reason=str(exc))), flush=True)
            return runtime.RESUME_EXIT_CODE


if __name__ == '__main__':
    sys.exit(main())
