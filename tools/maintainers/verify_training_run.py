"""Verify a completed release run's budget, validation selection and checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def verify(run):
    run = Path(run)
    config_path = run / 'effective_config.json'
    config = json.loads(config_path.read_text())
    complete = json.loads((run / 'training_complete.json').read_text())
    last = torch.load(run / 'last.pth', map_location='cpu', weights_only=False)
    best = torch.load(run / 'best.pth', map_location='cpu', weights_only=False)
    if complete['config_sha256'] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError('Completion receipt configuration hash mismatch')
    if last['config'] != config or best['config'] != config:
        raise ValueError('Checkpoint configuration mismatch')
    if last['step'] != config['updates'] or complete['steps'] != config['updates']:
        raise ValueError('Training update budget was not completed')
    interval = config['validation_interval_updates']
    steps = sorted(set(list(range(interval, config['updates'] + 1, interval)) + [config['updates']]))
    metrics = [json.loads(p.read_text()) for p in sorted(run.glob('validation_step_*.json'))]
    if [m['optimizer_steps'] for m in metrics] != steps or complete['validations'] != len(steps):
        raise ValueError('Unexpected validation schedule')
    metric = config['selection_metric']
    selected = max(metrics, key=lambda m: m[metric][1])  # stable max keeps the earliest tie
    if best['step'] != selected['optimizer_steps'] or best['validation_score'] != selected[metric][1]:
        raise ValueError('Best checkpoint does not follow the declared selection rule')
    for result in metrics:
        if result['labels'] != config['validation_labels'] or result['selection_metric'] != metric:
            raise ValueError('Validation annotations/metric differ from configuration')
    for step in config['keep_updates']:
        if step > config['updates']:
            continue
        snapshot = torch.load(run / ('step_%07d.pth' % step), map_location='cpu', weights_only=False)
        if snapshot['step'] != step or snapshot['config'] != config:
            raise ValueError('Invalid fixed-update source checkpoint')
    return dict(passed=True, updates=complete['steps'], validations=len(steps),
                selected_step=best['step'], selection_metric=metric,
                training_labels=config['labels'], validation_labels=config['validation_labels'],
                last_epoch_complete=complete['last_epoch_complete'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = verify(args.run)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
