import json
import os
import random
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from transfer import common, protocol_utils, runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        common.configure()

    def run_tiny(self, directory, trace, allocation_updates=0, interrupt_validation=False, config_edit=None):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Dropout(.3), torch.nn.Linear(7, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
        config = dict(seed=17, training_frames=5, optimizer_updates=23, validate_every=4,
                      limit_train=None, code_sha256='fixed code')
        if config_edit:
            config.update(config_edit)
        initialization = dict(tensors=runtime.tensor_state_sha256(model.state_dict()))
        ds = SimpleNamespace(rows=[dict(frame=i) for i in range(7)])
        # SimpleNamespace has no __len__; the real collector requires dataset length.
        class EvaluationDataset:
            rows = ds.rows

            def __len__(self):
                return len(self.rows)

        ds = EvaluationDataset()
        seen_validation = []

        def batches(indices, epoch):
            return iter(indices)

        def loss(index):
            trace.append(index)
            x = torch.tensor([index / 5, 1., -1.])
            value = model(x).sum()
            target = random.random() + np.random.random() + torch.rand(()).item()
            return (value - target).square(), {}

        def evaluation_loader(dataset, indices, workers):
            return (dict(metadata=[dataset.rows[index]]) for index in indices)

        def validate(step, stop):
            def check():
                stop()
                if interrupt_validation and step == 4 and len(seen_validation) == 2:
                    raise runtime.Preempted('test interruption inside validation')

            def predict(batch):
                index = batch['metadata'][0]['frame']
                seen_validation.append((step, index))
                with torch.no_grad():
                    prediction = model(torch.tensor([index / 7, 1., -1.])).item()
                error = abs(prediction - 1.)
                return np.array([error, error * error, error > 1, error > 2, 1]), None, None

            with patch.object(common, 'evaluation_loader', evaluation_loader):
                progress = common._collect(model, ds, 0, directory, 'val_%d' % step, False,
                                           predict, dict(step=step), check)
            # Validation cannot perturb subsequent optimizer randomness.
            random.random()
            np.random.random()
            torch.rand(3)
            return dict(depth=common._depth_summary(progress.sums)), 1.0  # deliberate ties

        result = runtime.train_loop(model, optimizer, directory, config, {}, initialization,
                                    batches, loss, validate, save_every=3, keep_updates=[8],
                                    allocation_updates=allocation_updates)
        return result, torch.load(Path(directory) / 'last.pth', map_location='cpu', weights_only=False), seen_validation

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_exact_budget_training_resume_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as reference, tempfile.TemporaryDirectory() as resumed:
            expected_order, actual_order = [], []
            code, expected, expected_validation = self.run_tiny(reference, expected_order)
            self.assertEqual(code, 0)
            code, interrupted, _ = self.run_tiny(resumed, actual_order, allocation_updates=3)
            self.assertEqual(code, 75)
            self.assertEqual((interrupted['epoch'], interrupted['cursor'], interrupted['step']), (0, 3, 3))
            self.assertFalse((Path(resumed) / 'training_complete.json').exists())
            code, actual, actual_validation = self.run_tiny(resumed, actual_order)
            self.assertEqual(code, 0)
            self.assertEqual(actual_order, expected_order)
            self.assertEqual(actual_validation, expected_validation)
            self.assert_nested_equal(actual, expected)
            self.assertEqual((actual['step'], actual['epoch'], actual['cursor']), (23, 4, 3))
            self.assertEqual((actual['best_step'], actual['validations']), (4, 6))
            best = torch.load(Path(resumed) / 'best.pth', weights_only=False)
            final = torch.load(Path(resumed) / 'final.pth', weights_only=False)
            retained = torch.load(Path(resumed) / 'step_0000008.pth', weights_only=False)
            self.assertEqual((best['step'], retained['step'], final['step']), (4, 8, 23))
            summary = json.loads((Path(resumed) / 'training_complete.json').read_text())
            self.assertEqual(summary['validation_steps'], [4, 8, 12, 16, 20, 23])

    def test_partial_validation_resume_has_no_missing_or_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as reference, tempfile.TemporaryDirectory() as resumed:
            expected_order, actual_order = [], []
            _, expected, expected_validation = self.run_tiny(reference, expected_order)
            code, interrupted, first_validation = self.run_tiny(resumed, actual_order, interrupt_validation=True)
            self.assertEqual(code, 75)
            self.assertEqual((interrupted['step'], interrupted['last_validated_step']), (4, 0))
            self.assertEqual(first_validation, [(4, 0), (4, 1)])
            code, actual, rest_validation = self.run_tiny(resumed, actual_order)
            self.assertEqual(code, 0)
            self.assertEqual(expected_order, actual_order)
            self.assertEqual(expected_validation, first_validation + rest_validation)
            self.assert_nested_equal(expected, actual)
            for step in [4, 8, 12, 16, 20, 23]:
                self.assertEqual(json.loads((Path(reference) / ('validation_step_%07d_metrics.json' % step)).read_text()),
                                 json.loads((Path(resumed) / ('validation_step_%07d_metrics.json' % step)).read_text()))

    def test_changed_configuration_cannot_resume(self):
        with tempfile.TemporaryDirectory() as run:
            self.run_tiny(run, [], allocation_updates=2)
            with self.assertRaisesRegex(ValueError, 'different runtime/configuration'):
                self.run_tiny(run, [], config_edit=dict(code_sha256='edited code'))

    def test_evaluation_cache_rejects_changed_weights_or_inputs(self):
        with tempfile.TemporaryDirectory() as run:
            path = Path(run) / 'partial.pkl'
            state = runtime.EvaluationProgress(path, dict(weights='one', rows='same'), 3, True)
            state.append(0, [1, 1, 0, 0, 1], {'gt': 1}, {'pred': 1})
            state.save()
            loaded = runtime.EvaluationProgress(path, dict(weights='one', rows='same'), 3, True)
            self.assertEqual(loaded.cursor, 1)
            with self.assertRaises(ValueError):
                loaded.append(0, [1, 1, 0, 0, 1], {}, {})
            for changed in [dict(weights='two', rows='same'), dict(weights='one', rows='changed')]:
                with self.assertRaisesRegex(ValueError, 'checkpoint/data/code'):
                    runtime.EvaluationProgress(path, changed, 3, True)

    def test_identity_lock_signal_and_tensor_hash(self):
        with tempfile.TemporaryDirectory() as run:
            path = Path(run) / 'identity.json'
            runtime.ensure_identity(path, {'code': 1})
            with self.assertRaises(ValueError):
                runtime.ensure_identity(path, {'code': 2})
            with runtime.run_lock(run), self.assertRaisesRegex(RuntimeError, 'Another process'):
                with runtime.run_lock(run):
                    pass
        with patch.dict(os.environ, {'SLURM_JOB_END_TIME': ''}), runtime.Budget() as budget:
            os.kill(os.getpid(), signal.SIGUSR1)
            with self.assertRaisesRegex(runtime.Preempted, 'SIGUSR1'):
                budget.check()
        values = dict(scalar=torch.tensor(2, dtype=torch.bfloat16), vector=torch.arange(4))
        self.assertEqual(runtime.tensor_state_sha256(values), runtime.tensor_state_sha256(dict(reversed(list(values.items())))))
        self.assertNotEqual(runtime.tensor_state_sha256({'x': torch.ones(2)}),
                            runtime.tensor_state_sha256({'x': torch.ones(1, 2)}))

    def test_full_budget_has_sixteen_common_selection_slots(self):
        candidates = runtime.validation_steps(62496, 3906)
        self.assertEqual(len(candidates), 16)
        self.assertEqual((candidates[0], candidates[-1]), (3906, 62496))


if __name__ == '__main__':
    unittest.main()
