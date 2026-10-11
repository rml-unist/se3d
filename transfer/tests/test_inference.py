import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from se3d.backends import apply_backend, backend_state, configure_backend
from transfer import common, protocol_utils as protocol, runtime, test as inference


class InferenceTests(unittest.TestCase):
    def test_older_training_code_evaluates_but_partial_results_require_same_evaluation_code(self):
        original_backend = backend_state()
        self.addCleanup(apply_backend, original_backend)
        anchors = json.loads(common.ANCHORS.read_text())
        base = protocol.load_protocol(protocol.PROTOCOL)
        rows = base['splits']['test'][:3]

        class Dataset:
            def __init__(self):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

        seen = []
        interrupt = True

        def check_stop():
            if interrupt and len(seen) == 1:
                raise runtime.Preempted('interrupted inference')

        def evaluation_loader(dataset, indices, workers):
            return (dict(metadata=[dataset.rows[index]]) for index in indices)

        def evaluate(model, dataset, workers, output, tag, *, context, check_stop, device):
            def predict(batch):
                seen.append(batch['metadata'][0])
                with torch.no_grad():
                    error = abs(model(torch.ones(1)).item())
                return np.array([error, error * error, error > 1, error > 2, 1]), None, None

            progress = common._collect(model, dataset, workers, output, tag, False,
                                       predict, context, check_stop)
            return dict(frames=progress.cursor, depth=common._depth_summary(progress.sums))

        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            checkpoint = root / 'checkpoint.pth'
            output = root / 'evaluation'
            config = dict(model='se_cff', code_sha256='older training code',
                          anchor_payload_sha256=protocol.json_digest(anchors),
                          protocol_sha256=protocol.sha256(protocol.PROTOCOL),
                          backend=configure_backend('historical'))
            torch.save(dict(schema=runtime.SCHEMA, model=torch.nn.Linear(1, 1).state_dict(),
                            config=config, anchors=anchors, step=4), checkpoint)
            args = SimpleNamespace(checkpoint=checkpoint, model='se_cff', protocol=protocol.PROTOCOL,
                                   anchors=None, limit_test=3, dsec_root=root, labels_root=root,
                                   cache_root=None, output=output, metrics_python=None,
                                   device='cpu', workers=0)
            code = dict(sha256='current evaluation code', files={})
            for obj, name, value in (
                (runtime, 'code_fingerprint', lambda root: dict(code)),
                (protocol, 'audit_input_files', lambda *args: dict(rows_sha256=protocol.json_digest(rows),
                                                                 files_sha256='fixed inputs')),
                (common, 'build_model', lambda name: torch.nn.Linear(1, 1)),
                (common, 'dataset', lambda *args, **kwargs: Dataset()),
                (common, 'evaluation_loader', evaluation_loader),
                (common, 'evaluate_depth', evaluate),
            ):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            budget = SimpleNamespace(check=check_stop)
            with self.assertWarnsRegex(RuntimeWarning, 'checkpoint training code'):
                with self.assertRaises(runtime.Preempted):
                    inference.run(args, budget)
            identity = json.loads((output / 'evaluation_config.json').read_text())
            self.assertEqual(identity['training_code_sha256'], 'older training code')
            self.assertEqual(identity['code_sha256'], 'current evaluation code')
            self.assertEqual(seen, rows[:1])
            partial = output / 'fixed_test_progress.pkl'
            saved = partial.read_bytes()
            interrupt = False
            code['sha256'] = 'changed evaluation code'
            with self.assertWarnsRegex(RuntimeWarning, 'checkpoint training code'):
                with self.assertRaisesRegex(ValueError, 'different provenance'):
                    inference.run(args, budget)
            self.assertEqual(partial.read_bytes(), saved)
            self.assertEqual(seen, rows[:1])
            code['sha256'] = 'current evaluation code'
            with self.assertWarnsRegex(RuntimeWarning, 'checkpoint training code'):
                self.assertEqual(inference.run(args, budget), 0)
            self.assertEqual(seen, rows)
            result = json.loads((output / 'test_metrics.json').read_text())
            self.assertEqual(result['evaluation_config'], identity)
            self.assertEqual(result['frames'], 3)


if __name__ == '__main__':
    unittest.main()
