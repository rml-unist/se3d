import contextlib
import io
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import train


class TinyDataset:
    def __len__(self):
        return 5


class TinyModel(torch.nn.Module):
    def __init__(self, interrupt):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(.2)
        self.weight = torch.nn.Parameter(torch.tensor(.4))
        self.interrupt = interrupt
        self.calls = 0

    def cuda(self):
        return self

    def forward(self, index):
        self.calls += 1
        x = torch.tensor([[1., 2.], [3., 5.], [8., 7.]]) + index
        y = self.bn(x)
        if self.interrupt:
            self.interrupt(self.calls)
        loss = ((self.dropout(y) * self.weight - 1.) ** 2).mean()
        return None, None, loss, loss


class TrainingResumeTests(unittest.TestCase):
    def run_training(self, directory, interrupt=None, extra=()):
        def evaluate(model, *args, **kwargs):
            model.eval()
            return {'3d_mAP40': [1., 1., 1.], 'per_class': {'Car': {'valid_gt': [1, 1, 1]}}}, None

        original_load = torch.load

        def load_cpu(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)

        command = ['train.py', '--model', 'emod', '--data-root', str(directory),
                   '--output', str(directory), '--updates', '3', '--validate-every', '3',
                   '--workers', '0', *extra]
        with contextlib.ExitStack() as stack:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                stack.callback(signal.signal, signum, signal.getsignal(signum))
            for obj, name, value in (
                (sys, 'argv', command),
                (train, 'SE3DFrames', lambda *args, **kwargs: TinyDataset()),
                (train, 'configure_classes', lambda *args: None),
                (train, 'annotation_fingerprint', lambda *args: 'test annotations'),
                (train, 'code_fingerprint', lambda: 'test code'),
                (train, 'build_model', lambda *args: TinyModel(interrupt)),
                (train, 'model_args', lambda index: {'index': index}),
                (train, 'loader', lambda dataset, indices, *args: iter(indices)),
                (train, 'evaluate', evaluate),
                (torch.cuda, 'get_device_name', lambda: 'cpu'),
                (torch.cuda, 'manual_seed_all', lambda *args: None),
                (torch.cuda, 'get_rng_state_all', lambda: []),
                (torch.cuda, 'set_rng_state_all', lambda *args: None),
                (torch, 'load', load_cpu),
            ):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return train.main()

    def test_sigint_finishes_update_and_resumes_identically(self):
        def interrupt_second_forward(count):
            if count == 2:
                os.kill(os.getpid(), signal.SIGINT)

        with tempfile.TemporaryDirectory() as temporary:
            reference, resumed = Path(temporary) / 'reference', Path(temporary) / 'resumed'
            self.assertEqual(self.run_training(reference), 0)
            self.assertEqual(self.run_training(resumed, interrupt_second_forward), 75)
            state = torch.load(resumed / 'last.pth', weights_only=False)
            self.assertEqual((state['step'], state['cursor']), (2, 2))
            self.assertEqual(state['model']['bn.num_batches_tracked'].item(), 2)
            self.assertFalse((resumed / 'training_complete.json').exists())
            self.assertEqual(self.run_training(resumed), 0)
            expected = torch.load(reference / 'last.pth', weights_only=False)
            actual = torch.load(resumed / 'last.pth', weights_only=False)
            self.assertEqual(actual['step'], 3)
            self.assertEqual(actual['model']['bn.num_batches_tracked'].item(), 3)
            torch.testing.assert_close(actual['model'], expected['model'], rtol=0, atol=0)
            torch.testing.assert_close(actual['optimizer'], expected['optimizer'], rtol=0, atol=0)
            self.assertTrue(torch.equal(actual['torch_rng'], expected['torch_rng']))

    def test_arbitrary_keyboard_interrupt_preserves_last_checkpoint(self):
        def interrupt_forward(count):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.assertEqual(self.run_training(run, extra=('--allocation-updates', '1')), 75)
            checkpoint = (run / 'last.pth').read_bytes()
            with self.assertRaises(KeyboardInterrupt):
                self.run_training(run, interrupt_forward)
            self.assertEqual((run / 'last.pth').read_bytes(), checkpoint)


if __name__ == '__main__':
    unittest.main()
