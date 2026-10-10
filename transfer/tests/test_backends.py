import os
import unittest
from unittest.mock import patch

import torch

from se3d.backends import apply_backend, backend_state, configure_backend, restore_checkpoint_backend


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.original = backend_state()

    def tearDown(self):
        apply_backend(self.original)

    def test_checkpoint_restores_training_settings_after_process_defaults_change(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            saved = configure_backend('historical')
            configure_backend('reproducible', strict=True)
            restored, origin = restore_checkpoint_backend({'backend': saved})
            self.assertEqual(restored, saved)
            self.assertEqual(origin, 'checkpoint.backend')
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', os.environ)

    def test_reproducible_checkpoint_restores_flags_and_workspace(self):
        saved = configure_backend('reproducible')
        configure_backend('historical')
        restored, _ = restore_checkpoint_backend({'backend': saved})
        self.assertEqual(restored, saved)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_legacy_target_environment_is_restored(self):
        saved = configure_backend('reproducible')
        environment = {k: v for k, v in saved.items() if k not in ('profile', 'deterministic_warn_only')}
        configure_backend('historical')
        restored, origin = restore_checkpoint_backend({'environment': environment})
        self.assertEqual(origin, 'checkpoint.environment')
        self.assertEqual({k: restored[k] for k in environment}, environment)

    def test_unrecorded_historical_flags_are_an_explicit_assumption(self):
        configure_backend('reproducible')
        with self.assertWarnsRegex(UserWarning, 'no recorded backend flags'):
            restored, origin = restore_checkpoint_backend({'seed': 20260909})
        self.assertEqual(origin, 'assumed_historical_profile')
        self.assertEqual(restored['profile'], 'historical')
        self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_invalid_or_incomplete_checkpoint_flags_fail_without_partial_application(self):
        saved = configure_backend('historical')
        for value in ('false', None):
            invalid = dict(saved, cudnn_allow_tf32=value)
            with self.assertRaisesRegex(ValueError, 'must be a boolean'):
                restore_checkpoint_backend({'backend': invalid})
            self.assertEqual(backend_state(), {k: v for k, v in saved.items() if k != 'profile'})

    def test_strict_mode_cannot_silently_override_explicit_historical_profile(self):
        saved = configure_backend('historical')
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            configure_backend('historical', strict=True)
        self.assertEqual(backend_state(), {k: v for k, v in saved.items() if k != 'profile'})
        strict = configure_backend(strict=True)
        self.assertEqual(strict['profile'], 'reproducible')
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_workspace_mismatch_after_cuda_start_requires_a_fresh_process(self):
        saved = configure_backend('historical')
        changed = dict(saved, cublas_workspace_config=':16:8' if
                       saved['cublas_workspace_config'] != ':16:8' else ':4096:8')
        with patch('torch.cuda.is_initialized', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'fresh process'):
                apply_backend(changed)
        self.assertEqual(backend_state(), {k: v for k, v in saved.items() if k != 'profile'})

    def test_inference_rejects_changed_driver_overrides_before_applying_flags(self):
        saved = configure_backend('historical')
        with patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '0'}):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint environment'):
                restore_checkpoint_backend({'backend': saved})

    def test_training_rejects_environment_overrides_that_conflict_with_profile(self):
        with patch.dict(os.environ, {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                configure_backend('reproducible')
        with patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '0'}):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                configure_backend('historical')


if __name__ == '__main__':
    unittest.main()
