"""Explicit PyTorch backend settings shared by source, transfer and inference.

The historical profile pins the PyTorch 2.5 defaults used by the archived
entrypoints. It reconstructs a convention, not unrecorded historical state.
"""
import os
import warnings

import torch


PROFILES = ('historical', 'reproducible')
BOOL_FIELDS = ('cudnn_deterministic', 'cudnn_benchmark', 'cudnn_allow_tf32',
               'matmul_allow_tf32', 'deterministic_algorithms', 'deterministic_warn_only')
OVERRIDE_ENV = {'nvidia_tf32_override': 'NVIDIA_TF32_OVERRIDE',
                'torch_allow_tf32_cublas_override': 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'}


def add_backend_arguments(parser):
    parser.add_argument('--backend-profile', choices=PROFILES,
                        help='shared numerical settings: historical (default) or reproducible; '
                             '--strict-determinism alone selects reproducible')
    parser.add_argument('--strict-determinism', action='store_true',
                        help='with reproducible settings, error on nondeterministic operations; '
                             'incompatible with --backend-profile historical')


def backend_state():
    return dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                **{key: os.environ.get(name) for key, name in OVERRIDE_ENV.items()})


def apply_backend(settings):
    """Restore recorded settings before creating CUDA tensors or library handles."""
    for key in BOOL_FIELDS:
        if not isinstance(settings.get(key), bool):
            raise ValueError('Backend field must be a boolean: ' + key)
    for key, name in OVERRIDE_ENV.items():
        if key in settings and settings[key] != os.environ.get(name):
            raise RuntimeError('Recorded %s differs; start a new process with the checkpoint environment' % name)
    if not settings['matmul_allow_tf32'] and os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise ValueError('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 conflicts with disabled matrix-multiply TF32')
    workspace = settings.get('cublas_workspace_config')
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be a string or null')
    if settings['deterministic_algorithms'] and workspace not in (':4096:8', ':16:8'):
        raise ValueError('Strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    if workspace != os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        if torch.cuda.is_initialized():
            raise RuntimeError('Restore the checkpoint backend in a fresh process before CUDA initialization')
        if workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    torch.backends.cudnn.deterministic = settings['cudnn_deterministic']
    torch.backends.cudnn.benchmark = settings['cudnn_benchmark']
    torch.backends.cudnn.allow_tf32 = settings['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = settings['matmul_allow_tf32']
    torch.use_deterministic_algorithms(settings['deterministic_algorithms'],
                                       warn_only=settings['deterministic_warn_only'])
    return dict(profile=settings.get('profile', 'recorded'), **backend_state())


def configure_backend(profile=None, strict=False):
    profile = profile or ('reproducible' if strict else 'historical')
    if profile not in PROFILES:
        raise ValueError('Unknown backend profile: ' + str(profile))
    if profile == 'historical' and strict:
        raise ValueError('--strict-determinism cannot be combined with --backend-profile historical')
    reproducible = profile == 'reproducible'
    if not reproducible and os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise ValueError('NVIDIA_TF32_OVERRIDE=0 conflicts with the historical profile; '
                         'unset it or select the reproducible profile')
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if reproducible and workspace is None:
        workspace = ':4096:8'
    return apply_backend(dict(profile=profile, cudnn_deterministic=reproducible,
                              cudnn_benchmark=False, cudnn_allow_tf32=not reproducible,
                              matmul_allow_tf32=False, deterministic_algorithms=strict,
                              deterministic_warn_only=False, cublas_workspace_config=workspace))


def restore_checkpoint_backend(config):
    """Return effective settings and their provenance, including legacy assumptions."""
    if 'backend' in config:
        return apply_backend(config['backend']), 'checkpoint.backend'
    environment = config.get('environment', {})
    if all(key in environment for key in BOOL_FIELDS[:-1]) and 'cublas_workspace_config' in environment:
        settings = {key: environment[key] for key in BOOL_FIELDS[:-1]}
        settings.update(deterministic_warn_only=environment.get('deterministic_warn_only', False),
                        cublas_workspace_config=environment['cublas_workspace_config'])
        settings.update({key: environment[key] for key in OVERRIDE_ENV if key in environment})
        source = 'checkpoint.environment'
        if not all(key in environment for key in OVERRIDE_ENV):
            warnings.warn('Legacy checkpoint did not record TF32 environment overrides; '
                          'using the current process values and recording that assumption.', UserWarning)
            source += '_with_assumed_overrides'
        return apply_backend(settings), source
    if all(key in config for key in ('cudnn_deterministic', 'cudnn_benchmark', 'allow_tf32')):
        settings = dict(cudnn_deterministic=config['cudnn_deterministic'],
                        cudnn_benchmark=config['cudnn_benchmark'],
                        cudnn_allow_tf32=config['allow_tf32'], matmul_allow_tf32=config['allow_tf32'],
                        deterministic_algorithms=config.get('strict_determinism', False),
                        deterministic_warn_only=False,
                        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))
        warnings.warn('Legacy checkpoint records backend flags but not the cuBLAS workspace; '
                      'the evaluation record identifies the assumed workspace.', UserWarning)
        return apply_backend(settings), 'legacy_flags_with_assumed_workspace'
    warnings.warn('Checkpoint has no recorded backend flags; using the explicit historical '
                  'PyTorch 2.5 profile. This is not verified historical runtime provenance.', UserWarning)
    return configure_backend('historical'), 'assumed_historical_profile'
