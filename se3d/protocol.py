"""Shared annotation names and split metadata, without model dependencies."""
import argparse
import json
import warnings
from pathlib import Path

from . import CONDITIONS, SPLITS_FILE

LABELS = ('label', 'label_original')
LABEL_ALIASES = {'corrected': 'label', 'deduplicated': 'label', 'original': 'label_original'}


def label_name(value):
    """Normalize old command-line names; never guess a directory on disk."""
    if value in LABEL_ALIASES:
        canonical = LABEL_ALIASES[value]
        warnings.warn('--labels %s is deprecated; use --labels %s' % (value, canonical),
                      FutureWarning, stacklevel=2)
        return canonical
    if value not in LABELS:
        raise argparse.ArgumentTypeError('annotations must be label or label_original, got %r' % value)
    return value


def add_labels_argument(parser, **kwargs):
    parser.add_argument('--labels', type=label_name, choices=LABELS, default='label',
                        help='annotation directory: label (release) or label_original (historical comparison)',
                        **kwargs)


def load_splits(path=SPLITS_FILE):
    return json.loads(Path(path).read_text())


def condition_of(sequence):
    condition = '_'.join(sequence.split('/')[-1].split('_')[1:3])
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition in sequence name: ' + sequence)
    return condition
