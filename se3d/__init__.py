"""SE3D benchmark runtime shared by the scripts in tools/.

The baselines keep their original import layout: EMOD code is imported as
``lib``, ``utils`` and ``configs`` from emod/, and DSGN-event as ``dsgn`` from
dsgn_event/. Importing this package puts both on sys.path.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EMOD_ROOT = REPO_ROOT / 'emod'
DSGN_EVENT_ROOT = REPO_ROOT / 'dsgn_event'
SPLITS_FILE = REPO_ROOT / 'splits' / 'se3d_splits.json'
ANCHORS_ROOT = REPO_ROOT / 'anchors'

for _path in (str(DSGN_EVENT_ROOT), str(EMOD_ROOT), str(EMOD_ROOT / 'src')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CLASSES = ('Car', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Truck', 'Van', 'Bus')
CONDITIONS = ('day_sunny', 'day_rain', 'day_heavyrain', 'night_sunny', 'night_rain', 'night_heavyrain')
