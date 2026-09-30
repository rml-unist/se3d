"""Apply explicit revision class/anchor configuration before model creation."""
import json
from pathlib import Path
from .od_cfg import cfg

CLASSES = ['Car', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Truck', 'Van', 'Bus']


def configure_seven_classes(statistics_path):
    with Path(statistics_path).open() as stream:
        statistics = json.load(stream)
    anchors = statistics['training_anchor_dimensions']
    cfg.class_names = list(CLASSES)
    cfg.num_classes = len(CLASSES)
    cfg.valid_classes = list(range(1, len(CLASSES) + 1))
    for key, field in [('ANCHORS_HEIGHT', 'height'), ('ANCHORS_WIDTH', 'width'),
                       ('ANCHORS_LENGTH', 'length'), ('ANCHORS_Y', 'center_y')]:
        setattr(cfg.RPN3D, key, [anchors[name][field] for name in CLASSES])
    return cfg


def configure_dsec_classes(statistics_path):
    statistics=json.loads(Path(statistics_path).read_text())
    classes=['Vehicle','Pedestrian','Cyclist']
    anchors=statistics['training_anchor_dimensions']
    cfg.class_names=classes
    cfg.num_classes=len(classes)
    cfg.valid_classes=list(range(1,len(classes)+1))
    for key,field in [('ANCHORS_HEIGHT','height'),('ANCHORS_WIDTH','width'),('ANCHORS_LENGTH','length'),('ANCHORS_Y','center_y')]:
        setattr(cfg.RPN3D,key,[anchors[name][field] for name in classes])
    return cfg
