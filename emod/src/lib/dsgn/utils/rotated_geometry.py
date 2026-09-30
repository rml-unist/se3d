"""DSGN postprocessor adapter: it supplies [x, z, width, length, yaw]."""
from utils.geometry import rotated_overlap


def compute_iou_fast(boxes, query_boxes):
    return rotated_overlap(boxes[:, [0, 1, 3, 2, 4]], query_boxes[:, [0, 1, 3, 2, 4]])
