"""Rotated rectangle overlap in KITTI camera BEV coordinates.

Input columns: x, z, length-along-x, width-along-z, clockwise yaw.
This CPU implementation also enables projection/evaluator tests before GPU
allocation. It uses convex polygon clipping, not axis-aligned approximation.
"""
import math
import numba
import numpy as np


@numba.njit(cache=True)
def corners(box):
    points = np.empty((4, 2), dtype=np.float64)
    signs = ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))
    c, s = math.cos(box[4]), math.sin(box[4])
    for i in range(4):
        x, z = signs[i][0] * box[2] / 2, signs[i][1] * box[3] / 2
        points[i, 0] = c * x + s * z + box[0]
        points[i, 1] = -s * x + c * z + box[1]
    return points


@numba.njit(cache=True)
def intersection_area(first, second):
    polygon = np.zeros((16, 2), dtype=np.float64)
    polygon[:4] = first
    count = 4
    for edge in range(4):
        a, b = second[edge], second[(edge + 1) % 4]
        output = np.zeros((16, 2), dtype=np.float64)
        nout = 0
        if count == 0:
            return 0.0
        prev = polygon[count - 1].copy()
        dp = (b[0] - a[0]) * (prev[1] - a[1]) - (b[1] - a[1]) * (prev[0] - a[0])
        for i in range(count):
            cur = polygon[i]
            dc = (b[0] - a[0]) * (cur[1] - a[1]) - (b[1] - a[1]) * (cur[0] - a[0])
            if (dc >= 0) != (dp >= 0):
                fraction = dp / (dp - dc)
                output[nout] = prev + fraction * (cur - prev)
                nout += 1
            if dc >= 0:
                output[nout] = cur
                nout += 1
            prev = cur.copy()
            dp = dc
        polygon = output
        count = nout
    area = 0.0
    for i in range(count):
        j = (i + 1) % count
        area += polygon[i, 0] * polygon[j, 1] - polygon[i, 1] * polygon[j, 0]
    return abs(area) / 2


@numba.njit(cache=True)
def rotated_overlap(boxes, query_boxes, criterion=-1):
    output = np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
    first = np.empty((len(boxes), 4, 2), dtype=np.float64)
    second = np.empty((len(query_boxes), 4, 2), dtype=np.float64)
    for i in range(len(boxes)):
        first[i] = corners(boxes[i])
    for j in range(len(query_boxes)):
        second[j] = corners(query_boxes[j])
    for i in range(len(boxes)):
        for j in range(len(query_boxes)):
            a, b = boxes[i, 2] * boxes[i, 3], query_boxes[j, 2] * query_boxes[j, 3]
            if a <= 0 or b <= 0:
                continue
            # Cheap rejection before polygon clipping.
            if (first[i, :, 0].max() < second[j, :, 0].min() or
                second[j, :, 0].max() < first[i, :, 0].min() or
                first[i, :, 1].max() < second[j, :, 1].min() or
                second[j, :, 1].max() < first[i, :, 1].min()):
                continue
            inter = intersection_area(first[i], second[j])
            denom = a + b - inter if criterion == -1 else a if criterion == 0 else b if criterion == 1 else 1.0
            if denom > 0:
                output[i, j] = inter / denom
    return output
