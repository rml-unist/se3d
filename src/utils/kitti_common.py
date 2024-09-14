import concurrent.futures as futures
import os,math
import pathlib
import re
from collections import OrderedDict
from PIL import Image, ImageDraw
import numpy as np
from skimage import io
import cv2
def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training,
                               relative_path)


def get_label_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training,
                               relative_path)


def get_velodyne_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path)


def get_calib_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path)


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    # image_infos = []
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        image_info['img_path'] = get_image_path(idx, path, training,
                                                relative_path)
        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array(
                [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
                    [3, 4])
            P1 = np.array(
                [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
                    [3, 4])
            P2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
                    [3, 4])
            P3 = np.array(
                [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
                    [3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            image_info['calib/P0'] = P0
            image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            image_info['calib/P3'] = P3
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def filter_kitti_anno(image_anno,
                      used_classes,
                      used_difficulty=None,
                      dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations

def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', None),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path, eval_dist=None):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:

    content = [line.strip().split(' ') for line in lines]

    if eval_dist:
        content = [x for x in content if float(x[13]) >= eval_dist[0] and float(x[13]) < eval_dist[1]]

    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations

def get_label_annos(label_folder, image_ids=None, return_image_ids=False, eval_dist=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        annos.append(get_label_anno(label_filename, eval_dist=eval_dist))
    if return_image_ids:
        return annos, image_ids
    score_thresh=-1
    if score_thresh > 0:
        annos = filter_annos_low_score(dt_annos, score_thresh)    
    return annos

def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union

def lidar_to_top_coords(x, y):
    Xn = int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
    yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

    return xx, yy
def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project_to_image(pts_3d, P):
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)

MATRIX_Mt = np.array(
    [
        [2.34773698e-04, 1.04494074e-02, 9.99945389e-01, 0.00000000e00],
        [-9.99944155e-01, 1.05653536e-02, 1.24365378e-04, 0.00000000e00],
        [-1.05634778e-02, -9.99889574e-01, 1.04513030e-02, 0.00000000e00],
        [5.93721868e-02, -7.51087914e-02, -2.72132796e-01, 1.00000000e00],
    ]
)

MATRIX_Kt = np.array(
    [[721.5377, 0.0, 0.0], [0.0, 721.5377, 0.0], [609.5593, 172.854, 1.0]]
)

def box3d_to_rgb_box00(box3d):

    # box3d = boxes3d[n]
    Ps = np.hstack((box3d, np.ones((8, 1))))
    Qs = np.matmul(Ps, MATRIX_Mt)
    Qs = Qs[:, 0:3]
    qs = np.matmul(Qs, MATRIX_Kt)
    zs = qs[:, 2].reshape(8, 1)
    qs = qs / zs

    return qs[:, 0:2]

def show_image_with_boxes(img, objects, calib=None, show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    img3 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        if obj.type == "Pedestrian":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (255, 255, 0),
            2,
        )
        if obj.type == "Cyclist":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 255),
            2,
        )

        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))
    
    return img1, img2

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image


class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_diffculty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
            bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))
def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr
def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan

def save_image(img_filename, img):
    return cv2.imwrite(img_filename, img)
TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.3
def lidar_to_top(lidar):

    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    if 1:  # new method
        for x in range(Xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x

            for y in range(Yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y

                top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                top[yy, xx, Zn] = quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where(
                        (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                    )
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z

                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    top[yy, xx, zz] = max_height
    return top

def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    divisor = np.max(top_image) - np.min(top_image)
    top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image

def draw_box3d_on_top(
    image,
    boxes3d,
    color=(255, 255, 255),
    thickness=1,
    scores=None,
    text_lables=[],
    is_gt=False,
):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        if is_gt:
            color = (0, 255, 0)
            startx = 5
        else:
            color = heat_map_rgb(0.0, 1.0, scores[n]) if scores is not None else 255
            startx = 85
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    return img

def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    top_view = lidar_to_top(pc_velo)
    top_image = draw_top_image(top_view)
    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    return top_image
class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs["P2"]
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        data = {}
        cam2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
        )
        velo2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
        )
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R0_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        """
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        if constraint_box:
            depth_box_fov_inds = (
                (depth_pc_velo[:, 0] < cbox[0][1])
                & (depth_pc_velo[:, 0] >= cbox[0][0])
                & (depth_pc_velo[:, 1] < cbox[1][1])
                & (depth_pc_velo[:, 1] >= cbox[1][0])
                & (depth_pc_velo[:, 2] < cbox[2][1])
                & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo
