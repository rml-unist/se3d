import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from PIL import Image, ImageDraw

import numpy as np
from scipy import sparse

import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import hflip

from lib.datasets.dsec.labels.base.utils.KITTILoader3D import get_kitti_annos
from lib.dsgn.utils.numpy_utils import *
from lib.dsgn.utils.bounding_box import Box3DList
from configs.od_cfg import cfg
from lib.dsgn.utils.torch_utils import compute_locations_bev
import lib.dsgn.utils.preprocess
import utils.kitti_util as kitti_util 



class LabelsDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps_with_label.txt',
        'labels': 'label_2',
        'left_img': 'image_2',
    }
    _DOMAIN = ['labels', 'left_img']
    NO_VALUE = 0.0

    def __init__(self, root, freeze_mode, training=True, generate_target=True):
        self.root = root

        self.training = training
        self.generate_target = generate_target
        self.cfg = cfg
        self.save_path = cfg.save_path + 'anchor_{}angles'.format(self.cfg.num_angles)

        self.freeze_mode = freeze_mode
        if self.freeze_mode == 'disparity': # object detection task
            self._PATH_DICT['timestamp'] = 'timestamps_with_label.txt'


        self.timestamps = load_timestamp(os.path.join(root.replace('label_2', 'disparity'), self._PATH_DICT['timestamp']))

        self.labels_path_list = {}
        self.timestamp_to_labels_path = {}
        for domain in self._DOMAIN:
            self.timestamp_to_labels_path[domain] = {}
            self.labels_path_list[domain] = get_path_list(os.path.join(os.path.dirname(root), self._PATH_DICT[domain]))
            for timestamp, filepath in zip(self.timestamps, self.labels_path_list[domain]):
                if timestamp == "":
                    continue
                else:
                    self.timestamp_to_labels_path[domain][int(timestamp)] = filepath 
        self.timestamps = np.array([int(timestamp)for timestamp in self.timestamps if timestamp is not ""])

        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_labels_path['labels'][timestamp]))[0])
            for timestamp in self.timestamp_to_labels_path['labels'].keys()
        }

        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        if self.valid_classes is not None:
            self.save_path += '_validclass_{}'.format('_'.join(list(
                map(lambda x:str(x), self.valid_classes)
            )))
        if 2 in self.valid_classes:
            self.less_car = getattr(cfg, 'less_car_pos', False)
            if self.less_car:
                self.save_path += '_lesscar'

        if 1 in self.valid_classes or 3 in self.valid_classes:
            self.less_human = getattr(cfg, 'less_human_pos', False)
            if self.less_human:
                self.save_path += '_lesshuman'

        self.valid_classes = self.cfg.valid_classes

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        return self.load_labels(self.timestamp_to_labels_path['labels'][timestamp])

    @staticmethod
    def collate_fn(batch):
        return batch

    def load_labels(self, root):
        image_index = int(root.split('/')[-1].split('.')[0])
        target = None

        # left image
        left_img_path = root.replace("label", "image").replace(".txt", ".png")
        left_img = Image.open(left_img_path).convert('L')
        left_img_size = left_img.size

        num_bbox = 0
        # box labels 
        if self.training or self.generate_target:
            if self.cfg.RPN3D_ENABLE:
                labels = kitti_util.read_label(root)
                boxes, box3ds, ori_classes = get_kitti_annos(labels, valid_classes=self.valid_classes)

                calib_path = "/workspace/data/calib.txt"                

                calib = kitti_util.Calibration.fromfile(calib_path)
                calib_R = kitti_util.Calibration.fromrightfile(calib_path)

                num_bbox = len(boxes)
                if len(boxes) == 0:
                    print('There is no label for', root)
                    print(labels)
                if len(boxes) > 0:
                    boxes[:, [2,3]] = boxes[:, [0,1]] + boxes[:, [2,3]]
                    boxes = clip_boxes(boxes, left_img.size, remove_empty=False)

                    # sort(far -> near)
                    inds = box3ds[:, 5].argsort()[::-1]
                    box3ds = box3ds[inds]
                    boxes = boxes[inds]
                    ori_classes = ori_classes[inds]

                    # sort by classes
                    inds = ori_classes.argsort(kind='stable')
                    box3ds = box3ds[inds]
                    boxes = boxes[inds]
                    ori_classes = ori_classes[inds]

                    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                    box3ds = torch.as_tensor(box3ds).reshape(-1, 7)

                    #transform to viewpoint from camera 
                    if cfg.learn_viewpoint:
                        h, w, l, x, y, z, alpha = torch.split(box3ds, [1,1,1,1,1,1,1], 1)
                        box3ds[:, 6:] = convert_to_viewpoint_torch(alpha, z, x)

                    target = Box3DList(boxes, left_img.size, mode="xyxy", box3d=box3ds, Proj=calib.P, Proj_R=calib_R.P)
                    classes = torch.as_tensor(ori_classes)
                    target.add_field('labels', classes)

                    if not cfg.flip_this_image:
                        save_file = '{}/{:06d}.npz'.format(self.save_path, image_index)
                        save_label_file = '{}/{:06d}_labels.npz'.format(self.save_path, image_index)
                    else:
                        save_file = '{}/{:06d}_flip.npz'.format(self.save_path, image_index)
                        save_label_file = '{}/{:06d}_flip_labels.npz'.format(self.save_path, image_index)

                    if self.generate_target:
                        locations = compute_locations_bev(self.cfg.Z_MIN, self.cfg.Z_MAX, self.cfg.VOXEL_Z_SIZE, 
                            self.cfg.X_MIN, self.cfg.X_MAX, self.cfg.VOXEL_X_SIZE, torch.device('cpu'))
                        xs, zs = locations[:, 0], locations[:, 1]

                        labels_maps = []
                        dist_bevs = []
                        ious = []

                        for cls in self.valid_classes:
                            ANCHORS_Y = self.cfg.RPN3D.ANCHORS_Y[cls-1]
                            ANCHORS_HEIGHT, ANCHORS_WIDTH, ANCHORS_LENGTH = self.cfg.RPN3D.ANCHORS_HEIGHT[cls-1], self.cfg.RPN3D.ANCHORS_WIDTH[cls-1], self.cfg.RPN3D.ANCHORS_LENGTH[cls-1]

                            ys = torch.zeros_like(xs) + ANCHORS_Y
                            locations3d = torch.stack([xs, ys, zs], dim=1)
                            locations3d = locations3d[:, None].repeat(1, self.cfg.num_angles, 1)

                            hwl = torch.as_tensor(np.array([ANCHORS_HEIGHT, ANCHORS_WIDTH, ANCHORS_LENGTH]))
                            hwl = hwl[None, None].repeat(len(locations3d), self.cfg.num_angles, 1)

                            angles = torch.as_tensor(np.array(self.cfg.ANCHOR_ANGLES))
                            angles = angles[None].repeat(len(locations3d), 1)
                            sin, cos = torch.sin(angles), torch.cos(angles)

                            z_size, y_size, x_size = self.cfg.GRID_SIZE

                            anchors = torch.cat([hwl, locations3d, angles[:, :, None]], dim=2)
                            anchors[:, :, 4] += anchors[:, :, 0] / 2.
                            anchors = anchors.reshape(-1, 7)
                            anchors_boxlist = Box3DList(torch.zeros(len(anchors), 4), left_img.size, mode='xyxy', box3d=anchors, Proj=calib.P, Proj_R=calib_R.P)

                            inds_this_class = target.get_field('labels') == cls
                            target_box3ds = target.box3d[inds_this_class]

                            target_corners = (target.box_corners() + target.box3d[:, None, 3:6])[inds_this_class]
                            anchor_corners = anchors_boxlist.box_corners() + anchors_boxlist.box3d[:, None, 3:6]

                            dist_bev = torch.norm(anchor_corners[:, None, :4, [0,2]] - target_corners[None, :, :4, [0,2]], dim=-1)
                            dist_bev = dist_bev.mean(dim=-1)

                            dist_bev[dist_bev > 5.] = 5.
                            dist_bevs.append(dist_bev)

                            # note that this can one anchor <-> many labels
                            labels_map = torch.zeros((len(dist_bev), len(target_box3ds)), dtype=torch.uint8)
                            for i in range(len(target_box3ds)):
                                if (cls == 2 and self.less_car) or ((cls == 1 or cls == 3) and self.less_human):
                                    box_pixels = (target_box3ds[i, 1] * target_box3ds[i, 2]) / np.fabs(self.cfg.VOXEL_X_SIZE * self.cfg.VOXEL_Z_SIZE) / 4.
                                else:
                                    box_pixels = (target_box3ds[i, 1] * target_box3ds[i, 2]) / np.fabs(self.cfg.VOXEL_X_SIZE * self.cfg.VOXEL_Z_SIZE)
                                box_pixels = abs(int(box_pixels))
                                topk_mindistance, topk_mindistance_ind = torch.topk(dist_bev[:, i], box_pixels, largest=False, sorted=False)

                                labels_map[topk_mindistance_ind[topk_mindistance < 5.], i] = cls
                            labels_maps.append(labels_map)

                        dist_bev = torch.cat(dist_bevs, dim=1)
                        labels_map = torch.cat(labels_maps, dim=1)
                        
                        iou = sparse.csr_matrix(dist_bev)
                        labels_map = sparse.csr_matrix(labels_map)
                    else:
                        if self.training:
                            iou = sparse.load_npz(save_file)
                            labels_map = sparse.load_npz(save_label_file)


        outputs = [
            np.asarray(left_img)[np.newaxis, :, :].astype(np.float32)/255,
            image_index,
            left_img_size,
            iou,
            labels_map,
            calib,
            calib_R,
            target
        ]

        return outputs

def load_timestamp(root):
    with open(root, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = lines[i].replace("\n", "")

    return lines

def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]

def convert_to_viewpoint_torch(alpha, z, x):
    return alpha + torch.atan2(z, x) - np.pi / 2

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

def project_rect_to_image(bbox, P):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = compute_box_3d(dim=bbox[0:3], location=bbox[3:6], rotation_y=bbox[-1])
    pts_2d = np.dot(pts_3d_rect, np.transpose(P[:,0:3])) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    print(pts_2d)
    return pts_2d[:,0:2]


def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def draw_projected_3d_bounding_boxes(img, bounding_boxes):

    img1 = ImageDraw.Draw(img)

    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        # draw lines
        # base
        img1.line([points[0], points[1]], fill='red', width=2)
        img1.line([points[0], points[1]], fill='red', width=2)
        img1.line([points[1], points[2]], fill='red', width=2)
        img1.line([points[2], points[3]], fill='red', width=2)
        img1.line([points[3], points[0]], fill='red', width=2)
        # top
        img1.line([points[4], points[5]], fill='red', width=2)
        img1.line([points[5], points[6]], fill='red', width=2)
        img1.line([points[6], points[7]], fill='red', width=2)
        img1.line([points[7], points[4]], fill='red', width=2)
        # base-top
        img1.line([points[0], points[4]], fill='red', width=2)
        img1.line([points[1], points[5]], fill='red', width=2)
        img1.line([points[2], points[6]], fill='red', width=2)
        img1.line([points[3], points[7]], fill='red', width=2)
    img.show()

    return img


if __name__ == '__main__':
    root = '/workspace/data/'
    root = '/home/song/RML_NAS/EPL/csha/training6/label_2'
    dataset = LabelsDataset(root=root, training=True, generate_target=True)
    outputs = dataset[794576059265]
    calib, calib_R, image_index, target= outputs

    print(root)
    left_img_path = os.path.join(root, 'label_2', image_index + ".png")
    left_img = Image.open(left_img_path).convert('RGB')

    print(dataset.timestamp_to_labels_path['labels'][794576059265])
    print(calib)
    bboxs = target.box3d
    projected_bbox = []
    for bbox in bboxs:
        projected_bbox.append(project_rect_to_image(bbox=bbox, P=calib.P))

    draw_projected_3d_bounding_boxes(left_img, projected_bbox)