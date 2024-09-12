from __future__ import print_function

from .submodule import *
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from lib.dsgn.utils.bounding_box import compute_corners, quan_to_angle, \
    angle_to_quan, quan_to_rotation, compute_corners_sc
# from lib.dsgn.layers import BuildCostVolume

def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n,1))
    if pts_3d_rect.is_cuda:
        ones = ones.cuda()
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

class ObjectDetectionNet(nn.Module):
    def __init__(self, cfg=None):
        super(ObjectDetectionNet, self).__init__()
        self.cfg = cfg
        self.num_classes = self.cfg.num_classes
        self.hg_rpn_conv3d = getattr(self.cfg, 'hg_rpn_conv3d', False)
        self.hg_rpn_conv = getattr(self.cfg, 'hg_rpn_conv', False)
        self.centerness4class = getattr(self.cfg, 'centerness4class', False)
        self.class4angles = getattr(self.cfg, 'class4angles', True)
        self.box_corner_parameters = getattr(self.cfg, 'box_corner_parameters', True)
        self.PlaneSweepVolume = getattr(self.cfg, 'PlaneSweepVolume', True)
        self.img_feature_attentionbydisp = getattr(self.cfg, 'img_feature_attentionbydisp', False)
        self.voxel_attentionbydisp = getattr(self.cfg, 'voxel_attentionbydisp', False)
        self.loss_disp = getattr(self.cfg, 'loss_disp', True)
        self.fix_centerness_bug = getattr(self.cfg, 'fix_centerness_bug', False)
        self.rpn3d_conv_kernel = getattr(self.cfg, 'rpn3d_conv_kernel', 3)

        self.anchor_angles = torch.as_tensor(np.array(self.cfg.ANCHOR_ANGLES))
        self.num_angles = self.cfg.num_angles
        res_dim = 1

        self.cat_disp = getattr(self.cfg, 'cat_disp', False)
        self.cat_img_feature = getattr(self.cfg, 'cat_img_feature', False)
        self.feature_extraction = feature_extraction(self.cfg)
        self.num_convs = getattr(self.cfg.RPN3D, 'NUM_CONVS', 4)
        self.num_3dconvs = getattr(self.cfg.RPN3D, 'NUM_3DCONVS', 1)
        RPN3D_INPUT_DIM = 0
        if self.PlaneSweepVolume: RPN3D_INPUT_DIM += res_dim  # res_dim = 64
        if self.cat_img_feature: RPN3D_INPUT_DIM += self.cfg.RPN_CONVDIM  # cfg.RPN_CONVDIM = 32
        if self.cfg.RPN3D_ENABLE:
            conv3d_dim = getattr(self.cfg, 'conv3d_dim', 64)

            self.rpn3d_conv = nn.Sequential(convbn_3d(RPN3D_INPUT_DIM, conv3d_dim, self.rpn3d_conv_kernel, 1, 
                1 if self.rpn3d_conv_kernel == 3 else 0, gn=cfg.GN), nn.ReLU(inplace=True))

            if self.num_3dconvs > 1:
                self.rpn_3dconv1 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_3dconvs > 2:
                self.rpn_3dconv2 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_3dconvs > 3:
                self.rpn_3dconv3 = nn.Sequential(convbn_3d(conv3d_dim, conv3d_dim, 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))

            if self.hg_rpn_conv3d:
                self.hg_rpn3d_conv = hourglass(conv3d_dim, gn=cfg.GN)

            self.rpn3d_pool = torch.nn.AvgPool3d((1, 4, 1), stride=(1, 4, 1))
            self.rpn3d_conv2 = nn.Sequential(convbn(conv3d_dim * 5, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True), nn.Dropout(0.2))

            if not self.hg_rpn_conv:
                self.rpn3d_conv3 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
            else:
                self.rpn3d_conv3 = hourglass2d(conv3d_dim * 2, gn=cfg.GN)

            self.rpn3d_cls_convs = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            self.rpn3d_bbox_convs = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True))
            if self.num_convs > 1:
                self.rpn3d_cls_convs2 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True), nn.Dropout(0.2))
                self.rpn3d_bbox_convs2 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True), nn.Dropout(0.2))
            if self.num_convs > 2:
                self.rpn3d_cls_convs3 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
                self.rpn3d_bbox_convs3 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
            if self.num_convs > 3:
                self.rpn3d_cls_convs4 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))
                self.rpn3d_bbox_convs4 = nn.Sequential(convbn(conv3d_dim * 2, conv3d_dim * 2, 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True))

            if self.class4angles:
                self.bbox_cls = nn.Conv2d(conv3d_dim * 2, self.num_angles * self.num_classes, kernel_size=3, padding=1, stride=1)
            else:
                self.bbox_cls = nn.Conv2d(conv3d_dim * 2, self.num_classes, kernel_size=3, padding=1, stride=1)

            centerness_dim = 1
            centerness_dim *= self.num_angles
            if self.centerness4class:
                centerness_dim *= self.num_classes
            self.bbox_centerness = nn.Conv2d(conv3d_dim * 2, centerness_dim, kernel_size=3, padding=1, stride=1)

            self.each_angle_dim = 1

            self.hwl_dim = 3
            self.xyz_dim = 3
            self.bbox_reg = nn.Conv2d(conv3d_dim * 2, self.num_classes * (self.xyz_dim + self.hwl_dim + self.num_angles * self.each_angle_dim), kernel_size=3, padding=1, stride=1)
            self.anchor_size = torch.as_tensor(np.array([cfg.RPN3D.ANCHORS_HEIGHT, cfg.RPN3D.ANCHORS_WIDTH, cfg.RPN3D.ANCHORS_LENGTH])).transpose(1, 0)

        if self.cfg.RPN3D_ENABLE:
            torch.nn.init.normal_(self.bbox_cls.weight, std=0.1)
            torch.nn.init.constant_(self.bbox_cls.bias, 0)
            torch.nn.init.normal_(self.bbox_centerness.weight, std=0.1)
            torch.nn.init.constant_(self.bbox_centerness.bias, 0)
            torch.nn.init.normal_(self.bbox_reg.weight, std=0.02)
            torch.nn.init.constant_(self.bbox_reg.bias, 0)

            prior_prob = cfg.RPN3D.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.bbox_cls.bias, bias_value)

        self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN = cfg.CV_X_MIN, cfg.CV_Y_MIN, cfg.CV_Z_MIN
        self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX = cfg.CV_X_MAX, cfg.CV_Y_MAX, cfg.CV_Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = cfg.X_MIN, cfg.Y_MIN, cfg.Z_MIN
        self.X_MAX, self.Y_MAX, self.Z_MAX = cfg.X_MAX, cfg.Y_MAX, cfg.Z_MAX
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE
        self.GRID_SIZE = cfg.GRID_SIZE

        zs = torch.arange(self.Z_MIN, self.Z_MAX, self.VOXEL_Z_SIZE) + self.VOXEL_Z_SIZE / 2.
        ys = torch.arange(self.Y_MIN, self.Y_MAX, self.VOXEL_Y_SIZE) + self.VOXEL_Y_SIZE / 2.
        xs = torch.arange(self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE) + self.VOXEL_X_SIZE / 2.
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coord_rect = torch.stack([xs, ys, zs], dim=-1)
        self.coord_rect = coord_rect.cuda()
        self.conv1 = nn.Conv2d(128, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(112, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv3d(128, 32, (3, 3, 3), 1)
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(112, 64, (3, 3, 3), 1)
        self.bn4 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()

    #def forward(self, left_img, pred_disp, cost_volume, calibs_Proj):
    def forward(self, left_feature, pred_disp, cost_volume,calibs_Proj):
        N = cost_volume[0].shape[0]
                   
        if self.cfg.RPN3D_ENABLE:
            coord_rect = self.coord_rect.cuda()

            norm_coord_imgs = []
            for i in range(N):
                coord_img = torch.as_tensor(
                    project_rect_to_image(
                        coord_rect.reshape(-1, 3),
                        calibs_Proj[i].float().cuda()
                    ).reshape(*self.coord_rect.shape[:3], 2), 
                dtype=torch.float32, device='cuda')

                coord_img = torch.cat([coord_img, self.coord_rect[..., 2:]], dim=-1)
                norm_coord_img = (coord_img - torch.as_tensor(np.array([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN]), device=coord_rect.device)[None, None, None, :]) / \
                    (torch.as_tensor(np.array([self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX]), device=coord_rect.device) - torch.as_tensor(np.array([self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN]), device=coord_rect.device))[None, None, None, :]
                norm_coord_img = norm_coord_img * 2. - 1.
                norm_coord_imgs.append(norm_coord_img)
            norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
            norm_coord_imgs = norm_coord_imgs.cuda().float()

            valids = (norm_coord_imgs[..., 0] >= -1.) & (norm_coord_imgs[..., 0] <= 1.) & \
                (norm_coord_imgs[..., 1] >= -1.) & (norm_coord_imgs[..., 1] <= 1.) & \
                (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
            valids = valids.float()
            out1 = F.interpolate(cost_volume[0], scale_factor=(3, 3))  # B,64,H,W
            out2 = F.interpolate(cost_volume[1], scale_factor=(6, 6))  # B,32,H,W
            out3 = F.interpolate(cost_volume[2], scale_factor=(12, 12))  # B,16,H,W
            out = torch.cat([out1, out2, out3], dim=1)  # B,112,H,W
            out = self.relu(self.bn2(self.conv2(out))) # B,D=64,H,W

            if self.PlaneSweepVolume:
                CV_feature = out
                CV_feature = CV_feature.unsqueeze(1) # B,C,D=64,H,W

                Voxel = F.grid_sample(CV_feature, norm_coord_imgs)
                Voxel = Voxel * valids[:, None, :, :, :] # ( B, C, 192, 20, 304)
                
            else:
                Voxel = None
            
            if self.cat_img_feature:
                RPN_feature = self.relu(self.bn1(self.conv1(left_feature[-1])))
                valids = (norm_coord_imgs[..., 0] >= -1.) & (norm_coord_imgs[..., 0] <= 1.) & \
                    (norm_coord_imgs[..., 1] >= -1.) & (norm_coord_imgs[..., 1] <= 1.)
                valids = valids.float() 

                Voxel_2D = []
                pred_disps = []
                for i in range(N):
                    RPN_feature_per_im = RPN_feature[i:i+1]
                    for j in range(len(norm_coord_imgs[i])):
                        Voxel_2D_feature = F.grid_sample(RPN_feature_per_im, norm_coord_imgs[i, j:j+1, :, :, :2])
                        Voxel_2D.append(Voxel_2D_feature)
                Voxel_2D = torch.cat(Voxel_2D, dim=0)
                Voxel_2D = Voxel_2D.reshape(N, self.GRID_SIZE[0], -1, self.GRID_SIZE[1], self.GRID_SIZE[2]).transpose(1,2)
                Voxel_2D = Voxel_2D * valids[:, None, :, :, :]

                if Voxel is not None:
                    Voxel = torch.cat([Voxel, Voxel_2D], dim=1)
                else:
                    Voxel = Voxel_2D

            Voxel = self.rpn3d_conv(Voxel) # (B, C=64, D=190, H=20, W=300)
            if self.hg_rpn_conv3d:
                Voxel1, pre_Voxel, post_Voxel = self.hg_rpn3d_conv(Voxel, None, None)
                Voxel = Voxel1 + Voxel

            Voxel = self.rpn3d_pool(Voxel) # (B, C=64, D=190, H=5, W=300)
            Voxel = Voxel.permute(0, 1, 3, 2, 4).reshape(N, -1, self.GRID_SIZE[0], self.GRID_SIZE[2]).contiguous() # (B,C,H,D,W)


        Voxel_BEV = self.rpn3d_conv2(Voxel)
        if not self.hg_rpn_conv:
            Voxel_BEV = self.rpn3d_conv3(Voxel_BEV)
        else:
            Voxel_BEV1, pre_BEV, post_BEV = self.rpn3d_conv3(Voxel_BEV, None, None)
            Voxel_BEV = Voxel_BEV1 # some bug

        Voxel_BEV_cls = self.rpn3d_cls_convs(Voxel_BEV)
        Voxel_BEV_bbox = self.rpn3d_bbox_convs(Voxel_BEV)
        if self.num_convs > 1:
            Voxel_BEV_cls = self.rpn3d_cls_convs2(Voxel_BEV_cls)
            Voxel_BEV_bbox = self.rpn3d_bbox_convs2(Voxel_BEV_bbox)
        if self.num_convs > 2:
            Voxel_BEV_cls = self.rpn3d_cls_convs3(Voxel_BEV_cls)
            Voxel_BEV_bbox = self.rpn3d_bbox_convs3(Voxel_BEV_bbox)
        if self.num_convs > 3:
            Voxel_BEV_cls = self.rpn3d_cls_convs4(Voxel_BEV_cls)
            Voxel_BEV_bbox = self.rpn3d_bbox_convs4(Voxel_BEV_bbox)

        bbox_cls = self.bbox_cls(Voxel_BEV_cls)
        if not self.fix_centerness_bug:
            bbox_reg = self.bbox_reg(Voxel_BEV_cls)
            bbox_centerness = self.bbox_centerness(Voxel_BEV_bbox)
        else:
            bbox_reg = self.bbox_reg(Voxel_BEV_bbox)
            bbox_centerness = self.bbox_centerness(Voxel_BEV_bbox)

        N, C, H, W = bbox_reg.shape

        dxyz, dhwl, angle_reg = torch.split(bbox_reg.reshape(N, self.num_classes, C // self.num_classes, H, W), \
                [self.xyz_dim, self.hwl_dim, self.each_angle_dim * self.num_angles], dim=2)

        angle_reg = angle_reg.permute(0, 3, 4, 2, 1).reshape(-1, self.each_angle_dim * self.num_angles, self.num_classes)

        angle_range = np.pi * 2 / self.num_angles
        q = angle_reg.tanh() * angle_range / 2.
        q = q + self.anchor_angles.cuda()[None, :, None]
        sin_d, cos_d = torch.sin(q), torch.cos(q)

        dxyz = dxyz[:, None, :].repeat(1, self.num_angles, 1, 1, 1, 1)

        dhwl = dhwl.permute(0, 3, 4, 1, 2).reshape(-1, self.num_classes, self.hwl_dim)
        dhwl = dhwl[:, None, :, :].repeat(1, self.num_angles, 1, 1)
        hwl = self.anchor_size.cuda().reshape(1, 1, self.num_classes, 3) * torch.exp(dhwl)
        hwl = hwl.reshape(-1, self.num_angles, self.num_classes, 3)

        if not self.box_corner_parameters:
            hwl = hwl.reshape(N, H, W, self.num_angles, self.num_classes, 3)
            hwl = hwl.permute(0, 3, 4, 5, 1, 2)

            q = q.reshape(N, H, W, self.num_angles, self.num_classes)
            q = q.permute(0, 3, 4, 1, 2)

            bbox_reg = torch.cat( [dxyz, hwl, q[:, :, :, None]], dim=3)
            bbox_reg = bbox_reg.reshape(N, self.num_angles * self.num_classes * 7, H, W)
        else:
            box_corners = compute_corners_sc(
                hwl.reshape(-1, 3), 
                sin_d.reshape(-1), 
                cos_d.reshape(-1)
            ).reshape(N, H, W, self.num_angles, self.num_classes, 3, 8)
            box_corners[:, :, :, :, :, 1, :] += hwl.reshape(N, H, W, self.num_angles, self.num_classes, 3)[:, :, :, :, :, 0:1] / 2.
            box_corners = box_corners.permute(0, 3, 4, 6, 5, 1, 2) 
            bbox_reg = box_corners + dxyz[:, :, :, None]
            bbox_reg = bbox_reg.reshape(N, self.num_angles * self.num_classes * 24, H, W)


        return bbox_cls, bbox_reg, bbox_centerness
