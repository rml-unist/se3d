import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lib.opt as opt

from einops import rearrange

from lib.se_cff.concentration import ConcentrationNet
from lib.se_cff.stereo_matching import StereoMatchingNetwork
from lib.dsgn.object_detection import ObjectDetectionNet
from lib.dsgn.loss3d import RPN3DLoss
from configs.od_cfg import cfg

INF = 10000000

class MyModel(nn.Module):
    def __init__(self, 
                 concentration_net=None,
                 disparity_estimator=None):
        super(MyModel, self).__init__()
        self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)
        self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)

        self.criterion = nn.SmoothL1Loss(reduction='none')

        self.object_detection_net = ObjectDetectionNet(cfg=cfg)

        self.losses = dict()

    def forward(self, 
                left_event,
                right_event,
                # left_img,
                gt_disparity=None,
                calibs_fu=None,
                calibs_baseline=None,
                calibs_Proj=None,
                calibs_Proj_R=None,
                targets=None,
                ious=None,
                labels_map=None,
                is_test=False
        ):

        event_stack = {
            'l': left_event.clone(),
            'r': right_event.clone(),
        }
        # print(event_stack['l'].shape)
    
        height, width = cfg.input_size
        

        concentrated_event_stack = {}
        for loc in ['l', 'r']:
            if event_stack[loc].shape[2] > 1:
                    event_stack[loc] = event_stack[loc][:,:,:1,:,:,:]
            # new_frames, flow_estimate = opt.estimate_optical_flow(separated_event[loc], width=width, height=height)
            # flow_estimate = opt.filter_flow_by_magnitude(flow_estimate, threshold=10)
            # flow_map = opt.create_full_flow_map(*flow_estimate, (height, width))
            event_stack[loc] = rearrange(event_stack[loc], 'b s t h w c -> b (c s t) h w')
            concentrated_event_stack[loc] = self.concentration_net(event_stack[loc])
            # concentrated_event_stack[loc] = opt.warp_features(concentrated_event_stack[loc], flow_map)
            
        # concentrated_event_stack['l'] = opt.warp_features(concentrated_event_stack['l'], flow_map)

        # disparity map regression
        left_feature, cost_volume, pred_disparity_pyramid = self.stereo_matching_net(
            concentrated_event_stack['l'],
            concentrated_event_stack['r'],
            # left_img
        )
        
        if targets is not None:
            for i in range(len(targets)):
                targets[i].bbox = targets[i].bbox.to(left_event.device)
                targets[i].box3d = targets[i].box3d.to(left_event.device)



        #bbox_cls, bbox_reg, bbox_centerness = self.object_detection_net(concentrated_event_stack['l'], pred_disparity_pyramid[-1], cost_volume, calibs_Proj)
        bbox_cls, bbox_reg, bbox_centerness = self.object_detection_net(left_feature, pred_disparity_pyramid[-1], cost_volume, calibs_Proj)
        detection_outputs = {}
        detection_outputs['bbox_cls'] = bbox_cls
        detection_outputs['bbox_reg'] = bbox_reg
        detection_outputs['bbox_centerness'] = bbox_centerness

        # calculate disparity map loss
        loss_disp = None
        # if gt_disparity is not None and is_test:
        if gt_disparity is not None:
            loss_disp = self._cal_disp_loss(pred_disparity_pyramid, gt_disparity)

        # calculate object detection loss
        loss_od = None
        if not is_test:
            loss_od = self._od_loss(detection_outputs, targets, calibs_Proj, calibs_Proj_R, ious, labels_map)
        
            # for idx in range(len(bbox_cls)):
            #     cur_detection_outputs = {}
            #     cur_detection_outputs['bbox_cls'] = bbox_cls[idx]
            #     cur_detection_outputs['bbox_reg'] = bbox_reg[idx]
            #     cur_detection_outputs['bbox_centerness'] = bbox_centerness[idx]
            #     if loss_od == None:


        return pred_disparity_pyramid[-1], detection_outputs, loss_disp, loss_od 

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

    def _cal_disp_loss(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        mask = (gt_disparity > 0) & torch.isfinite(gt_disparity) & (gt_disparity < INF)
        if not mask.any():
            loss = pred_disparity_pyramid[-1].sum() * 0.0
            self.losses.update(disp_loss=loss)
            return loss
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                    gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = self.criterion(pred_disp[mask], gt_disparity[mask])
            loss += weight * cur_loss

        self.losses.update(disp_loss=loss)

        return loss

    def _od_loss(self, detection_outputs, targets, calibs_Proj, calibs_Proj_R, ious, labels_map):
        loss = 0.

        bbox_cls = detection_outputs['bbox_cls']
        bbox_reg = detection_outputs['bbox_reg']
        bbox_centerness = detection_outputs['bbox_centerness']
            
        rpn3d_loss, rpn3d_cls_loss, rpn3d_reg_loss, rpn3d_centerness_loss = RPN3DLoss(cfg)(
            bbox_cls, bbox_reg, bbox_centerness, targets, calibs_Proj, calibs_Proj_R, 
            ious=ious, labels_map=labels_map)

        self.losses.update(rpn3d_cls_loss=rpn3d_cls_loss, 
            rpn3d_reg_loss=rpn3d_reg_loss, 
            rpn3d_centerness_loss=rpn3d_centerness_loss)

        loss += rpn3d_loss


        return loss


if __name__ == 'main':
    pass    
    # name = model_cfg.NAME
    # parameters = model_cfg.PARAMS

    # model = getattr(models, name)(**parameters)

    # data = {'left': torch.tensor((1, w, 1284, 1, 10, 1)),
    #         'right': torch.tensor((1, 480, 1284, 1, 10, 1)),
    #         }
