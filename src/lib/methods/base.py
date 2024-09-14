import torch
import torch.distributed as dist
import numpy as np

# import sys
# import os
# sys.path.append('/workspace/code/src')
# sys.path.append('/workspace/code')

from tqdm import tqdm
from collections import OrderedDict

from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError
from utils import visualizer
from lib.dsgn.utils.inference3d import make_fcos3d_postprocessor

from configs.od_cfg import cfg




def train(model,
          data_loader,
          optimizer,
          optimizer_detection,
          epoch,
          is_distributed=False,
          world_size=1,
          calib=None,
          calib_R=None,
          image_indexes=None,
          targets=None,
          ious=None,
          labels_map=None,
          freeze_mode=None):
    model.train()

    log_dict = OrderedDict([
        ('Disparity_Loss', AverageMeter(string_format='%6.3lf')),
        ('Detection_Loss', AverageMeter(string_format='%6.3lf')),
        ('Loss', AverageMeter(string_format='%6.3lf')),
        ('EPE', EndPointError(average_by='image', string_format='%6.3lf')),
        ('1PE', NPixelError(n=1, average_by='image', string_format='%6.3lf')),
        ('2PE', NPixelError(n=2, average_by='image', string_format='%6.3lf')),
        ('RMSE', RootMeanSquareError(average_by='image', string_format='%6.3lf')),
    ])

    pbar = tqdm(total=len(data_loader)*1)
    alpha = 0.9
    beta = 0.1
    alpha_ewma = AverageMeter(string_format='%6.3lf')
    beta_ewma = AverageMeter(string_format='%6.3lf')
    accumulation_steps = 4
    for i, batch_data in enumerate(data_loader):
        batch_data = batch_to_cuda(batch_data)

        mask = (batch_data['disparity'] > 0) * (batch_data['disparity'] < 10000000)
        if not mask.any():
            continue

        calib = [data[-3] for data in batch_data['labels']]
        calib_R = [ data[-2] for data in batch_data['labels']]

        calibs_fu = torch.as_tensor(np.array([c.f_u for c in calib]))
        calibs_baseline = torch.abs(torch.as_tensor(np.array([(c.P[0,3]-c_R.P[0,3])/c.P[0,0] for c, c_R in zip(calib, calib_R)])))
        calibs_Proj = torch.as_tensor(np.array([c.P for c in calib]))
        calibs_Proj_R = torch.as_tensor(np.array([c.P for c in calib_R]))

        ious = [data[-5] for data in batch_data['labels']]
        labels_map = [ data[-4] for data in batch_data['labels']]


        targets = [ data[-1] for data in batch_data['labels'] ]
        pred, detection_pred, loss, detection_loss = model(left_event=batch_data['event']['left'],
                           right_event=batch_data['event']['right'],
                           separated_event=batch_data['separated_event'],
                           gt_disparity=batch_data['disparity'],
                           calibs_fu=calibs_fu,
                           calibs_baseline=calibs_baseline,
                           calibs_Proj=calibs_Proj,
                           calibs_Proj_R=calibs_Proj_R,
                           targets=targets,
                           ious=ious,
                           labels_map=labels_map)
        if is_distributed:
            tensor_list = [torch.zeros([1], dtype=torch.int).cuda() for _ in range(world_size)]
            cur_tensor = torch.tensor([loss.size(0)], dtype=torch.int).cuda()
            dist.all_gather(tensor_list, cur_tensor)
            total_point = torch.sum(torch.Tensor(tensor_list))
            loss = loss.sum() / total_point * world_size
        else:
            loss = loss.mean()
        
        total_loss = 0.8*loss+0.2*detection_loss
        total_loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        log_dict['Loss'].update(total_loss.item())
        log_dict['Disparity_Loss'].update(loss.item())
        log_dict['Detection_Loss'].update(detection_loss.item())
        log_dict['EPE'].update(pred, batch_data['disparity'], mask)
        log_dict['1PE'].update(pred, batch_data['disparity'], mask)
        log_dict['2PE'].update(pred, batch_data['disparity'], mask)
        log_dict['RMSE'].update(pred, batch_data['disparity'], mask)
        
        pbar.set_description('Epoch {} Loss: {:.4f}'.format(epoch+1, total_loss.item()))
        pbar.update()
        

    return log_dict


@torch.no_grad()
def test(model,
         data_loader,
         calib=None,
         calib_R=None,
         image_indexes=None,
         targets=None,
         ious=None,
         labels_map=None,
         freeze_mode='disparity'):
    model.eval()
    pred_list = []

    for batch_data in tqdm(data_loader):
        calib = [data[-3] for data in batch_data['labels']]
        calib_R = [ data[-2] for data in batch_data['labels']]

        calibs_fu = torch.as_tensor(np.array([c.f_u for c in calib]))
        calibs_baseline = torch.abs(torch.as_tensor(np.array([(c.P[0,3]-c_R.P[0,3])/c.P[0,0] for c, c_R in zip(calib, calib_R)])))
        calibs_Proj = torch.as_tensor(np.array([c.P for c in calib]))
        calibs_Proj_R = torch.as_tensor(np.array([c.P for c in calib_R]))

        ious = [data[-5] for data in batch_data['labels']]
        labels_map = [ data[-4] for data in batch_data['labels']] 
        
        batch_data = batch_to_cuda(batch_data)

        pred, detection_pred, _, _ = model(left_event=batch_data['event']['left'],
                        right_event=batch_data['event']['right'],
                        separated_event=batch_data['separated_event'],
                        gt_disparity=None,
                        calibs_fu=calibs_fu,
                        calibs_baseline=calibs_baseline,
                        calibs_Proj=calibs_Proj,
                        calibs_Proj_R=calibs_Proj_R,
                        targets=targets,
                        ious=ious,
                        labels_map=labels_map,
                        is_test=True)

        for idx in range(pred.size(0)):
            width = data_loader.dataset.WIDTH
            height = data_loader.dataset.HEIGHT
            cur_pred = pred[idx, :height, :width].cpu()
            box_pred = make_fcos3d_postprocessor(cfg)(
                        detection_pred['bbox_cls'][idx].unsqueeze(0), detection_pred['bbox_reg'][idx].unsqueeze(0), detection_pred['bbox_centerness'][idx].unsqueeze(0),
                    image_sizes=(data_loader.dataset.HEIGHT, data_loader.dataset.WIDTH), calibs_Proj=calibs_Proj
            )
            cur_pred_dict = {
                'file_name': str(batch_data['file_index'][idx].item()).zfill(6) + '.png',
                'pred': visualizer.tensor_to_disparity_image(cur_pred),
                'pred_magma': visualizer.tensor_to_disparity_magma_image(cur_pred, vmax=100),
                'box_pred': box_pred,
                'cur_pred': cur_pred
            }
            pred_list.append(cur_pred_dict)

    return pred_list


def batch_to_cuda(batch_data):
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError

        return batch_data

    for domain in ['event']:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ['left', 'right']:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else:
                batch_data[domain][location] = None
    if 'disparity' in batch_data.keys() and batch_data['disparity'] is not None:
        batch_data['disparity'] = batch_data['disparity'].cuda()
    if 'separated_event' in batch_data.keys() and batch_data['separated_event'] is not None:
        for loc in ['l', 'r']:
            batch_data['separated_event'][loc] = [[x for x in batch_data['separated_event'][loc][0]],[y for y in batch_data['separated_event'][loc][1]]]

    return batch_data