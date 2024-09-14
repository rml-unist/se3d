# -*- coding: utf-8 -*-
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from lib import net
from lib import datasets
from lib import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metric import SummationMeter, Metric
from utils.evalod import get_official_eval_result, kitti_output
from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError
import utils.kitti_common as kitti
import numpy as np
import wandb


class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None

        if self.cfg is not None:
            self._init_from_cfg(cfg)
        
        if args.is_master:
            wandb.init()
            # run_name = "Ours_synth_"+ "trial" + '_'+ str(1)
            run_name = "Ours_synth_"+ args.save_root.split("/")[-1]
            wandb.init(project='se-od', reinit=True)
            wandb.run.name = run_name
            wandb.run.save()
            wandb.config.update(args)

        self.current_epoch = 0

        self.freeze_mode = args.freeze_mode # disparity, detection
        self.pretrained_weight_path = cfg.PRETRAINED_WEIGHT_PATH

        

    def _init_from_cfg(self, cfg):
        assert cfg is not None
        self.cfg = cfg
        
        self.model = _prepare_model(self.cfg.MODEL,
                                    is_distributed=self.args.is_distributed,
                                    local_rank=self.args.local_rank if self.args.is_distributed else None)
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)

        self.optimizer_detection = None

        self.get_train_loader = getattr(datasets, self.cfg.DATASET.TRAIN.NAME).get_dataloader
        self.get_test_loader = getattr(datasets, self.cfg.DATASET.TEST.NAME).get_dataloader

        self.method = getattr(methods, self.cfg.METHOD)

    def train(self):
        if self.args.is_master:
            self._log_before_train()
            wandb.watch(self.model)
        train_loader = self.get_train_loader(args=self.args,
                                             dataset_cfg=self.cfg.DATASET.TRAIN,
                                             dataloader_cfg=self.cfg.DATALOADER.TRAIN,
                                             is_distributed=self.args.is_distributed)

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()

        if self.pretrained_weight_path is not None:
            pretrained_dict = torch.load(self.pretrained_weight_path, map_location=torch.device('cpu'))['model']
            model_state_dict = self.model.module.state_dict()


            model_state_dict.update(pretrained_dict) 
            self.model.module.load_state_dict(model_state_dict)


        if self.freeze_mode=="detection":
            freeze_weight = list(filter(lambda key:"object_detection_net" in key, self.model.module.state_dict().keys()))
            for name, param in self.model.module.named_parameters():
                if name in freeze_weight:
                    param.requires_grad = False

        elif self.freeze_mode=="disparity":
            freeze_weight = list(filter(lambda key: not("object_detection_net" in key), self.model.module.state_dict().keys()))
            for name, param in self.model.module.named_parameters():
                if name in freeze_weight:
                    param.requires_grad = False
        else: 
            pass

        

        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)
            train_log_dict = self.method.train(model=self.model,
                                               data_loader=train_loader,
                                               optimizer=self.optimizer,
                                               optimizer_detection=self.optimizer_detection,
                                               is_distributed=self.args.is_distributed,
                                               world_size=self.args.world_size,
                                               freeze_mode=self.freeze_mode,
                                               epoch = epoch)

            self.scheduler.step()
            self.current_epoch += 1
            if self.args.is_distributed:
                train_log_dict = self._gather_log_all_reduce(train_log_dict)
            if self.args.is_master:
                self._log_after_epoch(epoch + 1, time_checker, train_log_dict, 'train')
                wandb.log({"train/epoch": 100. * (epoch + 1) / (self.cfg.TOTAL_EPOCH + 1),
                            "train/loss_total": train_log_dict['Loss'].value,
                            "train/loss_disp": train_log_dict['Disparity_Loss'].value,
                            "train/loss_det": train_log_dict['Detection_Loss'].value,
                            "train/EPE": train_log_dict['EPE'].value,
                            "train/1PE": train_log_dict['1PE'].value,
                            "train/2PE": train_log_dict['2PE'].value,
                            "train/RMSE": train_log_dict['RMSE'].value})

    def test(self):
        if self.args.is_master:
            wandb.watch(self.model)
            test_loader = self.get_test_loader(args=self.args,
                                               dataset_cfg=self.cfg.DATASET.TEST,
                                               dataloader_cfg=self.cfg.DATALOADER.TEST,)

            self.logger.test()

            total_save_path_eval = []
            for sequence_dataloader in test_loader:
                sequence_name = sequence_dataloader.dataset.sequence_name
                sequence_pred_list = self.method.test(model=self.model,
                                                      data_loader=sequence_dataloader,)

                avg_values = {
                        'epe': [],
                        'n1': [],
                        'n2': [],
                        'rmse': []
                    }

                for cur_pred_dict in sequence_pred_list:
                    file_name = cur_pred_dict.pop('file_name')
                    box_pred = cur_pred_dict.pop('box_pred')
                    cur_pred = cur_pred_dict.pop('cur_pred')

                    dis_image = np.load(os.path.join(self.args.data_root , sequence_name[0], sequence_name[1] +"/disparity/event" , file_name.replace('.png', '.npy'))).astype(np.float32)
                    cur_pred = cur_pred.unsqueeze(0)
                    dis_image = torch.from_numpy(dis_image).unsqueeze(0)
                    mask = dis_image > 0
                    if not mask.any():
                        continue
                    metrics = {
                        'epe': EndPointError(average_by='image', string_format='%6.3lf'),
                        'n1': NPixelError(n=1, average_by='image', string_format='%6.3lf'),
                        'n2': NPixelError(n=2, average_by='image', string_format='%6.3lf'),
                        'rmse': RootMeanSquareError(average_by='image', string_format='%6.3lf')
                    }

                    

                    for key in cur_pred_dict:
                        self.logger.save_visualize(image=cur_pred_dict[key],
                                                   visual_type=key,
                                                   sequence_name=sequence_name,
                                                   image_name=file_name)
                    
                    for metric_name, metric_calculator in metrics.items():
                        metric_calculator.update(cur_pred, dis_image, mask)
                        value = metric_calculator.value
                        avg_values[metric_name].append(value)

                    save_path = os.path.join(self.args.save_root, 'preds')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_dir = os.path.join(save_path, sequence_name[0], sequence_name[1],'txt')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_vis = os.path.join(save_path, sequence_name[0], sequence_name[1],'bbox3d')
                    if not os.path.exists(save_vis):
                        os.makedirs(save_vis)
                    save_directory = os.path.join(save_path, sequence_name[0], sequence_name[1],'image')
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)
                    save_directory_vel = os.path.join(save_path, sequence_name[0], sequence_name[1],'vel')
                    if not os.path.exists(save_directory_vel):
                        os.makedirs(save_directory_vel)
                    kitti_output(box_pred[0], int(file_name.rstrip('.png')), save_dir)
                    
                    # Read Calibration file
                    calib = kitti.Calibration(os.path.join(self.args.data_root, 'calib.txt'))

                    # Read detection file and make a bbox
                    img_bbox2d, img_bbox3d = kitti.show_image_with_boxes(kitti.load_image(os.path.join(self.args.data_root,sequence_name[0], sequence_name[1], "image_2" , file_name)), 
                                                                        kitti.read_label( os.path.join(save_dir, file_name.replace('.png', '.txt')) ), calib)
                    # Visualize the 3D bbox on the RGB images
                    kitti.save_image(os.path.join(save_vis, file_name), img_bbox3d)

                    # Same above to visualize 3D bbox on GT
                    if not os.path.exists(save_vis.replace('bbox3d', 'bbox3d_gt')):
                        os.makedirs(save_vis.replace('bbox3d', 'bbox3d_gt'))
                    img_bbox2d, img_bbox3d = kitti.show_image_with_boxes(kitti.load_image(os.path.join(self.args.data_root, sequence_name[0], sequence_name[1], "image_2" , file_name)), 
                                                                        kitti.read_label( os.path.join(self.args.data_root, sequence_name[0], sequence_name[1], 'label_2', file_name.replace('.png', '.txt')) ), calib)
                    kitti.save_image(os.path.join(save_vis.replace('bbox3d', 'bbox3d_gt'), file_name), img_bbox3d)
                    
    
                    pc_velo = kitti.load_velo_scan(os.path.join(self.args.data_root,sequence_name[0], sequence_name[1] +"/velodyne" , file_name.replace('.png','.bin')))
                    img_bev = kitti.show_lidar_topview_with_boxes(pc_velo,kitti.read_label(  os.path.join(save_dir, file_name.replace('.png', '.txt'))), calib)
                    save_file_path_vel = os.path.join(save_directory_vel, file_name)
                    kitti.save_image(save_file_path_vel, img_bev) 


                dt_annos, image_ids = kitti.get_label_annos(save_dir,
                                                            return_image_ids=True,
                                                            eval_dist=self.args.is_distributed,
                                                            )
                score_thresh=0
                if score_thresh > 0:
                    dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)

                gt_annos = kitti.get_label_annos(os.path.join(self.args.data_root, sequence_name[0], sequence_name[1], 'label_2'),
                                                    image_ids=image_ids,
                                                    eval_dist=self.args.is_distributed
                                                    )
                                                    
                result = get_official_eval_result(gt_annos,
                                                dt_annos,
                                                [0]
                                                )
                save_path_eval = os.path.join(self.args.save_root,'eval_detection',sequence_name[0], sequence_name[1])
                
                total_save_path_eval.append(save_path_eval)
                if not os.path.exists(save_path_eval):
                    os.makedirs(save_path_eval)

                with open(save_path_eval+'/'+'evaluation_detection.txt', 'w') as f:
                    f.write(str(get_official_eval_result(gt_annos, dt_annos, [0])))
            for metric_name, values in avg_values.items():
                print(f"{metric_name.upper()}: {np.mean(values):.3f}", end=' ')
            
            def parse_evaluation_data(file_content):
                data = {}
                lines = file_content.strip().split('\n')
                for line in lines:
                    if 'Car AP' in line:
                        ap_key = line.strip()
                        data[ap_key] = {}
                    else:
                        key, values_str = line.split(':')
                        values = list(map(float, re.findall(r'\d+\.\d+', values_str)))
                        data[ap_key][key.strip()] = values
                return data

            def calculate_averages(data_list):
                averages = {}
                for data in data_list:
                    for ap_key, ap_values in data.items():
                        if ap_key not in averages:
                            averages[ap_key] = {}
                        for key, values in ap_values.items():
                            if key not in averages[ap_key]:
                                averages[ap_key][key] = [0] * len(values)
                            for i, value in enumerate(values):
                                averages[ap_key][key][i] += value
                            
                sequence_count = len(data_list)
                for ap_key, ap_values in averages.items():
                    for key, values in ap_values.items():
                        for i in range(len(values)):
                            values[i] /= sequence_count
                return averages

            sequence_folders =total_save_path_eval  # Add the path to the evaluation_detection.txt files
            data_list = []

            for folder in sequence_folders:
                file_path = os.path.join(folder, 'evaluation_detection.txt')
                with open(file_path, 'r') as f:
                    file_content = f.read()
                data = parse_evaluation_data(file_content)
                data_list.append(data)

            averages = calculate_averages(data_list)

            for ap_key, ap_values in averages.items():
                print(ap_key)
                for key, values in ap_values.items():
                    print(f"{key}: {', '.join([f'{value:.2f}' for value in values])}")
                print()

    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def load(self, name):
        if self.args.is_master:
            checkpoint = self.logger.load_checkpoint(name)
            self.model.module.load_state_dict(checkpoint['model'])
            self.current_epoch = checkpoint['epoch']
        dist.barrier()  # Ensure that the master process has loaded the checkpoint before continuing

        # Broadcast the model and current_epoch from the master process to all other processes
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        current_epoch_tensor = torch.tensor(self.current_epoch).to(self.model.device)
        dist.broadcast(current_epoch_tensor, src=0)
        self.current_epoch = current_epoch_tensor.item()

    def _make_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'args': self.args,
            'cfg': self.cfg,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        return checkpoint

    def _gather_log(self, log_dict):
        if log_dict is None:
            return None

        for key in log_dict.keys():
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                log_dict[key].all_gather(self.args.world_size)

        return log_dict

    def _gather_log_all_reduce(self, log_dict):
        for key in log_dict.keys():
            if torch.is_tensor(log_dict[key].value):
                log_dict[key].value = self._all_reduce_tensor(log_dict[key].value)
        return log_dict

    def _all_reduce_tensor(self, tensor):
        if not self.args.is_distributed:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.args.world_size
        return tensor

    def _log_before_train(self):
        self.logger.train()
        self.logger.save_args(self.args)
        self.logger.save_cfg(self.cfg)
        # self.logger.log_model(self.model)
        # self.logger.log_optimizer(self.optimizer)
        self.logger.save_src(os.path.dirname(os.path.abspath(__file__)))

    def _log_after_epoch(self, epoch, time_checker, log_dict, part):
        # Calculate Time
        time_checker.update(epoch)

        # Log Time
        self.logger.write('Epoch: %d | time per epoch: %s | eta: %s' %
                          (epoch, time_checker.time_per_epoch, time_checker.eta))

        # Log Learning Process
        log = '%5s' % part
        for key in log_dict.keys():
            log += ' | %s: %s' % (key, str(log_dict[key]))
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key].value, epoch)
            else:
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key], epoch)
        self.logger.write(log=log)

        # Make Checkpoint
        checkpoint = self._make_checkpoint()

        # Save Checkpoint
        self.logger.save_checkpoint(checkpoint, 'final.pth')
        if epoch % self.args.save_term == 0:
            self.logger.save_checkpoint(checkpoint, '%d.pth' % epoch)


def _prepare_model(model_cfg, is_distributed=False, local_rank=None):
    name = model_cfg.NAME
    parameters = model_cfg.PARAMS

    model = getattr(net, name)(**parameters)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model).cuda()

    return model


def _prepare_optimizer(optimizer_cfg, model):
    name = optimizer_cfg.NAME
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr

    params_group = model.module.get_params_group(learning_rate)

    optimizer = getattr(optim, name)(params_group, **parameters)

    return optimizer


def _prepare_scheduler(scheduler_cfg, optimizer):
    name = scheduler_cfg.NAME
    parameters = scheduler_cfg.PARAMS

    if name == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
    else:
        scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

    return scheduler
