import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# from models.gaze.gazenet import GazeNet
from models.gaze.itracker import ITracker, ITrackerAttention, ITrackerMultiHeadAttention
from models.gaze.ITracker_xgaze import ITrackerModel, itracker_transformer
from utils.losses import get_loss, get_rsn_loss, AngularLoss, get_ohem_loss
from utils.metrics import AverageMeterTensor, angular_error
from utils.io import save_model
from utils.drawing import draw_bbox
from utils.logger import my_visdom, my_logger


def get_optimizer(optim_type, parameters, lr, weight_decay):
    if optim_type == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)
    elif optim_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)
    elif optim_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'momentum':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_type == 'nesterov':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError


def get_scheduler(scheduler_type, optimizer, step_size, gamma, last_epoch):
    if scheduler_type == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    if scheduler_type == 'mslr':
        assert isinstance(step_size, list)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma, last_epoch=last_epoch)
    if scheduler_type == 'cosinelr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, last_epoch=last_epoch)
    if scheduler_type == 'cosineawr':
        assert isinstance(step_size, list)
        T0, Tm = step_size
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm, last_epoch=last_epoch)
    if scheduler_type == 'explr':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    else:
        raise NotImplementedError


def get_model(args):
    if args.model == 'GazeNet':
        return GazeNet(args.backbones, pretrained=args.pretrained, dropout=args.dropout)
    elif args.model == 'ITracker':
        return ITracker(pretrained=args.pretrained)
    elif args.model == 'ITrackerAttention':
        return ITrackerAttention(pretrained=args.pretrained)
    elif args.model == 'ITrackerMultiHeadAttention':
        return ITrackerMultiHeadAttention(pretrained=True)
    elif args.model == 'ITracker_xgaze':
        return ITrackerModel()
    elif args.model == 'itracker_transformer':
        return itracker_transformer()
    else:
        raise NotImplementedError


def train_one_step(model, data):
    output = model(data)
    return output


class Trainer:
    def __init__(self, args):
        self.gpu = args.gpu
        self.fresh_per_iter = args.fresh_per_iter
        self.accumulate_count = args.accumulate_count
        self.save_dir = os.path.join(args.ckpt_dir, args.checkname)

        self.model = get_model(args)

        self.model.cuda(self.gpu)

        self.criterion = get_loss(args.loss).cuda(self.gpu)
        self.optimizer = get_optimizer(args.optim, self.model.parameters(), args.lr, args.weight_decay)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args.step_size, args.gamma, -1)

        self.total_train_step = 0
        self.total_val_step = 0
        self.train_fresh_step = 0
        self.val_fresh_step = 0
        self.is_best = False
        self.best_epoch = {
            'epoch': 0,
            'error': float('inf'),
            'loss': float('inf')
        }

        if args.resume:
            # assert args.last_epoch > -1, 'Please set an available last-epoch, not {}.'.format(args.last_epoch)
            self.optimizer = get_optimizer(args.optim, [{"params":self.model.parameters(),"initial_lr":args.init_lr}], args.init_lr, args.weight_decay)
            self.scheduler = get_scheduler(args.scheduler, self.optimizer, args.step_size, args.gamma, args.last_epoch)
            ckpt_name = 'model_' + str(args.last_epoch) + '.pth.tar'
            ckpt_fn = os.path.join(args.ckpt_dir, args.checkname, ckpt_name)
            assert os.path.exists(ckpt_fn), 'Checkpoint {} is not exists!'.format(ckpt_fn)

            ckpt = torch.load(ckpt_fn, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.best_epoch['error'] = ckpt['current_predict']
            self.best_epoch['epoch'] = ckpt['epoch']

            print('Load checkpoint from', ckpt_fn)
            print(self.optimizer.param_groups[0]['lr'])

        if args.load_ckpt:
            ckpt_fn = os.path.join(args.ckpt_dir, args.checkname, args.ckpt_name)
            assert os.path.exists(ckpt_fn), 'Checkpoint {} is not exists!'.format(ckpt_fn)

            ckpt = torch.load(ckpt_fn, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.optimizer.param_groups[0]['lr'] = args.lr
            # self.scheduler.load_state_dict(ckpt['scheduler'])
            self.best_epoch['error'] = ckpt['current_predict']
            self.best_epoch['epoch'] = ckpt['epoch']

            print('Load checkpoint from', ckpt_fn)
            print(self.optimizer.param_groups[0]['lr'])

        if args.is_distribute:
            DDP = DistributedDataParallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model) #同步BN层 
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=True)
            self.sync_all_devices = dist.barrier

        self.eval_dist = args.eval_dist
        self.ohem = args.use_ohem
        self.ohem_keep_num = 0
        if args.use_ohem:
            self.ohem_keep_num = int(args.ohem_frac * args.batch_size)

    def train_one_epoch(self, epoch, data_loader, logger=None, vis=None):
        if logger is not None:
            logger.train_log_lr(epoch, self.optimizer.param_groups[0]['lr'])
        self.model.train()

        loss_avger = AverageMeterTensor().cuda(self.gpu)
        error_avger = AverageMeterTensor().cuda(self.gpu)

        for i, (imFace, imEyeL, imEyeR, gaze_direction) in enumerate(data_loader):
            imFace = imFace.cuda(self.gpu)
            imEyeL = imEyeL.cuda(self.gpu)
            imEyeR = imEyeR.cuda(self.gpu)
            gaze_direction = gaze_direction.cuda(self.gpu)
            output = self.model(imFace, imEyeL, imEyeR)
            self.total_train_step += 1

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), gaze_direction.cpu().data.numpy()))
            error_avger.update(gaze_error.item(), imFace.size(0))

            loss = self.criterion(output, gaze_direction)
            if self.ohem:
                ohem_loss, ohem_idx = get_ohem_loss(output, gaze_direction, keep_num=self.ohem_keep_num)
                loss += ohem_loss
            loss_avger.update(loss.item(), imFace.size(0))

            # use flood
            # loss = torch.abs(loss - 0.005) + 0.005

            loss = loss / self.accumulate_count
            loss.backward()

            if (i+1) % self.accumulate_count == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if logger is not None:
                if self.total_train_step % self.fresh_per_iter == 0:
                    self.train_fresh_step += 1
                    logger.train_log_n_step(epoch, i, len(data_loader), loss_avger, error_avger)
                    vis.train_vis((self.train_fresh_step, loss_avger.val), (self.train_fresh_step, error_avger.val))
        
        if logger is not None:
            logger.train_log_per_epoch(epoch, loss_avger, error_avger, self.optimizer.param_groups[0]['lr'])

    def save_ckpt(self, epoch, is_best):
        state_dict = {
            'epoch': epoch,
            'current_predict': self.predict,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        save_model(state_dict, is_best, self.save_dir)

    def eval(self, epoch, data_loader, logger=None, vis = None):
        if data_loader is None:
            return

        self.model.eval()

        loss_avger = AverageMeterTensor().cuda(self.gpu)
        error_avger = AverageMeterTensor().cuda(self.gpu)

        for i, (imFace, imEyeL, imEyeR, gaze_direction) in enumerate(data_loader):
            imFace = imFace.cuda(self.gpu)
            imEyeL = imEyeL.cuda(self.gpu)
            imEyeR = imEyeR.cuda(self.gpu)
            gaze_direction = gaze_direction.cuda(self.gpu)
            self.total_val_step += 1
            with torch.no_grad():
                output = self.model(imFace, imEyeL, imEyeR)

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), gaze_direction.cpu().data.numpy()))
            error_avger.update(gaze_error.item(), imFace.size(0))

            loss = self.criterion(output, gaze_direction)
            loss_avger.update(loss.item(), imFace.size(0))

            if logger is not None:
                if self.total_val_step % self.fresh_per_iter == 0:
                    self.val_fresh_step += 1
                    logger.val_log_n_step(epoch, i, len(data_loader), loss_avger, error_avger)
                    vis.validate_vis((self.val_fresh_step, loss_avger.val), (self.val_fresh_step, error_avger.val))

        if self.eval_dist:
            # sum all evaluated loss and error from different devices
            loss_and_error = torch.tensor([loss_avger.sum.clone(), error_avger.sum.clone(), loss_avger.count.clone()],
                                          dtype=torch.float64, device=self.gpu)
            self.sync_all_devices()
            dist.all_reduce(loss_and_error, dist.ReduceOp.SUM)
            loss_sum, error_sum, count_sum = loss_and_error.tolist()

            loss_avg = loss_sum / count_sum
            error_avg = error_sum / count_sum
        else:
            loss_avg = loss_avger.avg.item()
            error_avg = error_avger.avg.item()

        self.predict = error_avg

        self.sync_all_devices()
        if logger is not None:
            logger.val_log_per_epoch(epoch, loss_avger, error_avger)

            if error_avg < self.best_epoch['error']:
                self.is_best = True
                self.best_epoch['epoch'] = epoch
                self.best_epoch['error'] = error_avg
                self.best_epoch['loss'] = loss_avg
            else:
                self.is_best = False
            self.save_ckpt(epoch, self.is_best)
        self.sync_all_devices()
            
    def update_scheduler(self):
        self.scheduler.step()



