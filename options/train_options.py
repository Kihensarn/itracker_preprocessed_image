import os
import argparse
from pathlib import Path

from utils.io import save_json, load_configs


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Gaze')

        # yaml file
        parser.add_argument('--yaml', type=str, default='../options/configs/train_itracker_mhsa.yaml',
                            help='if not none, all setting will be loaded from yaml file')

        # data settings
        parser.add_argument('--data-dir', type=str, default='/home/data/wjc_data/xgaze_224',
                            help='dataset dir (default: /home/data/wjc_data/xgaze_224)')

        # model params
        # parser.add_argument('--backbone', type=str, default='resnet50',
        #                     help='network model type (default: resnet50)')
        # parser.add_argument('--pretrained', action='store_true',
        #                     default=True, help='load pretrianed mode')
        parser.add_argument('--model', type=str, default='iTracker',
                            help='model name (default: iTracker')

        # training params
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=25, metavar='N',
                            help='number of epochs to train (default: 25)')
        parser.add_argument('--start-epoch', type=int, default=0,
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--resume', default=False,
                            help='Resume training from last saved checkpoint.')
        parser.add_argument('--last-epoch', type=int, default=-1,
                            help='Resume training from last epoch checkpoint.')

        # eval params
        parser.add_argument('--eval', default=True)
        parser.add_argument('--eval-batch-size', type=int, default=64, metavar='N',
                            help='batch size for testing (default: 64)')
        # parser.add_argument('--eval-dist', action='store_true', default=False)

        # loss
        parser.add_argument('--loss', type=str, default='l1',
                            help='loss type (default: l1)')

        # optimizer
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                            help='learning rate (default: 0.0001)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar ='M', help='SGD weight decay (default: 1e-4)')

        # scheduler
        parser.add_argument('--step-size', type=int, default=10,
                            help='step size (default: 10)')
        parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                            help='lr decay factor (default: 0.1)')

        # seed
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

        # checking point
        parser.add_argument('--ckpt_dir', type=str, default='runs',
                            help='Root of checkpoints folder.')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')

        # distributed
        parser.add_argument('--nodes', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')

        # log
        parser.add_argument('--log_dir', type=str, default='/home/wjc/PycharmProjects/MyGazeProject/results/log_dir')
        parser.add_argument('--fresh-per-iter', type=int, default=100)

        # debug
        parser.add_argument('--debug', action='store_true', default=False)

        self.parser = parser

    def parse(self, save_configs=True):
        args = self.parser.parse_args()
        if args.yaml is not None:
            args_dict = load_configs(args.yaml)
            args = argparse.Namespace(**args_dict)

        if args.log_dir is not None:
            log_dir_path = Path(args.log_dir)
            if not log_dir_path.is_dir():
                log_dir_path.mkdir(parents=True)

        config_fn = os.path.join(args.log_dir, 'opt.json')
        save_json(config_fn, args.__dict__)
        return args

if __name__ == '__main__':
    args = Options().parse()
    args.haha = 1
    print(args.haha)