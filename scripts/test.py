import os
import argparse

import torch
import torch.nn as nn

from utils.io import save_results, load_configs
from data.xgaze_dataset import get_data_loader
# from models.gaze.gazenet import GazeNet
# from models.gaze.itracker import ITracker, ITrackerAttention, ITrackerMultiHeadAttention
from models.gaze.ITracker_xgaze import ITrackerModel
from utils.modules import get_model
from utils.logger import my_logger
import numpy as np 

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='XGaze test')
        parser.add_argument('--data_dir', type=str, default='/home/data/wjc_data/xgaze_224_prepare_two',
                            help='dataset dir (default: /home/data/wjc_data/xgaze_224_prepare_two)')
        # model params 
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='network model type (default: resnet50)')
        # data loader
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        # cuda, seed
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--yaml_path', type=str, default='/home/wjc/PycharmProjects/MyGazeProject/options/configs/test_itracker.yaml',
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    # init the args
    args = Options().parse()
    opt_dict = load_configs(args.yaml_path)
    for k, v in opt_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visable_device

    ckpt_path = os.path.join(args.ckpt_dir, args.checkname, args.ckpt_name)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init dataloader
    test_loader = get_data_loader(
        args.data_dir, 
        args.batch_size,
        args.image_scale,
        mode='test', 
        num_workers=args.workers, 
        distributed=False,
        debug=args.debug)

    # add logger
    logger = my_logger(args)

    #use DP to load model
    model = get_model(args)
    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # load pretrained checkpoint
    if ckpt_path:
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            if args.no_cuda:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(ckpt_path))

    #tansform to test mode
    model.eval()
    save_index = 0
    test_num = len(test_loader.dataset)
    gaze_predict_all = np.zeros((test_num, 2))
    total_test_step = 0
    
    for i, (imFace, imEyeL, imEyeR, gaze) in enumerate(test_loader):
        # set the data to GPU
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR)

        gaze_predict_all[save_index:save_index+output.shape[0], :] = output.data.cpu().numpy()
        save_index += output.shape[0]
        if total_test_step % args.fresh_per_iter == 0:
            logger.test_log(i, len(test_loader))       

    if save_index == test_num:
        print('the number match')

    save_results(gaze_predict_all)


if __name__ == "__main__":
    main()

