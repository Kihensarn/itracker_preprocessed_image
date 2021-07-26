import os
import json
import h5py
import random

from numpy.core.fromnumeric import resize
import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import List
from PIL import Image


import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

INPUT_SIZE = (224, 224)

trans = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

transform_test = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

transform_train = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    # transforms.Pad(10),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])


def get_data_loader(data_dir, batch_size, image_scale=1, mode='train', num_workers=4, distributed=True, debug=False):

    if mode == 'train':
        is_shuffle = True
        is_load_label = True
        drop_last = True
        transform = transform_train
    elif mode == 'test':
        is_shuffle = False
        is_load_label = False
        drop_last = False
        transform = transform_test
    elif mode == 'eval':
        is_shuffle = False
        is_load_label = True
        drop_last = True
        transform = transform_test
    elif mode == 'test_specific':
        raise NotImplementedError
    else:
        raise ValueError

    data_set = GazeDataset(
        mode=mode,
        dataset_path=data_dir,
        transform=transform,
        is_load_label=is_load_label,
        image_scale=image_scale,
        debug=debug
    )

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        sampler = torch.utils.data.distributed.DistributedSampler(data_set, shuffle=is_shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=drop_last)
        data_loader = DataLoader(
            data_set,
            batch_sampler=batchsampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    return data_loader


def loadMeta(path):
    try:
        meta = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except:
        print('fail to load the {}'.format(path.stem))
        return None
    return meta

class GazeDataset(Dataset):
    def __init__(self,
            mode,
            dataset_path: str,
            transform=None,
            is_load_label=True,
            image_scale=1,
            debug=False
            ):
        self.dataPath = dataset_path
        self.trans = transform
        self.is_load_label = is_load_label
        self.image_scale = image_scale
        self.debug = debug

        if mode == 'test':
            self.dataset = Path(self.dataPath) / 'test'
            self.meta = loadMeta(self.dataset.joinpath('meta_test.mat'))
            print('test')
        elif mode == 'train':
            self.dataset = Path(self.dataPath) / 'train'
            self.meta = loadMeta(self.dataset.joinpath('meta_train.mat'))
            print('train')
        else:
            self.dataset = Path(self.dataPath) / 'val'
            self.meta = loadMeta(self.dataset.joinpath('meta_validate.mat'))
            print('val')

    def loadImage(self, path):
        try:
            image = Image.open(str(path)).convert('RGB')
        except:
            raise RuntimeError('Could load the image')
        return image

    def __getitem__(self, index):
        face_path = self.dataset.joinpath(
            'subject{:0>4d}/face/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))
        left_eye_path = self.dataset.joinpath(
            'subject{:0>4d}/left_eye/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))
        right_eye_path = self.dataset.joinpath(
            'subject{:0>4d}/right_eye/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))

        face = self.loadImage(face_path)
        left_eye = self.loadImage(left_eye_path)
        right_eye = self.loadImage(right_eye_path)

        face = self.trans(face)
        left_eye = self.trans(left_eye)
        right_eye = self.trans(right_eye)

        if self.is_load_label:
            eye_direction = self.meta['face_gaze_direction'][index]
            eye_direction = np.array(eye_direction)
            eye_direction = torch.FloatTensor(eye_direction)
        else:
            eye_direction = []
        return face, left_eye, right_eye, eye_direction

    def __len__(self):
        if self.debug:
            return 1000
        else:
            return self.meta['subject'].shape[0]

