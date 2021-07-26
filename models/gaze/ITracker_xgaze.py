import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
# from models.backbone.my_resnet import resnet50
from models.backbone.resnet import resnet50

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiHeadAttBlock(nn.Module):
    def __init__(self, features_dim, num_head, d_k, qkv_bias=True):
        super(MultiHeadAttBlock, self).__init__()

        self.dim = features_dim
        self.num_head = num_head
        self.d_k = d_k
        # assert head_dim * self.num_head == self.dim, "head num setting wrong"

        self.Wq = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)
        self.Wk = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)
        self.Wv = nn.Linear(self.dim, self.num_head * self.d_k, bias=qkv_bias)

        self.proj = nn.Linear(self.num_head * self.d_k, self.dim)

    def forward(self, x):
        # x: b, s, c
        B, S, C = x.shape

        # qkv: b, nhead, s, d_k
        q = self.Wq(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, S, self.num_head, self.d_k).transpose(1, 2)

        # scores: b, nhead, s, s
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # x_attn: b, nhead, s, d_k
        x_attn = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.num_head * self.d_k)
        output = self.proj(x_attn)
        return output


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        # self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(12*12*64, 1024),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )

        # multi-head attention
        self.mha = MultiHeadAttBlock(
            features_dim=1024,
            num_head=4,
            d_k=256
        )
        self.norm1 = nn.LayerNorm(1024)
        self.ffn = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024)
        )
        self.norm2 = nn.LayerNorm(1024)

        # fc output
        self.fc_eye = nn.Sequential(
            nn.Linear(1024 * 2, 128),
            nn.ReLU(True)
        )
        self.fc_face = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def forward(self, faces, eyesLeft, eyesRight):
        B = faces.shape[0]
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft).view(B, 1, -1)
        xEyeR = self.eyeModel(eyesRight).view(B, 1, -1)
        xEyeL = self.eyesFC(xEyeL).view(B, 1, -1)
        xEyeR = self.eyesFC(xEyeR).view(B, 1, -1)

        # Face net
        xFace = self.faceModel(faces).view(B, 1, -1)
        # xGrid = self.gridModel(faceGrids)

        x_seq = torch.cat([xEyeL, xEyeR, xFace], dim=1)
        x_seq = x_seq + self.norm1(self.mha(x_seq))
        x_ffn = x_seq + self.norm2(self.ffn(x_seq))
        xEyeL, xEyeR, xFace = torch.unbind(x_ffn, dim=1)

        # Cat and FC
        xEyes = torch.cat([xEyeL, xEyeR], 1)
        xEyes = self.fc_eye(xEyes)
        xFace = self.fc_face(xFace)

        # Cat all
        x = torch.cat((xEyes, xFace), 1)
        x = self.fc_out(x)
        
        return x


class itracker_transformer(nn.Module):
    def __init__(self):
        super(itracker_transformer, self).__init__()

        self.face_backbone = resnet50(pretrained=True)
        self.left_eye_backbone = resnet50(pretrained=True, replace_stride_with_dilation=[True, True, True])
        self.right_eye_backbone = resnet50(pretrained=True, replace_stride_with_dilation=[True, True, True])

        self.mha = MultiHeadAttBlock(features_dim=2048, num_head=4, d_k=256)
        self.norm1 = nn.LayerNorm(2048)
        self.ffn = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048)
        )
        self.norm2 = nn.LayerNorm(2048)

        self.fc_face = nn.Sequential(
            nn.Linear(2048,128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )
        self.fc_eye = nn.Sequential(
            nn.Linear(2048 * 2, 128),
            nn.ReLU(True)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def forward(self, imFace, Leye, Reye):
        B = imFace.shape[0]

        imFace = self.face_backbone(imFace).view(B, 1, -1)
        Leye = self.left_eye_backbone(Leye).view(B, 1, -1)
        Reye = self.right_eye_backbone(Reye).view(B, 1, -1)

        seq = torch.cat([imFace, Leye, Reye], dim=1)
        seq = seq + self.norm1(self.mha(seq))
        ffn = seq + self.norm2(self.ffn(seq))
        imFace, Leye, Reye = torch.unbind(ffn, dim=1)

        eye = torch.cat([Leye, Reye], dim=1)
        eye = self.fc_eye(eye)
        imFace = self.fc_face(imFace)

        x = torch.cat([eye, imFace], dim=1)
        x = self.fc_out(x)

        return x


