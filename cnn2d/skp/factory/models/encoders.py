import torch
import torch.nn as nn

import pretrainedmodels 

import re

from .resnext_wsl import (
    resnext101_32x8d_wsl  as rx101_32x8, 
    resnext101_32x16d_wsl as rx101_32x16, 
    resnext101_32x32d_wsl as rx101_32x32,
    resnext101_32x48d_wsl as rx101_32x48
)

BACKBONE_CHANNELS = {
    'resnet18': (512, 256, 128, 64, 64),
    'resnet34': (512, 256, 128, 64, 64),
    'resnet50': (2048, 1024, 512, 256, 64),
    'resnet101': (2048, 1024, 512, 256, 64),
    'resnet152': (2048, 1024, 512, 256, 64),
    'se_resnet50': (2048, 1024, 512, 256, 64),
    'se_resnet101': (2048, 1024, 512, 256, 64),
    'se_resnet152': (2048, 1024, 512, 256, 64),
    'se_resnext50_32x4d': (2048, 1024, 512, 256, 64),
    'se_resnext101_32x4d': (2048, 1024, 512, 256, 64),
    'resnext50_32x4d': (2048, 1024, 512, 256, 64),
    'resnext101_32x8d': (2048, 1024, 512, 256, 64),
    'resnext101_32x16d': (2048, 1024, 512, 256, 64),
    'resnext101_32x32d': (2048, 1024, 512, 256, 64),
    'resnext101_32x48d': (2048, 1024, 512, 256, 64),
    'densenet121': (1024, 1024, 512, 256, 64),
    'densenet169': (1664, 1280, 512, 256, 64),
    'densenet201': (1920, 1792, 512, 256, 64),
    'densenet161': (2208, 2112, 768, 384, 96),
}

class Encoder(nn.Module):

    def __init__(self, blocks, channels):
        super().__init__()

        self.blocks, self.channels = nn.ModuleList(blocks), channels

    def forward(self, x):

        x0 = self.blocks[-1](x)
        x1 = self.blocks[-2](x0)
        x2 = self.blocks[-3](x1)
        x3 = self.blocks[-4](x2)
        x4 = self.blocks[-5](x3)

        return [x4, x3, x2, x1, x0]


def generic_resnet_encoder(name, pretrained):
    model = getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)
    if re.search('^se_', name):
        x0 = nn.Sequential(model.layer0.conv1, model.layer0.bn1, model.layer0.relu1)
        x1 = nn.Sequential(model.layer0.pool, model.layer1)
    else:
        x0 = nn.Sequential(model.conv1, model.bn1, model.relu)
        x1 = nn.Sequential(model.maxpool, model.layer1)
    x2 = model.layer2
    x3 = model.layer3 
    x4 = model.layer4
    return Encoder(blocks=[x4, x3, x2, x1, x0], channels=BACKBONE_CHANNELS[name])

def resnet18_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('resnet18', pretrained=pretrained)

def resnet34_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('resnet34', pretrained=pretrained)

def resnet50_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('resnet50', pretrained=pretrained)

def resnet101_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('resnet101', pretrained=pretrained)

def resnet152_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('resnet152', pretrained=pretrained)

def se_resnet50_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('se_resnet50', pretrained=pretrained)

def se_resnet101_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('se_resnet101', pretrained=pretrained)

def se_resnet152_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('se_resnet152', pretrained=pretrained)

def se_resnext50_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('se_resnext50_32x4d', pretrained=pretrained)

def se_resnext101_encoder(pretrained='imagenet'):
    return generic_resnet_encoder('se_resnext101_32x4d', pretrained=pretrained)

def resnext101_wsl_encoder(d, pretrained='instagram'):
    model = eval('rx101_32x{}'.format(d))(pretrained=pretrained)
    x0 = nn.Sequential(model.conv1, model.bn1, model.relu)
    x1 = nn.Sequential(model.maxpool, model.layer1)
    x2 = model.layer2
    x3 = model.layer3 
    x4 = model.layer4
    return Encoder(blocks=[x4, x3, x2, x1, x0], channels=BACKBONE_CHANNELS['resnext101_32x{}d'.format(d)])

def resnext101_32x8d_wsl_encoder(pretrained='instagram'):
    return resnext101_wsl(8, pretrained=pretrained)
        
def resnext101_32x16d_wsl_encoder(pretrained='instagram'):
    return resnext101_wsl(16, pretrained=pretrained)

def resnext101_32x32d_wsl_encoder(pretrained='instagram'):
    return resnext101_wsl(32, pretrained=pretrained)

def resnext101_32x48d_wsl_encoder(pretrained='instagram'):
    return resnext101_wsl(48, pretrained=pretrained)