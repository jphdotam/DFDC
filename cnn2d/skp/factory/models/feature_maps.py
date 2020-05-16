import pretrainedmodels 
import pretrainedmodels.utils
import numpy as np
import torch
import torch.nn as nn

def resnet34_fm(pretrained='imagenet', num_input_channels=3):
    model  = getattr(pretrainedmodels, 'resnet34')(pretrained=pretrained)
    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu)
    layer1 = nn.Sequential(nn.MaxPool2d(2,2), model.layer1)
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return (layer0, layer1, layer2, layer3, layer4), model.last_linear.in_features