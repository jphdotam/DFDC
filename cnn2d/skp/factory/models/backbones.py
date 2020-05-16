import pretrainedmodels 
import pretrainedmodels.utils
import numpy as np
import torch
import torch.nn as nn

from .feature_maps import *
from .xception3d import Xception3D
from .mobilenet3d import MobileNet3D
from .efficientnet import EfficientNet
from .resnext_wsl import (
    resnext101_32x8d_wsl  as rx101_32x8, 
    resnext101_32x16d_wsl as rx101_32x16, 
    resnext101_32x32d_wsl as rx101_32x32,
    resnext101_32x48d_wsl as rx101_32x48
)

def densenet121(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet121')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def densenet161(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet161')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def densenet169(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet169')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def generic(name, pretrained, num_input_channels=3):
    model = getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnet18(pretrained='imagenet', num_input_channels=3):
    return generic('resnet18', pretrained=pretrained, num_input_channels=num_input_channels)

def resnet34(pretrained='imagenet', num_input_channels=3):
    return generic('resnet34', pretrained=pretrained, num_input_channels=num_input_channels)

def resnet50(pretrained='imagenet', num_input_channels=3):
    return generic('resnet50', pretrained=pretrained, num_input_channels=num_input_channels)

def resnet101(pretrained='imagenet', num_input_channels=3):
    return generic('resnet101', pretrained=pretrained, num_input_channels=num_input_channels)

def resnet152(pretrained='imagenet', num_input_channels=3):
    return generic('resnet152', pretrained=pretrained, num_input_channels=num_input_channels)

def se_resnet50(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet50', pretrained=pretrained, num_input_channels=num_input_channels)

def se_resnet101(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet101', pretrained=pretrained, num_input_channels=num_input_channels)

def se_resnet152(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet152', pretrained=pretrained, num_input_channels=num_input_channels)

def se_resnext50(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnext50_32x4d', pretrained=pretrained, num_input_channels=num_input_channels)

def se_resnext101(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnext101_32x4d', pretrained=pretrained, num_input_channels=num_input_channels)

def inceptionv3(pretrained='imagenet', num_input_channels=3):
    model, dim_feats = generic('inceptionv3', pretrained=pretrained, num_input_channels=num_input_channels)
    model.aux_logits = False
    return model, dim_feats

def inceptionv4(pretrained='imagenet', num_input_channels=3):
    return generic('inceptionv4', pretrained=pretrained, num_input_channels=num_input_channels)

def inceptionresnetv2(pretrained='imagenet', num_input_channels=3):
    return generic('inceptionresnetv2', pretrained=pretrained, num_input_channels=num_input_channels)

def resnext101_wsl(d, pretrained='instagram', num_input_channels=3):
    model = eval('rx101_32x{}'.format(d))(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats

def resnext101_32x8d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(8, pretrained=pretrained, num_input_channels=num_input_channels)
        
def resnext101_32x16d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(16, pretrained=pretrained, num_input_channels=num_input_channels)

def resnext101_32x32d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(32, pretrained=pretrained, num_input_channels=num_input_channels)

def resnext101_32x48d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(48, pretrained=pretrained, num_input_channels=num_input_channels)

def xception(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'xception')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def xception3d(pretrained='imagenet', num_input_channels=3):
    model = Xception3D(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats

def mobilenet3d(pretrained='imagenet', num_input_channels=3):
    model = MobileNet3D(pretrained=pretrained)
    dim_feats = model.classifier[1].in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet(b, pretrained, num_input_channels=3):
    if pretrained == 'imagenet':
        model = EfficientNet.from_pretrained('efficientnet-{}'.format(b))
    elif pretrained is None:
        model = EfficientNet.from_name('efficientnet-{}'.format(b))
    if num_input_channels != 3:
        first_layer_weights = model.state_dict()['_conv_stem.weight']
        layer_params = {'in_channels' : num_input_channels,
                        'out_channels': model._conv_stem.out_channels,
                        'kernel_size' : model._conv_stem.kernel_size,
                        'stride':  model._conv_stem.stride,
                        'padding': model._conv_stem.padding,
                        'bias': model._conv_stem.bias}
        model._conv_stem = nn.Conv2d(**layer_params)
        first_layer_weights = np.sum(first_layer_weights.cpu().numpy(), axis=1) / num_input_channels
        first_layer_weights = np.repeat(np.expand_dims(first_layer_weights, axis=1), num_input_channels, axis=1)
        model.state_dict()['_conv_stem.weight'].data.copy_(torch.from_numpy(first_layer_weights))
    dim_feats = model._fc.in_features
    model._dropout = pretrainedmodels.utils.Identity()
    model._fc = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b0(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b0', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b1(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b1', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b2(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b2', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b3(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b3', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b4(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b4', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b5(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b5', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b6(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b6', pretrained=pretrained, num_input_channels=num_input_channels)

def efficientnet_b7(pretrained='imagenet', num_input_channels=3):
    return efficientnet('b7', pretrained=pretrained, num_input_channels=num_input_channels)






