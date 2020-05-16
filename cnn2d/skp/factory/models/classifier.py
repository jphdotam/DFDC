import logging
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import pretrainedmodels.utils

from torch.autograd import Variable

from .decoder import Decoder
from .backbones import *
from .tcn import TemporalConvNet
from .ig65m.models import *

from pathlib import Path

_PATH = Path(__file__).parent

try:
    from mmaction.models.builder import build_recognizer 
    from mmcv import Config
except:
    print('Unable to import `mmaction`')



class irCSN(nn.Module):

    def __init__(self, 
                 num_classes=1,
                 dropout=0.5,
                 pretrained=True,
                 skip=None):

        super().__init__()

        cfg = Config.fromfile(osp.join(_PATH, 'mmaction/test_configs/CSN/ircsn_kinetics400_se_rgb_r152_seg1_32x2.py'))
        self.net = build_recognizer(cfg.model)
        if pretrained:
            print('Loading pretrained weights ...')
            weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/ircsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-9d6ed879.pth')
            self.net.load_state_dict(weights)

        self.net.cls_head.dropout = nn.Dropout(p=dropout)
        self.net.cls_head.fc_cls = nn.Linear(in_features=self.net.cls_head.fc_cls.in_features, out_features=num_classes)

        self.skip = skip

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class ipCSN(nn.Module):

    def __init__(self, 
                 num_classes=1,
                 dropout=0.5,
                 pretrained=True,
                 skip=None):

        super().__init__()

        cfg = Config.fromfile(osp.join(_PATH, 'mmaction/test_configs/CSN/ipcsn_kinetics400_se_rgb_r152_seg1_32x2.py'))
        self.net = build_recognizer(cfg.model)
        if pretrained:
            print('Loading pretrained weights ...')
            weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/ipcsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-ef39b9e3.pth')
            self.net.load_state_dict(weights)

        self.net.cls_head.dropout = nn.Dropout(p=dropout)
        self.net.cls_head.fc_cls = nn.Linear(in_features=self.net.cls_head.fc_cls.in_features, out_features=num_classes)

        self.skip = skip

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class Inception3D(nn.Module):

    def __init__(self,
                 num_classes,
                 dropout,
                 pretrained,
                 skip=None):

        super().__init__()
        
        cfg = Config.fromfile(osp.join(_PATH, 'mmaction/test_configs/I3D_RGB/i3d_kinetics400_3d_rgb_inception_v1_seg1_f64s1.py'))
        self.net = build_recognizer(cfg.model)
        if pretrained:
            print('Loading pretrained weights ...')
            weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics400_se_rgb_inception_v1_seg1_f64s1_imagenet_deepmind-9b8e02b3.pth')
            self.net.load_state_dict(weights)

        self.net.cls_head.dropout = nn.Dropout(p=dropout)
        self.net.cls_head.fc_cls = nn.Linear(in_features=self.net.cls_head.fc_cls.in_features, out_features=num_classes)
        self.skip = skip

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class R2Plus1D(nn.Module):

    def __init__(self,
                 model,
                 num_classes,
                 dropout,
                 pretrained,
                 skip=None):

        super().__init__()

        self.net = eval(model)(pretrained=pretrained)
        self.net.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=self.net.fc.in_features, out_features=num_classes))
        self.skip = skip

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class SlowFastResNet(nn.Module):

    def __init__(self,
                 num_classes,
                 dropout,
                 pretrained,
                 tau=16):

        super().__init__()

        cfg = Config.fromfile(osp.join(_PATH, 'mmaction/test_configs/SlowFast/slowfast_kinetics400_se_rgb_r50_seg1_4x16.py'))
        cfg.model.backbone.tau = tau

        self.net = build_recognizer(cfg.model)
        if pretrained:
            print('Loading pretrained weights ...')
            weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowfast_kinetics400_se_rgb_r50_4x16_finetune-4623cf03.pth')
            self.net.load_state_dict(weights)

        self.net.cls_head.dropout = nn.Dropout(p=dropout)
        self.net.cls_head.fc_cls = nn.Linear(in_features=self.net.cls_head.fc_cls.in_features, out_features=num_classes)

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class SlowOnlyResNet(nn.Module):

    def __init__(self,
                 num_classes,
                 dropout,
                 pretrained,
                 skip=None):

        super().__init__()

        cfg = Config.fromfile(osp.join(_PATH, 'mmaction/test_configs/SlowOnly/slowonly_kinetics400_se_rgb_r101_seg1_8x8.py'))

        self.net = build_recognizer(cfg.model)
        if pretrained:
            print('Loading pretrained weights ...')
            weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r101_8x8_finetune-b8455f97.pth')
            self.net.load_state_dict(weights)

        self.net.cls_head.dropout = nn.Dropout(p=dropout)
        self.net.cls_head.fc_cls = nn.Linear(in_features=self.net.cls_head.fc_cls.in_features, out_features=num_classes)

        self.skip = skip

    def forward_train(self, x):
        return self.net(x)[:,0]

    def forward_test(self, x):
        return torch.sigmoid(self.net(x)[:,0])

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]

        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)


class SingleHead(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3,
                 use_pool0=True):

        super(SingleHead, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

        if not use_pool0:
            self.backbone.layer0.pool = pretrainedmodels.utils.Identity()

    def forward_image(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)):
            preds.append(self.forward_image(x[:,:,i]))
        preds = torch.stack(preds, dim=1)
        return torch.median(torch.sigmoid(preds), dim=1)[0]

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SingleHeadRecon(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3,
                 recon_loss_weight=1.0):

        super().__init__()

        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)
        self.decoder = Decoder(num_classes=3, in_channels=dim_feats)
        self.recon_loss_weight = recon_loss_weight

    def forward_train(self, input):
        x, image = input['x'], input['image']
        feature_maps = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(self.backbone.layer0(x)))))
        # Reconstruct
        recon = self.decoder(feature_maps, image.size(-1))
        # Vectorize
        x = F.adaptive_avg_pool2d(feature_maps, (1,1))[:,:,0,0]
        x = self.dropout(x)
        recon_loss = F.mse_loss(image, recon)
        return self.fc(x)[:,0], recon_loss

    def forward_test(self, x):
        feature_maps = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(self.backbone.layer0(x)))))
        x = F.adaptive_avg_pool2d(feature_maps, (1,1))[:,:,0,0]
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)):
            preds.append(self.forward_test(x[:,:,i]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_video(x)


class SingleHeadReconSlim(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3):

        super().__init__()

        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward_image(self, x):
        feature_maps = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(self.backbone.layer0(x)))))
        x = F.adaptive_avg_pool2d(feature_maps, (1,1))[:,:,0,0]
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)):
            preds.append(self.forward_image(x[:,:,i]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class SingleHeadX3(SingleHead):

    def forward_image(self, x):
        # x.shape = (B, C, 3, H, W)
        x = self.backbone(x.mean(1))
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)-2):
            preds.append(self.forward_image(x[:,:,i:i+3]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class MaxPoolOverFeatures(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3,
                 skip=None):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

        self.skip = skip

    def forward_train(self, x):
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=1).max(dim=1)[0]
        return self.fc(feats)[:,0]

    def forward(self, x):
        if self.skip:
            x = x[:,:,::self.skip]
        if self.training:
            return self.forward_train(x)
        else:
            return torch.sigmoid(self.forward_train(x))


class HighFive(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Sequential(nn.Conv1d(dim_feats, dim_feats, kernel_size=3, stride=1, padding=0, bias=False),
                                   nn.BatchNorm1d(dim_feats),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(dim_feats, dim_feats, kernel_size=3, stride=1, padding=0, bias=False),
                                   nn.BatchNorm1d(dim_feats),
                                   nn.ReLU())
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward_image(self, x):
        # x.shape = (B, C, 5, H, W)
        assert x.size(2) == 5
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=2)
        feats = self.conv1(feats)
        feats = self.conv2(feats)[:,:,0]
        feats = self.dropout(feats)
        return self.fc(feats)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(0, x.size(2)-5, 3):
            preds.append(self.forward_image(x[:,:,i:i+5]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, padding=1, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Smush(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 freeze_backbone=True,
                 num_input_channels=3):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.smush1 = BasicBlock(3, 3, padding=(1,0,0))
        self.smush2 = BasicBlock(3, 3, padding=(1,0,0))
        self.smush3 = nn.Sequential(nn.Conv3d(3, 3, stride=1, kernel_size=(5,1,1), padding=0), nn.BatchNorm3d(3), nn.ReLU(inplace=True))

        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

        if freeze_backbone:
            self.frozen_backbone = freeze_backbone
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_image(self, x):
        x = self.smush3(self.smush2(self.smush1(x)))
        x = x[:,:,0]
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)):
            preds.append(self.forward_image(x[:,:,i:i+5]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            if self.frozen_backbone:
                if self.backbone.training:
                    print('Switching backbone from train to eval mode ...')
                    self.backbone = self.backbone.eval()

            return self.forward_image(x)
        else:
            return self.forward_video(x)


class TCN8(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.tcn = TemporalConvNet(dim_feats, [dim_feats]*3, 3, 0.2)
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward_image(self, x):
        # x.shape = (B, C, 8, H, W)
        assert x.size(2) == 8
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=2)
        feats = self.tcn(feats)
        feats = self.dropout(feats.mean(-1))
        return self.fc(feats)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(0, x.size(2)-8, 4):
            preds.append(self.forward_image(x[:,:,i:i+8]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class SingleHead3D(SingleHead): pass

class FeatureComb(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 combine='conv',
                 num_stack=21):

        super(FeatureComb, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        if combine == 'conv':
            self.combine = nn.Conv1d(num_stack, 1, kernel_size=1, stride=1, bias=False)
        elif combine == 'max':
            self.combine = 'max'
        elif combine == 'avg':
            self.combine = 'avg'
        self.fc = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        # x.shape = (B, N, C, H, W)
        # B = batch size
        # N = number of images
        # (C, H, W) = image
        # Get features over N
        feats = []
        for _ in range(x.size(1)):
            feats.append(self.dropout(self.backbone(x[:,_])).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        # feats.shape = (B, N, dim_feats)
        if isinstance(self.combine, nn.Module):
            combined = self.combine(feats)[:,0]
        elif self.combine == 'max':
            combined = torch.max(feats, dim=1)[0]
        elif self.combine == 'avg':
            combined = torch.mean(feats, dim=1)
        return self.fc(combined)[:,0]


class DiffModel(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 diff_only=False):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(int(2*dim_feats) if diff_only else int(4*dim_feats), num_classes)
        self.diff_only = diff_only

    def forward_image(self, x):
        assert x.size(2) == 2
        feat1 = self.backbone(x[:,:,0])
        feat2 = self.backbone(x[:,:,1])
        l1 = torch.abs(feat1-feat2)
        l2 = l1 ** 2
        if self.diff_only:
            feats = torch.cat((l1, l2), dim=1)
        else:
            ad = feat1+feat2
            mu = feat1*feat2
            feats = torch.cat((l1, l2, ad, mu), dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(0, x.size(2)-2, 2):
            preds.append(self.forward_image(x[:,:,i:i+2]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class DiffConvModel(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 diff_only=False,
                 num_frames=32):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.combine = nn.Conv1d(num_frames-1, 1, kernel_size=1, stride=1, bias=False)
        self.fc = nn.Linear(int(2*dim_feats) if diff_only else int(4*dim_feats), num_classes)
        self.diff_only = diff_only

    def get_feature(self, x, y):
        l1 = torch.abs(x-y)
        l2 = l1 ** 2
        if self.diff_only:
            feats = torch.cat((l1, l2), dim=1)
        else:
            ad = x+y
            mu = x*y
            feats = torch.cat((l1, l2, ad, mu), dim=1)
        return feats

    def forward(self, x):
        # x.shape = (B, C, N, H, W)
        # B = batch size
        # C = # channels
        # N = # frames
        # H, W = height, width of frame
        # Get features over N
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        bigfeats = []
        for ind in range(feats.size(1)-1):
            bigfeats.append(self.get_feature(feats[:,ind], feats[:,ind+1]).unsqueeze(1))
        del feats
        bigfeats = torch.cat(bigfeats, dim=1)
        bigfeats = self.combine(bigfeats)
        bigfeats = self.dropout(bigfeats)
        return self.fc(bigfeats)[:,0,0]


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


# class _RNN(nn.Module):

#     def __init__(self, 
#                  rnn_module,
#                  embed_size,
#                  hidden_size,
#                  num_classes,
#                  dropout):

#         super().__init__()
        
#         self.embedding_dropout = SpatialDropout(dropout)
#         self.embed_size = embed_size 

#         self.rnn1 = rnn_module(embed_size,    hidden_size, bidirectional=True, batch_first=True)
#         self.rnn2 = rnn_module(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

#         self.linear1 = nn.Linear(hidden_size*2, hidden_size*2)
#         self.linear2 = nn.Linear(hidden_size*2, hidden_size*2)
#         self.linear  = nn.Linear(hidden_size*2, num_classes)

#     def forward(self, x):
#         h_embedding = x

#         h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
#         h_rnn1, _ = self.rnn1(h_embedding)
#         h_rnn2, _ = self.rnn2(h_rnn1)
        
#         h_conc_linear1  = F.relu(self.linear1(h_rnn1))
#         h_conc_linear2  = F.relu(self.linear2(h_rnn2))
        
#         hidden = h_rnn1 + h_rnn2 + h_conc_linear1 + h_conc_linear2 + h_embadd

#         output = self.linear(hidden)
        
#         return output


class _RNN(nn.Module):

    def __init__(self, 
                 rnn_module,
                 embed_size,
                 hidden_size,
                 num_classes,
                 dropout,
                 bidirectional=False,
                 num_layers=1):

        super().__init__()
        
        self.embedding_dropout = SpatialDropout(dropout)
        self.embed_size = embed_size 

        self.rnn = rnn_module(embed_size, hidden_size, 
                              bidirectional=bidirectional, 
                              num_layers=num_layers, 
                              batch_first=True)
        self.linear1 = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_embedding = self.embedding_dropout(x)
        h_rnn, _ = self.rnn(h_embedding)
        out = self.linear1(h_rnn.mean(1))
        out = self.linear2(out)
        return out


class RNNHead(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 load_encoder=None,
                 freeze_encoder=True,
                 rnn='gru',
                 hidden_size=None,
                 bidirectional=False,
                 num_layers=1):

        super(RNNHead, self).__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        if load_encoder:
            print ('Loading encoder weights from {} ...'.format(load_encoder))
            encoder_weights = torch.load(load_encoder, map_location=lambda storage, loc: storage)
            encoder_weights = {k.replace('backbone.', '') : v for k,v in encoder_weights.items()}
            self.backbone.load_state_dict(encoder_weights, strict=False)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self._freeze_encoder()
            self.backbone.eval()

        if rnn.lower() == 'gru':
            rnn_module = nn.GRU
        elif rnn.lower() == 'lstm':
            rnn_module = nn.LSTM
        else:
            raise Exception('`rnn` must be one of [`GRU`, `LSTM`]')

        if hidden_size is None:
            hidden_size = dim_feats

        self.rnn = _RNN(rnn_module, dim_feats, hidden_size, num_classes, dropout,
                        bidirectional=bidirectional, num_layers=num_layers)

    def _freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_image(self, x):
        feats = []
        for _ in range(x.size(2)):
            feats.append(self.backbone(x[:,:,_]).unsqueeze(1))
        feats = torch.cat(feats, dim=1)
        out = self.rnn(feats)
        return out[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(0, x.size(2)-5, 3):
            preds.append(self.forward_image(x[:,:,i:i+5]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        # x.shape = (B, C, N, H, W)
        # B = batch size
        # C = # channels
        # N = # frames
        # H, W = height, width of frame
        # Get features over N
        if self.freeze_encoder:
            if self.backbone.training:
                self.backbone = self.backbone.eval()

        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class SingleConvHead(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_frames,
                 load_encoder=None,
                 freeze_encoder=True):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        if load_encoder:
            print ('Loading encoder weights from {} ...'.format(load_encoder))
            encoder_weights = torch.load(load_encoder, map_location=lambda storage, loc: storage)
            encoder_weights = {k.replace('backbone.', '') : v for k,v in encoder_weights.items()}
            self.backbone.load_state_dict(encoder_weights, strict=False)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self._freeze_encoder()
            self.backbone.eval()

        self.num_frames = num_frames
        self.conv = nn.Conv1d(dim_feats, num_classes, kernel_size=num_frames, stride=1, padding=0, bias=False)

    def _freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_image(self, x):
        feats = []
        for _ in range(x.size(2)):
            feats.append(self.backbone(x[:,:,_]))
        feats = torch.stack(feats, dim=2)
        out = self.conv(feats)
        return out[:,0,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(0, x.size(2)-self.num_frames, self.num_frames // 4):
            preds.append(self.forward_image(x[:,:,i:i+self.num_frames]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        # x.shape = (B, C, N, H, W)
        # B = batch size
        # C = # channels
        # N = # frames
        # H, W = height, width of frame
        # Get features over N
        if self.freeze_encoder:
            if self.backbone.training:
                print('Switching backbone to eval mode ...')
                self.backbone = self.backbone.eval()

        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class FMBackbone(nn.Module):

    def __init__(self,
                 layers):
        super().__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        fms = []
        for layer in self.layers:
            x = layer(x)
            fms.append(x)
        return fms


class DiffBackbone(nn.Module):

    def __init__(self,
                 layers):
        super().__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x, skips):
        fms = []
        # Process the difference image
        x = self.layers[0](x)
        for ind, layer in enumerate(self.layers[1:]):
            x = layer(x+skips[ind])
        return x


class TDN(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 freeze_inets=True):
        super().__init__()

        i_layers, dim_feats = eval(backbone)(pretrained=pretrained)
        d_layers, dim_feats = eval(backbone)(pretrained=pretrained)

        self.i_net = FMBackbone(i_layers)
        self.d_net = DiffBackbone(d_layers)

        if freeze_inets:
            self._freeze_inet()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(int(dim_feats * 4), num_classes)

    def _freeze_inet(self):
        for param in self.i_net.parameters():
            param.requires_grad = False

    def forward_image(self, x):
        # X.shape = (B, C, 2, H, W)
        # B = batch size
        # C = # channels
        # Compute feature maps for first image
        p0, p1, p2, p3, p4 = self.i_net(x[:,:,0])
        # Then the second
        q0, q1, q2, q3, q4 = self.i_net(x[:,:,1])
        # Take the differenes 
        d0, d1, d2, d3, d4 = q0-p0, q1-p1, q2-p2, q3-p3, q4-p4
        # Input it into difference subnetwork
        d_out = self.d_net(x[:,:,1]-x[:,:,0], [d0, d1, d2, d3])

        i_feat1 = p4.mean([-2,-1])
        i_feat2 = q4.mean([-2,-1])
        d_feat  = d_out.mean([-2,-1])

        # Concatenate 
        feat = torch.cat([i_feat1, i_feat2, d_feat, d_feat], dim=1)
        feat = self.dropout(feat)

        return self.fc(feat)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)-1):
            preds.append(self.forward_image(x[:,:,i:i+2]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(preds, dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)


class GrayDiffModel(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 num_input_channels=3):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, num_input_channels=num_input_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

    @staticmethod
    def rescale(x, xmin, xmax):
        return ((x-x.min()) * (xmax-xmin)) / (x.max()-x.min()) + xmin

    def forward_image(self, x): 
        assert x.shape[1] == 1
        assert x.shape[2] == 2
        # x.shape = (B, 1, 2, H, W)
        # B = batch size
        # Compute difference and scale
        d = self.rescale(x[:,:,1]-x[:,:,0], x.min(), x.max())
        # Stack
        x = torch.cat([x[:,0], d], dim=1)
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)[:,0]

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)-1):
            preds.append(self.forward_image(x[:,:,i:i+2]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(preds, dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            return self.forward_video(x)        







