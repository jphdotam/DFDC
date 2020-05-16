import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .encoders import *

# Wrapper for GroupNorm with 32 channels
class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels):
        super(GroupNorm32, self).__init__(num_channels=num_channels, num_groups=32)

########
# ASPP #
########

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.norm = norm_layer(planes)
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.elu = nn.ELU(True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.norm(x)
        return self.elu(x)

class ASPP(nn.Module):
    def __init__(self, dilations, inplanes, planes, norm_layer, dropout=0.5):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, planes, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, planes, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, planes, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.norm1 = norm_layer(planes)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                             norm_layer(planes),
                                             nn.ELU(True))
        self.conv1 = nn.Conv2d(5 * planes, planes, 1, bias=False)
        self.elu = nn.ELU(True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu(x)

        return self.dropout(x)


# Decoder for DeepLab
class Decoder(nn.Module):
    def __init__(self, num_classes, spp_inplanes, low_level_inplanes, inplanes, dropout, norm_layer):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, inplanes, 1, bias=False)
        self.norm1 = norm_layer(inplanes)
  
        self.elu = nn.ELU(True)
        self.last_conv = nn.Sequential(nn.Conv2d(spp_inplanes + inplanes, spp_inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(spp_inplanes),
                                       nn.ELU(True),
                                       nn.Dropout2d(dropout[0]),
                                       nn.Conv2d(spp_inplanes, spp_inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(spp_inplanes),
                                       nn.ELU(True),
                                       nn.Dropout2d(dropout[1]),
                                       nn.Conv2d(spp_inplanes, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm1(low_level_feat)
        low_level_feat = self.elu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear')
        x = torch.cat((x, low_level_feat), dim=1)
        decoder_output = x
        x = self.last_conv(x)
        return x

class DeepLabV3(nn.Module):
    def __init__(self, 
        encoder,
        num_classes,
        pretrained,
        dropout=dict(
            spp=0.5,
            dc0=0.5,
            dc1=0.1
        ),
        group_norm=False,
        norm_eval=False):
        super().__init__()

        self.encoder = eval(encoder)(pretrained=pretrained)

        # Specifies whether to freeze BatchNorm layers
        self.norm_eval = norm_eval

        norm_layer = GroupNorm32 if group_norm else nn.BatchNorm2d

        aspp_planes = 256
        aspp_dilations = (1, 6, 12, 18) # (1, 12, 24, 36) for output stride 8

        center_input_channels = self.encoder.channels[0]

        self.center = ASPP(aspp_dilations, 
                           inplanes=center_input_channels, 
                           planes=aspp_planes, 
                           dropout=dropout['spp'], 
                           norm_layer=norm_layer)
        self.decoder = Decoder(num_classes, 
                               spp_inplanes=aspp_planes, 
                               low_level_inplanes=self.encoder.channels[-2], 
                               inplanes=64, 
                               dropout=(dropout['dc0'], dropout['dc1']), 
                               norm_layer=norm_layer)

    def forward(self, x):
        out_size = x.size()[2:]
        x4, _, _, x1, _ = self.encoder(x) 
        x = self.center(x4)
        x = self.decoder(x, x1)
        x = F.interpolate(x, size=out_size, mode='bilinear')
        return x

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


class DeepLabCls(nn.Module):

    def __init__(self, 
                 encoder, 
                 pretrained_encoder,
                 num_outputs,
                 backbone,
                 pretrained_backbone,
                 num_classes,
                 fc_dropout,
                 sigmoid=True):

        super().__init__()

        self.deeplab = DeepLabV3(encoder, num_outputs, pretrained_encoder)
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained_backbone)
        self.dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(dim_feats, num_classes)
        # True if using L1/L2 loss
        # False if using BCE
        self.apply_sigmoid = sigmoid

    def forward_train(self, x):
        segmentation = self.deeplab(x)
        # segmentation.shape = (B, 1, H, W)
        feat = self.backbone(torch.cat([torch.sigmoid(segmentation)]*3, dim=1))
        feat = self.dropout(feat)
        return self.fc(feat)[:,0], torch.sigmoid(segmentation) if self.apply_sigmoid else segmentation

    def forward_test(self, x):
        segmentation = self.deeplab(x)
        if self.apply_sigmoid: segmentation = torch.sigmoid(segmentation)
        # segmentation.shape = (B, 1, H, W)
        feat = self.backbone(torch.cat([segmentation]*3, dim=1))
        feat = self.dropout(feat)
        return torch.sigmoid(self.fc(feat)[:,0])

    def forward_video(self, x):
        # x.shape = (B, C, N, H, W)
        preds = []
        for i in range(x.size(2)):
            preds.append(self.forward_train(x[:,:,i])[0])
        preds = torch.stack(preds, dim=1)
        return torch.mean(torch.sigmoid(preds), dim=1)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)










