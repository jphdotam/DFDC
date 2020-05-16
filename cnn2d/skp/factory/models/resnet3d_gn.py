from mmdet.models.builder import build_backbone
import torch, numpy as np

def resnet3d_gn_ws(pretrained=True):
    norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    backbone3d = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=dict(type='ConvWS3'))
    bb3 = build_backbone(backbone3d)
    if pretrained:
        backbone2d = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            style='pytorch',
            norm_cfg=norm_cfg,
            conv_cfg=dict(type='ConvWS'))
        bb2 = build_backbone(backbone2d)
        bb2.init_weights('open-mmlab://jhu/resnet50_gn_ws')
        bb2_weights = bb2.state_dict()
        for w in list(bb2_weights.keys()):
            try:
                bb3.state_dict()[w].data.copy_(bb2_weights[w])
            except RuntimeError:
                tmp_bb3_weights = bb2_weights[w].unsqueeze(-1).repeat(1,1,1,1,bb2.state_dict()[w].shape[-1])
                bb3.state_dict()[w].data.copy_(tmp_bb3_weights / tmp_bb3_weights.shape[-1])
    return bb3

# bb3 = resnet3d_gn_ws()
# X = torch.from_numpy(np.ones((1,3,32,224,224))).float()
# out = bb3(X)
