import torch
import torch.nn as nn
import os.path as osp
import re

from pathlib import Path

from .ops.models import TSN


_PATH = Path(__file__).parent

class TSM(nn.Module):

    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained 
        self.backbone = backbone

        self._load_model()

    def _load_model(self):
        self.model = TSN(400, 8, 'RGB',
            base_model=self.backbone,
            consensus_type='avg',
            img_feature_dim=256,
            pretrain=None,
            is_shift=True, 
            shift_div=8, 
            shift_place='blockres',
            non_local=True)
        if self.pretrained:
            print('Loading pretrained TSM model ...')
            _state_dict = torch.load(osp.join(_PATH, 'pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth'), map_location=lambda storage, loc: storage)['state_dict']
            self.model.load_state_dict({re.sub(r'^module\.', '', k) : v for k,v in _state_dict.items()})
        self.model.new_fc = nn.Linear(in_features=self.model.new_fc.in_features, out_features=self.num_classes)

    def forward_chunk(self, x):
        return self.model(x)[:,0]

    def forward_video(self, x):
        preds = []
        for i in range(0, x.size(1)-8, 4):
            preds.append(self.forward_chunk(x[:,i:i+8]))
        preds = torch.stack(preds, dim=1)
        return torch.mean(preds, dim=1)

    def forward(self, x):
        # Assumes input will be (n_batch, n_channels, n_frames, h, w)
        # So switch it to (n_batch, n_frames, n_channels, h, w)
        x = x.transpose(1,2)
        assert x.size(2) == 3
        if self.training:
            return self.forward_chunk(x)
        else:
            return self.forward_video(x)        

