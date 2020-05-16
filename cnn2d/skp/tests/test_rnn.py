import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from factory.models.backbones import resnet34
from factory.models import RNNHead

X = torch.from_numpy(np.ones((8,3,8,224,224))).float().cuda()

model = RNNHead(backbone='resnet50', pretrained=None, num_classes=1, dropout=0.2)
model = model.train().cuda()

yhat = model(X)

backbone, _ = resnet50(pretrained=None)
feats = []
for _ in range(X.size(1)):
    feats.append(backbone(X[:,_,...]))

feats = torch.stack(feats, dim=1)

lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=2, batch_first=True, bidirectional=True)
conv = nn.Conv1d(21, 1, kernel_size=1, stride=1, bias=False)


