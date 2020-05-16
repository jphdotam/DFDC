import torch, numpy as np

from factory.models.classifier import DiffModel

X = torch.from_numpy(np.ones((8,3,32,128,128))).float().cuda()

model = DiffModel('resnet34', 1, 0.2, 'imagenet').eval().cuda()
model(X)