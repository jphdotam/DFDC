from factory.models.classifier import TDN

import torch, numpy as np


X = torch.from_numpy(np.ones((8,3,2,224,224))).float().cuda()
m = TDN('resnet34_fm', num_classes=2, dropout=0.2, pretrained='imagenet')
m = m.train().cuda()

m(X)


ps = m.i_net(X[:,:,0])
for p in ps: print(p.shape)

ds = m.d_net(X[:,:,1]-X[:,:,0], skips=ps)
for d in ds: print(d.shape)
