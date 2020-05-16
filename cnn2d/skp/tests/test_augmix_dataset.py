import pandas as pd
import numpy as np
import yaml
import cv2
import os, os.path as osp

from torch.utils.data import DataLoader

from factory.data.datasets import FaceAugMixDataset
from factory.builder import build_dataloader

with open('configs/experiments/experiment028.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg['transform']['preprocess'] = None
df = pd.read_csv('../data/dfdc/train_minivideos_df.csv')

train_loader = build_dataloader(cfg, data_info={'vidfiles': [osp.join(cfg['dataset']['data_dir'], _) for _ in df['vidfile']], 'labels': list(df['label'])}, mode='train')
valid_loader = build_dataloader(cfg, data_info={'vidfiles': [osp.join(cfg['dataset']['data_dir'], _) for _ in df['vidfile']], 'labels': list(df['label'])}, mode='valid')
#train_loader.dataset.transform = None
train_loader.dataset.preprocessor = None

train_iter = iter(train_loader)
train_data = next(train_iter)

import matplotlib.pyplot as plt
for i in range(len(train_data[0]['orig'])):
    plt.subplot(1,3,1)
    plt.imshow(train_data[0]['orig'][i].numpy().transpose(1,2,0).astype('uint8'))
    plt.subplot(1,3,2)
    plt.imshow(train_data[0]['aug1'][i].numpy().transpose(1,2,0).astype('uint8'))
    plt.subplot(1,3,3)
    plt.imshow(train_data[0]['aug2'][i].numpy().transpose(1,2,0).astype('uint8'))
    plt.show()

valid_loader = iter(valid_loader)
valid_data = next(valid_loader)

# Save and examine
for ind, i in enumerate(train_data[0]):
    for j in range(2):
        status = cv2.imwrite('/home/ianpan/TEST{:02d}_{:02d}.png'.format(ind,j), i[:,j].cpu().numpy().transpose(1,2,0))


dset = FaceVideoDataset(vidfiles=['../data/dfdc/jph/videos/'+_ for _ in list(df['vidfile'])],
                        labels=list(df['label']),
                        pad=None,
                        resize=None,
                        crop=None,
                        transform=None,
                        preprocessor=None,
                        max_frames=32)

sampler = PartSampler(dset)

indices = list(sampler.__iter__())
vidfiles = np.asarray(df['vidfile'])[indices]

vidfiles50 = vidfiles[:100]

l,p = [],[]
for v in vidfiles50:
    l.append(v.split('/')[-2])
    p.append(int(v.split('/')[-1].split('_')[0]))

np.unique(l, return_counts=True)
np.unique(p, return_counts=True)

#

loader = DataLoader(dataset=dset, batch_size=8, num_workers=0)
loader = iter(loader)
data = next(loader)