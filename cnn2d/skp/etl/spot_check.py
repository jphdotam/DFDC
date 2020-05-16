import numpy as np
import decord
import glob, os, os.path as osp
import cv2

from tqdm import tqdm


VIDEODIR = '../../data/dfdc/mini-videos/'
CHECKDIR = '../../data/dfdc/spot-check/'

if not osp.exists(CHECKDIR): os.makedirs(CHECKDIR)

existing = glob.glob(osp.join(CHECKDIR, '*.png'))
existing = [_.split('/')[-1].replace('png','mp4') for _ in existing]

videos = glob.glob(osp.join(VIDEODIR, '*.mp4'))
videos = [_ for _ in videos if _.split('/')[-1] not in existing]

for i in tqdm(videos, total=len(videos)):
    vr = decord.VideoReader(i, ctx=decord.cpu())
    x = vr[0].asnumpy()
    y = vr[-1].asnumpy()
    x = x[...,::-1]
    y = y[...,::-1]
    x = np.hstack([x,y])
    status = cv2.imwrite(osp.join(CHECKDIR, i.split('/')[-1].replace('mp4','png')), cv2.resize(x, (256,128)))
