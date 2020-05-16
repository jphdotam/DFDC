import cv2
import glob
import os, os.path as osp
import numpy as np


MASKDIR = 'diffs2/mimgs/'
FAKEDIR = 'diffs2/fakes/'
COMBDIR = 'diffs2/overlay/'

if not osp.exists(COMBDIR): os.makedirs(COMBDIR)

masks = np.sort(glob.glob(osp.join(MASKDIR, '*')))

for m in masks:
    ma = cv2.imread(m)
    if np.sum(ma) == 0:
        continue
    fa = cv2.imread(m.replace(MASKDIR, FAKEDIR))
    combined = cv2.addWeighted(fa, 0.8, ma, 0.4, 0)
    status = cv2.imwrite(osp.join(COMBDIR, m.split('/')[-1]), combined)

