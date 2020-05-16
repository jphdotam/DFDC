import decord
import pandas as pd
import numpy as np
import glob, os, os.path as osp
import cv2

from tqdm import tqdm


def load_first_frame(filename):
    vr = decord.VideoReader(filename, ctx=decord.cpu())
    return vr.get_batch([0]).asnumpy()[0]


SAVEDIR = '../../data/dfdc/jph/spot-check/'
if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

videos = glob.glob('../../data/dfdc/jph/videos/*/*.mp4')
video_ids = [_.split('/')[-1].split('_')[1] for _ in videos]
video_df = pd.DataFrame({
        'vidfile':  videos,
        'video_id': video_ids
    })
video_ids, counts = np.unique(video_ids, return_counts=True)
videos_multiface = video_ids[counts > 1]

video_df = video_df[video_df['video_id'].isin(videos_multiface)]

# Assume if 1 face was detected, then it's valid 
# So only check videos with >1 face

for v in tqdm(video_df['vidfile'], total=len(video_df['vidfile'])):
    frame = load_first_frame(v)
    frame = cv2.resize(frame, (128,128))
    status = cv2.imwrite(osp.join(SAVEDIR, '{}_{}'.format(v.split('/')[-2], v.split('/')[-1].replace('mp4', 'png'))), frame)

