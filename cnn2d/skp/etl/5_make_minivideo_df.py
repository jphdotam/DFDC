import pandas as pd
import numpy as np
import glob
import os, os.path as osp

from tqdm import tqdm
from helper import *


ORIGINAL = '../../data/dfdc/videos/'
VIDEODIR = '../../data/dfdc/mini-videos/'

videos = glob.glob(osp.join(VIDEODIR, '*.mp4'))

with open('../../data/dfdc/videos_to_use.txt') as f:
    cleaned_videos = [_.strip().replace('png','mp4') for _ in f.readlines()]

video_df = pd.DataFrame({
        'vidfile': [_.split('/')[-1] for _ in videos]
    })
video_df = video_df[video_df['vidfile'].isin(cleaned_videos)]
video_df['filename'] = ['{}.mp4'.format(_.split('_')[1]) for _ in video_df['vidfile']]

metadatas = [osp.join(_, 'metadata.json') for _ in glob.glob(osp.join(ORIGINAL, '*'))]
meta_df = pd.concat([load_metadata(_) for _ in tqdm(metadatas, total=len(metadatas))])

video_df = video_df.merge(meta_df, on='filename')
video_df['part'] = [int(_.split('_')[-1]) for _ in video_df['train_part']]
video_df.loc[video_df['part'].isin([45,46,47,48,49]), 'split'] = 'valid'
video_df['split'].value_counts()

valid_df = video_df[video_df['split'] == 'valid']
new_valid_df = []
for _, _df in valid_df.groupby('part'):
    tmp_real = _df[_df['video_label'] == 'REAL']
    tmp_fake = _df[_df['video_label'] == 'FAKE'].sample(n=len(tmp_real), random_state=88)
    new_valid_df.extend([tmp_real, tmp_fake])

valid_df = pd.concat(new_valid_df)
train_df = video_df[video_df['split'] == 'train']

video_df = pd.concat([train_df, valid_df]).sample(frac=1).reset_index(drop=True)
video_df['split'].value_counts()

video_df['label'] = (video_df['video_label'] == 'FAKE').astype('float32')

video_df.to_csv('../../data/dfdc/train_minivideos_df.csv', index=False)
