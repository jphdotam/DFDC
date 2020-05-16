# Builds off of James' dataset ...
import pandas as pd
import numpy as np
import glob
import os, os.path as osp
import json


def load_metadata(jsonfile):
    with open(jsonfile) as f:
        x = json.load(f)
    part = jsonfile.split('/')[-2]
    videos = []
    labels = []
    splits = []
    originals = []
    for k,v in x.items():
        videos.append(k)
        labels.append(v['label'])
        splits.append(v['split'])
        try:
            originals.append(v['original'])
        except KeyError:
            originals.append(None)
    return pd.DataFrame({
        'filename': videos,
        'video_label': labels, 
        'split': splits,
        'original': originals,
        'train_part': [part] * len(videos)
        })


VIDEODIR = '../../data/dfdc/jph/videos/'
ORIGINAL = '../../data/dfdc/videos/'

videos = glob.glob(osp.join(VIDEODIR, '*/*'))
videos = [v for v in videos if v.split('.')[-1] != 'short']
video_df = pd.DataFrame({
    'vidfile': videos,
    'label': [v.split('/')[-2] for v in videos],
    'part': [int(v.split('/')[-1].split('_')[0]) for v in videos],
    'filename': [v.split('/')[-1].split('_')[1] for v in videos]
    })

metadatas = [osp.join(_, 'metadata.json') for _ in glob.glob(osp.join(ORIGINAL, '*'))]
meta_df = pd.concat([load_metadata(_) for _ in metadatas])

video_df = video_df.merge(meta_df, on='filename')
assert np.sum(video_df['label'] == video_df['video_label'])

video_df['label'] = [1 if _ == 'FAKE' else 0 for _ in video_df['label']]
video_df['label'].value_counts()

video_df.loc[video_df['part'].isin(list(range(45,50))), 'split'] = 'valid'
video_df['split'].value_counts()

train_df = video_df[video_df['split'] == 'train']
valid_df = video_df[video_df['split'] == 'valid']
subsampled_valid_df = []
for p in valid_df['part'].unique():
    tmp_reals = valid_df[((valid_df['part'] == p) & (valid_df['label'] == 0))]
    tmp_fakes = valid_df[((valid_df['part'] == p) & (valid_df['label'] == 1))].sample(n=len(tmp_reals), random_state=0)
    subsampled_valid_df.append(pd.concat([tmp_reals, tmp_fakes]))

valid_df = pd.concat(subsampled_valid_df)
valid_df['label'].value_counts()

video_df = pd.concat([train_df, valid_df]).sample(frac=1, random_state=0)

video_df['vidfile'] = [_.replace(VIDEODIR, '') for _ in video_df['vidfile']]
video_df.to_csv('../../data/dfdc/jph/train_video_with_splits.csv', index=False)



