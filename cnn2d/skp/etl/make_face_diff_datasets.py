import pandas as pd
import numpy as np
import glob
import os, os.path as osp


DIFFSDIR1 = '../../data/dfdc/diffs1/'
DIFFSDIR2 = '../../data/dfdc/diffs2/'


reals = glob.glob(osp.join(DIFFSDIR1, 'reals/*.png'))
reals.extend(glob.glob(osp.join(DIFFSDIR2, 'reals/*.png')))
fakes = glob.glob(osp.join(DIFFSDIR1, 'fakes/*.png'))
fakes.extend(glob.glob(osp.join(DIFFSDIR2, 'fakes/*.png')))
masks = glob.glob(osp.join(DIFFSDIR1, 'masks/*.npy'))
masks.extend(glob.glob(osp.join(DIFFSDIR2, 'masks/*.npy')))

df = pd.read_csv('../../data/dfdc/train.csv')

img_df = pd.DataFrame({
    'imgfile': [_.replace('../../data/dfdc/', '') for _ in reals+fakes],
    'label': [0] * len(reals) + [1] * len(fakes),
    })
mask_df = pd.DataFrame({
    'maskfile': [_.replace('../../data/dfdc/', '') for _ in masks]
    })
img_df['video'] = [_.split('/')[-1].replace('.png', '') for _ in img_df['imgfile']]
mask_df['video'] = [_.split('/')[-1].replace('.npy', '') for _ in mask_df['maskfile']]

img_df = img_df.merge(mask_df, on='video', how='left')
img_df['filename'] = ['{}.mp4'.format(_.split('_')[0]) for _ in img_df['video']]
img_df = img_df.merge(df[['filename', 'folder']], on='filename')
img_df['maskfile'] = img_df['maskfile'].fillna('empty_mask')
img_df['split'] = 'train'
img_df.loc[img_df['folder'].isin(['dfdc_train_part_{}'.format(i) for i in [45,46,47,48,49]]), 'split'] = 'valid'

train_df = img_df[img_df['split'] == 'train']
valid_df = img_df[img_df['split'] == 'valid']

# Sample to 50/50 validation
np.random.seed(88)
valid_videos_to_use = []
for part, _df in valid_df.groupby('folder'):
    reals = np.unique(_df[_df['label'] == 0]['filename'])
    fakes = np.unique(_df[_df['label'] == 1]['filename'])
    fakes = np.random.choice(fakes, len(reals), replace=False)
    valid_videos_to_use.extend(reals)
    valid_videos_to_use.extend(fakes)

valid_df = valid_df[valid_df['filename'].isin(valid_videos_to_use)]

img_df = pd.concat([train_df, valid_df])
img_df = img_df.sample(frac=1).reset_index(drop=True)

img_df.to_csv('../../data/dfdc/train_diff_df.csv', index=False)