import pandas as pd
import numpy as np
import os, os.path as osp
import cv2

from tqdm import tqdm


DATADIR = '/home/ianpan/ufrc/deepfake/data/dfdc/'
SAVEDIR = osp.join(DATADIR, 'pairs/')
if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)


df = pd.read_csv(osp.join(DATADIR, 'train_manyfaces_with_splits.csv'))

# Get 10th frame
df['frame_index'] = [int(_.split('/')[-1].split('-')[0].replace('FRAME', '')) for _ in df['imgfile']]
df = df[df['frame_index'] == 10]
df['face_number'] = [int(_.split('/')[-1].split('-')[1].split('.')[0].replace('FACE', '')) for _ in df['imgfile']]

df = df[df['label'] == 1]

def join_images(x, y):
    xh, xw = x.shape[:2]
    yh, yw = y.shape[:2]
    ratio = xh/yh
    y = cv2.resize(y, (int(yw*ratio), int(xh)))
    return np.hstack((x,y))

RESIZE_H = 150
for orig,_df in tqdm(df.groupby('original'), total=len(df['original'].unique())):
    for face_num, face_df in _df.groupby('face_number'):
        # Load in original face
        original_facefile = face_df['imgfile'].iloc[0].replace(face_df['filename'].iloc[0].replace('.mp4', ''), orig.replace('.mp4', ''))
        original_face = cv2.imread(osp.join(DATADIR, original_facefile))
        if type(original_face) == type(None):
            print('{} not found ! Skipping ...'.format(original_facefile))
            continue
        for fake_face in face_df['imgfile']:
            ff = cv2.imread(osp.join(DATADIR, fake_face))
            joined_image = join_images(original_face, ff)
            h, w = joined_image.shape[:2]
            ratio = RESIZE_H/h
            joined_image = cv2.resize(joined_image, (int(w*ratio), int(h*ratio)))
            tmp_save_dir = osp.join(SAVEDIR, face_df['folder'].iloc[0], orig.replace('.mp4', ''))
            if not osp.exists(tmp_save_dir): os.makedirs(tmp_save_dir)
            savefile = '{}_{}.png'.format(fake_face.split('/')[-2], face_num)
            cv2.imwrite(osp.join(tmp_save_dir, savefile), joined_image)


