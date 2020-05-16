import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--save-failures', type=str, default='./bad_videos.txt')
    return parser.parse_args()


import pandas as pd
import numpy as np
import glob
import os, os.path as osp
#import skvideo.io

from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper import *


args = parse_args()

FACEROIS = '../../data/dfdc/jph/face_roi/'
ORIGINAL = '../../data/dfdc/videos/'
SAVEDIR  = '../../data/dfdc/mini-numpy/'
NUMFRAMES = 30

if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

metadatas = [osp.join(_, 'metadata.json') for _ in glob.glob(osp.join(ORIGINAL, '*'))]
meta_df = pd.concat([load_metadata(_) for _ in tqdm(metadatas, total=len(metadatas))])
fake_df = meta_df[meta_df['video_label'] == 'FAKE']
# Make a mapping from FAKE to REAL
fake2real = {
    osp.join(fake_df['train_part'].iloc[_], fake_df['filename'].iloc[_]) : osp.join(fake_df['train_part'].iloc[_], fake_df['original'].iloc[_]) 
    for _ in range(len(fake_df))
    }
# Make a mapping from video to label
vid2label = {meta_df['filename'].iloc[_] : meta_df['video_label'].iloc[_] for _ in range(len(meta_df))}

roi_pickles = glob.glob(osp.join(FACEROIS, '*.pickle'))
roi_dicts = [load_roi(p) for p in tqdm(roi_pickles, total=len(roi_pickles))]

#########
# REALS #
#########
real_rois = {}
for _ in roi_dicts:
    real_rois.update({k : v for k,v in _.items() if vid2label[k.split('/')[-1]] == 'REAL'})

if args.mode == 'real':
    real_keys = np.sort([*real_rois])
    if args.end == -1:
        real_keys = real_keys[args.start:]
    else:
        real_keys = real_keys[args.start:args.end+1]
    filenames = [osp.join(ORIGINAL, _) for _ in real_keys]

    dataset = VideoFirstFramesDataset(videos=filenames, NUMFRAMES=NUMFRAMES)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    bad_videos = []
    for ind, vid in tqdm(enumerate(loader), total=len(loader)):
        try:
            v = loader.dataset.videos[ind].replace(ORIGINAL, '')
            if type(vid) != list:
                print('{} failed !'.format(v))
                bad_videos.append(v)
                continue
            vid, start = vid
            faces = get_faces(vid[0].numpy(), real_rois[v], start, NUMFRAMES=NUMFRAMES)
            if len(faces) == 0:
                bad_videos.append(v)
            #     print('0 faces detected for {} !'.format(v))
            for ind, f in enumerate(faces):
                if f.shape[1] > 256:
                    scale = 256./f.shape[1]
                    f = zoom(f, [1, scale,scale, 1], order=1, prefilter=False)
                    assert f.shape[1:3] == (256, 256)
                train_part = int(v.split('/')[-2].split('_')[-1])
                filename = v.split('/')[-1].replace('.mp4', '')
                filename = osp.join(SAVEDIR, '{:02d}_{}_{}.mp4'.format(train_part, filename, ind))
                np.save(filename, f.astype('uint8'))
                #skvideo.io.vwrite(filename, f.astype('uint8'))
        except Exception as e:
            print(e)
            bad_videos.append(v)

#########
# FAKES #
#########
if args.mode == 'fake':
    fake_keys = np.sort([*fake2real])
    if args.end == -1:
        fake_keys = fake_keys[args.start:]
    else:
        fake_keys = fake_keys[args.start:args.end+1]
    fake_rois = {k : real_rois[fake2real[k]] for k in fake_keys}
    filenames = [osp.join(ORIGINAL, _) for _ in [*fake_rois]]

    dataset = VideoFirstFramesDataset(videos=filenames, NUMFRAMES=NUMFRAMES)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    bad_videos = []
    for ind, vid in tqdm(enumerate(loader), total=len(loader)):
        try:
            v = loader.dataset.videos[ind].replace(ORIGINAL, '')
            if type(vid) != list:
                print('{} failed !'.format(v))
                bad_videos.append(v)
                continue
            vid, start = vid
            faces = get_faces(vid[0].numpy(), real_rois[fake2real[v]], start, NUMFRAMES=NUMFRAMES)
            if len(faces) == 0:
                bad_videos.append(v)
            #     print('0 faces detected for {} !'.format(v))
            for ind, f in enumerate(faces):
                if f.shape[1] > 256:
                    scale = 256./f.shape[1]
                    f = zoom(f, [1, scale,scale, 1], order=1, prefilter=False)
                    assert f.shape[1:3] == (256, 256)
                train_part = int(v.split('/')[-2].split('_')[-1])
                filename = v.split('/')[-1].replace('.mp4', '')
                filename = osp.join(SAVEDIR, '{:02d}_{}_{}.mp4'.format(train_part, filename, ind))
                np.save(filename, f.astype('uint8'))
                #skvideo.io.vwrite(filename, f.astype('uint8'))
        except Exception as e:
            print(e)
            bad_videos.append(v)


with open(args.save_failures, 'a') as f:
    for bv in bad_videos:
        f.write('{}\n'.format(bv))
