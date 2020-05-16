import pickle
import pandas as pd
import numpy as np
import glob, os, os.path as osp

from tqdm import tqdm
from helper import load_roi, load_metadata


FACEROIS = '../../data/dfdc/jph/face_roi/'
ORIGINAL = '../../data/dfdc/videos/'

real_rois = {}
for _ in roi_dicts:
    real_rois.update({k : v for k,v in _.items() if vid2label[k.split('/')[-1]] == 'REAL'})

metadatas = [osp.join(_, 'metadata.json') for _ in glob.glob(osp.join(ORIGINAL, '*'))]
meta_df = pd.concat([load_metadata(_) for _ in tqdm(metadatas, total=len(metadatas))])
fake_df = meta_df[meta_df['video_label'] == 'FAKE']
real_df = meta_df[meta_df['video_label'] == 'REAL']
# Make a mapping from FAKE to REAL
fake2real = {
    osp.join(fake_df['train_part'].iloc[_], fake_df['filename'].iloc[_]) : osp.join(fake_df['train_part'].iloc[_], fake_df['original'].iloc[_]) 
    for _ in range(len(fake_df))
    }
real2fake = {
    osp.join(real_df['train_part'].iloc[_], real_df['filename'].iloc[_]) : []
    for _ in range(len(real_df))
    }
for k,v in fake2real.items():
    real2fake[v].append(k)

# Make a mapping from video to label
vid2label = {meta_df['filename'].iloc[_] : meta_df['video_label'].iloc[_] for _ in range(len(meta_df))}

roi_pickles = glob.glob(osp.join(FACEROIS, '*.pickle'))
roi_dicts = [load_roi(p) for p in tqdm(roi_pickles, total=len(roi_pickles))]

real_rois = {}
for _ in roi_dicts:
    real_rois.update({k : v for k,v in _.items() if vid2label[k.split('/')[-1]] == 'REAL'})

fake_keys = np.sort([*fake2real])
fake_rois = {k : real_rois[fake2real[k]] for k in fake_keys}

np.random.seed(88)
subsampled_fakes = {}
for k,v in real2fake.items():
    randomv = np.random.choice(v)
    subsampled_fakes[randomv] = fake_rois[randomv]

real_rois.update(subsampled_fakes)

single_frame_rois = {}
np.random.seed(88)
for k,v in real_rois.items():
    if np.sum([type(_) == type(None) for _ in v]) == len(v):
        continue
    index = np.random.choice(len(v))
    while v[index] is None:
        index = np.random.choice(len(v))
    sampled_roi = v[index][:3] # up to 3 faces
    single_frame_rois[k] = (sampled_roi, index)

# Turn into DataFrame
frame_roi_df = {
    'filename': [],
    'frame_index': [],
    'x1': [],
    'y1': [],
    'x2': [],
    'y2': [],
    'num_face': []
}
for k,v in single_frame_rois.items():
    for _ in range(len(v[0])):
        frame_roi_df['filename'].append(k)
        frame_roi_df['frame_index'].append(v[1])
        x1, y1, x2, y2 = v[0][_]*2
        frame_roi_df['x1'].append(int(x1))
        frame_roi_df['y1'].append(int(y1))
        frame_roi_df['x2'].append(int(x2))
        frame_roi_df['y2'].append(int(y2))
        frame_roi_df['num_face'].append(_)

frame_roi_df = pd.DataFrame(frame_roi_df)
for corner in ['x1','y1','x2','y2']:
    frame_roi_df.loc[frame_roi_df[corner] < 0, corner] = 0

frame_roi_df.to_csv('../../data/dfdc/mtcnn_frame_face_rois.py', index=False)

# Check ...
import decord
import cv2

SAVEDIR = '../../data/dfdc/check-mtcnn/'
SAVEFRAMEDIR = '../../data/dfdc/frame-mtcnn/'
if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

if not osp.exists(SAVEFRAMEDIR): os.makedirs(SAVEFRAMEDIR)

start = 13793
for _ in tqdm(range(13793, len(frame_roi_df)), total=len(frame_roi_df)-13793):
    filename = frame_roi_df['filename'].iloc[_]
    try:
        vr = decord.VideoReader(osp.join(ORIGINAL, filename), ctx=decord.cpu())
        frame_index = frame_roi_df['frame_index'].iloc[_] * 10
        frame = vr.get_batch([frame_index]).asnumpy()
    except Exception as e:
        print(e)
        continue
    # Save frame for fast loading loader
    part_save_dir = osp.join(SAVEFRAMEDIR, filename.split('/')[0])
    if not osp.exists(part_save_dir): os.makedirs(part_save_dir)
    status = cv2.imwrite(osp.join(part_save_dir, filename.split('/')[-1].replace('mp4', 'png')), frame[0][...,::-1])    
    # Get face
    x1, y1, x2, y2 = frame_roi_df.iloc[_][['x1','y1','x2','y2']]
    face = frame[0,y1:y2,x1:x2]
    # Save face
    part_save_dir = osp.join(SAVEDIR, filename.split('/')[0])
    filename = '{}_{}.png'.format(filename.split('/')[-1], frame_roi_df['num_face'].iloc[_])
    if not osp.exists(part_save_dir): os.makedirs(part_save_dir)
    status = cv2.imwrite(osp.join(part_save_dir, filename), face[...,::-1])










