import pandas as pd
import numpy as np
import decord
import imagehash
import glob, os, os.path as osp

from tqdm import tqdm
from PIL import Image


def load_video(filename, max_frames=100):
    vr = decord.VideoReader(filename, ctx=decord.cpu())
    return np.asarray([vr[i].asnumpy() for i in range(max_frames)])


def compute_hashes(x, y):
    hash_funcs = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]
    x = Image.fromarray(x.astype('uint8'))
    y = Image.fromarray(y.astype('uint8'))
    hash_differences = []
    for hfunc in hash_funcs:
        hash_differences.append(hfunc(x)-hfunc(y))
    return hash_differences


VIDEODIR = '/home/ianpan/ufrc/deepfake/data/dfdc/videos/'

videos = glob.glob(osp.join(VIDEODIR, '*/*.mp4'))

df = pd.read_csv('/home/ianpan/ufrc/deepfake/data/dfdc/train.csv')
df = df[df['label'] == 'FAKE']

hash_diffs = {}
for orig, _df in tqdm(df.groupby('original'), total=len(df['original'].unique())):
    orig_filepath = _df['filepath'].iloc[0].replace(_df['filename'].iloc[0], orig)
    orig_video = load_video(osp.join(VIDEODIR, orig_filepath))
    for fake_rownum, fake_row in _df.iterrows():
        fake_video = load_video(osp.join(VIDEODIR, fake_row['filepath']))
        hash_diffs[fake_row['filename']] = []
        for real_frame, fake_frame in zip(orig_video, fake_video):
            hash_diffs[fake_row['filename']].append(compute_hashes(real_frame, fake_frame))





