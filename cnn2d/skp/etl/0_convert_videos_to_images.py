import cv2
import glob
import os, os.path as osp

from tqdm import tqdm


def load_video(vidfile, num_frames=64):
    cap = cv2.VideoCapture(vidfile)
    ret, frame = cap.read()
    frames = [frame]
    while ret and len(frames) < num_frames:
        ret, frame = cap.read()
        if type(frame) != type(None):
            frames.append(frame)
    assert len(frames) == num_frames
    cap.release()
    return frames


VIDEODIR = '../../data/dfdc/jph/videos/'
IMAGEDIR = '../../data/dfdc/jph/images/'

videos = glob.glob(osp.join(VIDEODIR, '*/*'))
lenv = len(videos)
videos = [v for v in videos if v.split('.')[-1] != 'short']
print('{} videos removed ...'.format(lenv-len(videos)))

for v in tqdm(videos, total=len(videos)):
    tmp_video = load_video(v)
    tmp_image_dir = osp.join(IMAGEDIR, v.split('/')[-1].split('.')[0])
    if not osp.exists(tmp_image_dir): os.makedirs(tmp_image_dir)
    for ind, frame in enumerate(tmp_video):
        status = cv2.imwrite(osp.join(tmp_image_dir, 'FRAME_{:03d}.png'.format(ind)), frame)
