import argparse
import pandas as pd
import numpy as np
import decord
import cv2
import os, os.path as osp

from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm
from facenet_pytorch import MTCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    return parser.parse_args()


def load_real_and_fake_frame(real, fake, random):
    vr = decord.VideoReader(real, ctx=decord.cpu())
    frame_index = list(np.random.choice(range(len(vr)), random))
    real = vr.get_batch(frame_index).asnumpy()
    vr = decord.VideoReader(fake, ctx=decord.cpu())
    fake = vr.get_batch(frame_index).asnumpy()
    return real, fake, frame_index


def get_face(real, fake, mtcnn, resize=None):
    # real.shape = fake.shape = (N, H, W, C)
    if resize:
        new_shape = (int(real.shape[2]*resize), int(real.shape[1]*resize))
        img = np.asarray([cv2.resize(real[i], new_shape) for i in range(len(real))])
    else:
        img = real
    boxes, probs = mtcnn.detect(img)
    if boxes.ndim == 1:
        # This means different # of faces detected per image
        # Just take the top one
        boxes = np.asarray([boxes[i][0] for i in range(len(boxes))])
        boxes = np.expand_dims(boxes, axis=1)
    # boxes.shape = (N, num_face, 4)
    boxes = (boxes / resize).astype('int')
    N, nf = boxes.shape[:2]
    real_faces = []
    fake_faces = []
    for num_sample in range(N):
        for num_face in range(nf):
            x1, y1, x2, y2 = boxes[num_sample, num_face]
            real_faces.append(real[num_sample, y1:y2, x1:x2])
            fake_faces.append(fake[num_sample, y1:y2, x1:x2])
    return real_faces, fake_faces


def generate_noise_diff(img):
    tv = denoise_tv_chambolle(img, weight=0.2, multichannel=True)
    noise = (tv - img / 255.) ** 2
    noise -= np.min(noise)
    noise /= np.max(noise)
    noise *= 255
    return noise.astype('uint8')


args = parse_args()

# Setup MTCNN
MTCNN_THRESHOLDS = (0.8, 0.8, 0.8)
MTCNN_FACTOR = 0.71

mtcnn = MTCNN(margin=0, 
              keep_all=True, 
              post_process=False, 
              select_largest=False, 
              thresholds=MTCNN_THRESHOLDS, 
              factor=MTCNN_FACTOR)


df = pd.read_csv('../../data/dfdc/train.csv')
fake_df = df[df['label'] == 'FAKE']

if args.end == -1:
    fake_df = fake_df.iloc[args.start:]
else:
    fake_df = fake_df.iloc[args.start:args.end]

VIDEODIR = '../../data/dfdc/videos/'
SAVEREALDIR = '../../data/dfdc/noise/reals/'
SAVEFAKEDIR = '../../data/dfdc/noise/fakes/'


if not osp.exists(SAVEREALDIR): os.makedirs(SAVEREALDIR)

if not osp.exists(SAVEFAKEDIR): os.makedirs(SAVEFAKEDIR)

failed = []
for rownum, row in tqdm(fake_df.iterrows(), total=len(fake_df)):
    try:
        real, fake, frame_index = load_real_and_fake_frame(osp.join(VIDEODIR, row['folder'], row['original']),
                                              osp.join(VIDEODIR, row['filepath']),
                                              random=5)
        real_faces, fake_faces = get_face(real, fake, mtcnn, resize=0.5)
        real_noise = [generate_noise_diff(rf) for rf in real_faces]
        fake_noise = [generate_noise_diff(ff) for ff in fake_faces]        
        for i, (rn, fn) in enumerate(zip(real_noise, fake_noise)):
            status = cv2.imwrite(osp.join(SAVEREALDIR, '{}_{:02d}.png'.format(row['original'].split('.')[0], i)), rn[...,::-1])
            status = cv2.imwrite(osp.join(SAVEFAKEDIR, '{}_{:02d}.png'.format(row['filename'].split('.')[0], i)), fn[...,::-1])
    except Exception as e:
        print(e)
        failed.append(row['original'])


with open('FAILED.txt', 'a') as f:
    for failure in failed:
        f.write('{}\n'.format(failure))


# failed = []
# for original, _df in tqdm(fake_df.groupby('original'), total=len(fake_df['original'].unique())):
#     try:
#         real_vr = decord.VideoReader(osp.join(VIDEODIR, _df['folder'].iloc[0], original), ctx=decord.cpu())
#         frame_index = np.random.choice(len(real_vr), 5)
#         real = real_vr.get_batch(list(frame_index)).asnumpy()
#         boxes = get_coords(real, mtcnn, resize=0.5)
#         boxes[boxes < 0] = 0
#         N, nf = boxes.shape[:2]
#         real_faces = []
#         for num_sample in range(N):
#             for num_face in range(nf):
#                 x1, y1, x2, y2 = boxes[num_sample, num_face]
#                 real_faces.append(real[num_sample, y1:y2, x1:x2])
#         for i, rf in enumerate(real_faces):
#             status = cv2.imwrite(osp.join(SAVEREALDIR, '{}_{:02d}.png'.format(original.split('.')[0], i)), rf[...,::-1])
#         for rownum, row in _df.iterrows():
#             fake_vr = decord.VideoReader(osp.join(VIDEODIR, row['filepath']))
#             fake = fake_vr.get_batch(list(frame_index)).asnumpy()
#             fake_faces, diff_masks = [], []
#             for num_sample in range(N):
#                 for num_face in range(nf):
#                     x1, y1, x2, y2 = boxes[num_sample, num_face]
#                     fake_faces.append(fake[num_sample, y1:y2, x1:x2])            
#             diff_masks = [generate_mask(real_faces[i], fake_faces[i], sigma=3) for i in range(len(real_faces))]
#             for i, (ff, dm) in enumerate(zip(fake_faces, diff_masks)):
#                 status = cv2.imwrite(osp.join(SAVEFAKEDIR, '{}_{:02d}.png'.format(row['filename'].split('.')[0], i)), ff[...,::-1])
#                 status = cv2.imwrite(osp.join(SAVEMIMGDIR, '{}_{:02d}.png'.format(row['filename'].split('.')[0], i)), (dm*255).astype('uint8'))
#                 np.save(osp.join(SAVEMASKDIR, '{}_{:02d}.npy'.format(row['filename'].split('.')[0], i)), dm)
#     except Exception as e:
#         print(e)
#         failed.append(original)


# with open('FAILED.txt', 'a') as f:
#     for failure in failed:
#         f.write('{}\n'.format(failure))






