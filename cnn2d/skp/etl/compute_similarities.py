import skimage.measure
import decord
import pandas as pd
import numpy as np
import glob
import os, os.path as osp

from scipy.ndimage.interpolation import zoom
from helper import load_metadata
from tqdm import tqdm


def load_video(filepath, num_frames, scale_factor):
    vr = decord.VideoReader(filepath, ctx=decord.cpu())
    vid = vr[:NUMFRAMES].asnumpy()
    if scale_factor != 1:
        vid = zoom(vid, [1, scale_factor, scale_factor, 1], prefilter=False, order=0)
    return vid


ORIGINAL = '../../data/dfdc/videos/'
JPHVIDEO = '../../data/dfdc/jph/videos/'
SCALE = 1
NUMFRAMES = 150

metadatas = [osp.join(_, 'metadata.json') for _ in glob.glob(osp.join(ORIGINAL, '*'))]
meta_df = pd.concat([load_metadata(_) for _ in tqdm(metadatas, total=len(metadatas))])
jph_videos = glob.glob(osp.join(JPHVIDEO, '*/*.mp4'))
jph_df = pd.DataFrame({
        'jphfile': [_.split('/')[-1] for _ in jph_videos],
        'train_part': ['dfdc_train_part_{:02d}'.format(int(_.split('/')[-1].split('_')[0])) for _ in jph_videos],
        'video_label': [_.split('/')[-2] for _ in jph_videos],
        'filename': [_.split('/')[-1].split('_')[1] for _ in jph_videos],
        'suffix': [_.split('/')[-1].split('.mp4_')[-1] for _ in jph_videos]
    })
jph_df = jph_df.merge(meta_df, on=['train_part','video_label','filename'])
jph_df['jphorig'] = ['{}_{}_{}'.format(int(jph_df['train_part'].iloc[rownum].split('_')[-1]), jph_df['original'].iloc[rownum], jph_df['suffix'].iloc[rownum]) for rownum in range(len(jph_df))]

jph_df['jphfile'] = [osp.join(JPHVIDEO, '{}/{}'.format(jph_df['video_label'].iloc[rownum], jph_df['jphfile'].iloc[rownum])) for rownum in range(len(jph_df))]
jph_df['jphorig'] = [osp.join(JPHVIDEO, 'REAL/{}'.format(jph_df['jphorig'].iloc[rownum])) for rownum in range(len(jph_df))]

jph_fake_df = jph_df[jph_df['video_label'] == 'FAKE']
real2fake = {k : list(_df['']) for k,_df in jph_fake_df.groupby('jphorig')}


list_of_reals, list_of_fakes = [], []
list_of_ssim, list_of_psnr = [], []
for real,fakes in tqdm(real2fake.items(), total=len(real2fake)):
    try:
        real_video = load_video(real, NUMFRAMES, scale_factor=SCALE)
        for fake in fakes:
            fake_video = load_video(fake, NUMFRAMES, scale_factor=SCALE)
            list_of_reals.append(real)
            list_of_fakes.append(fake)
            tmp_ssims = []
            tmp_psnrs = []
            for frame in range(len(fake_video)):
                tmp_ssims.append(skimage.measure.compare_ssim(real_video[frame], fake_video[frame], multichannel=True))
                tmp_psnrs.append(skimage.measure.compare_psnr(real_video[frame], fake_video[frame]))
            list_of_ssim.append(tmp_ssims)
            list_of_psnr.append(tmp_psnrs)
        print('Working ...')
    except Exception as e:
        continue


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

    dataset = VideoDataset(videos=filenames, NUMFRAMES=NUMFRAMES)
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
                #np.save(filename, f.astype('uint8'))
                skvideo.io.vwrite(filename, f.astype('uint8'))
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

    dataset = VideoDataset(videos=filenames, NUMFRAMES=NUMFRAMES)
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
                #np.save(filename, f.astype('uint8'))
                skvideo.io.vwrite(filename, f.astype('uint8'))
        except Exception as e:
            print(e)
            bad_videos.append(v)


with open(args.save_failures, 'a') as f:
    for bv in bad_videos:
        f.write('{}\n'.format(bv))
