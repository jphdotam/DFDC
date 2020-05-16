from torch.utils.data import Dataset, Sampler

import torch
import pandas as pd
import numpy as np
import math
import time
import cv2


class FaceVideoDataset(Dataset):

    def __init__(self, 
                 vidfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 max_frames,
                 test_frames=32,
                 test_mode=False,
                 grayscale=False,
                 to_rgb=True,
                 frame_skip=1):

        self.vidfiles = vidfiles
        self.labels = labels
        self.videos = [_.split('/')[-1].split('_')[1] for _ in vidfiles]
        self.parts  = [int(_.split('/')[-1].split('_')[0]) for _ in vidfiles]
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.max_frames = max_frames
        self.test_frames = test_frames
        self.test_mode = test_mode
        self.grayscale = grayscale
        self.badfiles = []
        self.to_rgb = to_rgb
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.vidfiles)

    # James' method
    def load_video(self, filename, every_n_frames=1, rescale=None):
        cap = cv2.VideoCapture(filename)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        NUMFRAMES = self.test_frames if self.test_mode else frameCount

        if rescale:
            out_video = np.zeros(
                (math.ceil(NUMFRAMES / every_n_frames), int(frameHeight * rescale), int(frameWidth * rescale), 3),
                np.dtype('uint8'))
        else:
            out_video = np.zeros((math.ceil(NUMFRAMES / every_n_frames), frameHeight, frameWidth, 3),
                                 np.dtype('uint8'))

        i_frame = 0
        ret = True

        while (i_frame * every_n_frames < NUMFRAMES and ret):
            cap.set(cv2.CAP_PROP_FRAME_COUNT, (i_frame * every_n_frames) - 1)
            ret, frame = cap.read()
            if rescale:
                frame = cv2.resize(frame, (0, 0), fx=rescale, fy=rescale)
            if self.to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_video[i_frame] = frame
            i_frame += 1

        cap.release()
        return out_video

    def load_pairs(self, filename):
        cap = cv2.VideoCapture(filename)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = np.random.randint(0, frameCount-self.frame_skip)
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame1 = cap.read() 
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+self.frame_skip)
        ret, frame2 = cap.read()
        if self.to_rgb:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        return np.asarray([frame1, frame2])

    def process_video(self, X):
        if self.pad: X = np.asarray([self.pad(_) for _ in X]) 
        if self.resize: X = np.asarray([self.resize(image=_)['image'] for _ in X]) 
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.transform: 
            to_transform = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_transform.update({'image': X[0]})
            transformed = self.transform(**to_transform)
            X = np.asarray([transformed['image']] + [transformed['image{}'.format(_)] for _ in range(1,len(X))])
        if self.grayscale: 
            assert self.to_rgb
            X = np.asarray([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in X])
            X = np.expand_dims(X, axis=-1)
        if self.preprocessor: X = self.preprocessor.preprocess(X)
        # X.shape = (N, H, W, C)
        return X.transpose(3, 0, 1, 2) # X.shape = (C, N, H, W)

    def fetch_frames(self, X):
        if self.test_mode: 
            return X[:self.test_frames]
        else:
            start = len(X) - self.max_frames
            assert start >= 0
            if start > 0:
                start = np.random.randint(start)
            return X[start:(start+self.max_frames)]

    def __getitem__(self, i):
        # We assume that the chance of sampling 2 consecutive
        # bad files is too small to be relevant
        try:
            if self.max_frames == 'pairs' and not self.test_mode:
                X = self.load_pairs(self.vidfiles[i])
            else:
                X = self.load_video(self.vidfiles[i])
                X = self.fetch_frames(X)
        except:
            self.bad_files.append(i)
            indices = list(range(len(self.vidfiles)))
            indices = list(set(indices) - set(self.badfiles))
            i = np.random.choice(indices)
            if self.max_frames == 'pairs' and not self.test_mode:
                X = self.load_pairs(self.vidfiles[i])
            else:
                X = self.load_video(self.vidfiles[i])
                X = self.fetch_frames(X)

        X = self.process_video(X)
        y = self.labels[i]
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return X, y

class PartSampler(Sampler):

    def __init__(self,
        dataset):
        super().__init__(data_source=dataset)
        parts  = np.asarray(dataset.parts)
        videos = np.asarray(dataset.vidfiles)        
        labels = np.asarray(dataset.labels)
        self.reals_dict = {p : np.where((parts == p) & (labels == 0))[0] for p in np.unique(parts)}
        self.fakes_dict = {p : np.where((parts == p) & (labels == 1))[0] for p in np.unique(parts)}
        # 1 epoch = # of real videos
        # FAKE = 1 for my labels
        self.length = len(dataset) - np.sum(dataset.labels)

    def __iter__(self):
        all_indices = []
        parts = np.unique(list(self.reals_dict.keys()))
        while len(all_indices) < self.length:
            # Shuffle parts
            parts = np.random.permutation(parts)
            # Get reals from the first half
            # Fakes from the second half
            shuf1 = parts[:len(parts)//2]
            shuf2 = parts[len(parts)//2:]
            # Now, sample a real video from shuf1
            real_indices = []
            for i in shuf1:
                real_indices.append(np.random.choice(self.reals_dict[i]))
            # Sample a fake video from shuf2
            fake_indices = []
            for i in shuf2:
                fake_indices.append(np.random.choice(self.fakes_dict[i]))
            # Merge alternating
            indices = real_indices + fake_indices
            # List assigned to [::2] needs to be the longer one
            # So try-except since the max length difference will always be 1
            try:
                indices[::2]  = real_indices
                indices[1::2] = fake_indices
            except ValueError:
                indices[::2]  = fake_indices
                indices[1::2] = real_indices
            # Permute
            indices = np.random.permutation(indices)
            all_indices.extend(indices)
        return iter(all_indices)

    def __len__(self):
        return self.length