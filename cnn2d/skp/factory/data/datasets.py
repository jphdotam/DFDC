from torch.utils.data import Dataset, Sampler

import torch
try:
    import decord
except:
    print ('`decord` not available !')
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
                 frame_skip=1,
                 flip=False,
                 rgb_shuffle=False):

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
        self.flip = flip
        self.rgb_shuffle = rgb_shuffle

        self.bad_files = []

    def __len__(self):
        return len(self.vidfiles)

    @staticmethod
    def _load_video(filename): 
        return decord.VideoReader(filename, ctx=decord.cpu())

    def load_whole_video(self, filename):
        vr = self._load_video(filename)
        return vr.get_batch(list(range(len(vr)))).asnumpy()

    def load_video_chunk(self, filename, num_frames, start=None):
        vr = self._load_video(filename)
        if type(start) == type(None): start = np.random.choice(range(len(vr)-num_frames+1))
        if self.test_mode:
            assert start == 0
            #num_frames = min(len(vr), num_frames)  
        return vr.get_batch(list(range(start, start+num_frames))).asnumpy()

    def load_video_frame(self, filename, index=None):
        vr = self._load_video(filename)
        if type(index) == type(None): index = np.random.choice(range(len(vr)))
        return vr.get_batch([index]).asnumpy()

    def load_video_pairs(self, filename, index=None):
        vr = self._load_video(filename)
        if type(index) == type(None): index = np.random.choice(range(len(vr)-self.frame_skip))
        return vr.get_batch([index, index+self.frame_skip]).asnumpy()

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

    def get(self, i):
        try:
            if self.max_frames == 'pairs' and not self.test_mode:
                X = self.load_video_pairs(self.vidfiles[i])
            elif self.max_frames == 1 and not self.test_mode:
                X = self.load_video_frame(self.vidfiles[i])
            else:
                start_frame = 0 if self.test_mode else None
                num_frames = self.test_frames if self.test_mode else self.max_frames
                X = self.load_video_chunk(self.vidfiles[i], num_frames=num_frames, start=start_frame)
        except Exception as e:
            print(e)
            X = None
        return X

    def __getitem__(self, i):
        X = self.get(i)
        while type(X) == type(None):
            i = np.random.choice(len(self.vidfiles))
            X = self.get(i)

        X = self.process_video(X)
        if not self.test_mode:
            assert X.ndim == 4
            if self.flip:
                # X.shape = (C, T, H, W)
                mode = np.random.randint(4)
                if mode == 1: # horizontal flip
                    X = X[:,:,:,::-1]
                elif mode == 2: # vertical flip
                    X = X[:,:,::-1,:]
                elif mode == 3: # both flips
                    X = X[:,:,::-1,::-1]
                X = np.ascontiguousarray(X)
            if self.rgb_shuffle:
                # channel should be dim 0
                channel_order = np.random.permutation(np.arange(X.shape[0]))
                X = X[channel_order]

        if self.max_frames == 1 and not self.test_mode:
            X = X[:,0]

        y = self.labels[i]
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return X, y


class FaceAugMixDataset(FaceVideoDataset):

    # Assume single frame only
    def process_image(self, X):
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        if self.crop: 
            X = self.crop(image=X)['image']
        if self.transform: 
            X = self.transform(image=X)['image']
        if self.preprocessor: 
            X = self.preprocessor.preprocess(X)
        # X.shape = (H, W, C)
        return X.transpose(2, 0, 1) # X.shape = (C, H, W)

    def process_without_transform(self, X):
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        if self.crop: 
            X = self.crop(image=X)['image']
        if self.preprocessor: 
            X = self.preprocessor.preprocess(X)
        # X.shape = (H, W, C)
        return X.transpose(2, 0, 1) # X.shape = (C, H, W)

    def get(self, i):
        try:
            if self.max_frames == 'pairs' and not self.test_mode:
                X = self.load_video_pairs(self.vidfiles[i])
            elif self.max_frames == 1 and not self.test_mode:
                X = self.load_video_frame(self.vidfiles[i])
            else:
                start_frame = 0 if self.test_mode else None
                num_frames = self.test_frames if self.test_mode else self.max_frames
                X = self.load_video_chunk(self.vidfiles[i], num_frames=num_frames, start=start_frame)
        except Exception as e:
            print(e)
            X = None
        return X

    def __getitem__(self, i):
        assert self.max_frames == 1, 'AugMix currently only supports training on single frames (max_frames=1)'

        X_orig = self.get(i)
        while type(X_orig) == type(None):
            i = np.random.choice(len(self.vidfiles))
            X_orig = self.get(i)

        if self.max_frames == 1 and not self.test_mode:
            X_orig = X_orig[0]

        y = self.labels[i]
        y = torch.tensor(y).float()

        if self.test_mode:
            X_orig = self.process_video(X_orig)   
            return torch.tensor(X_orig).float(), y        
        else:
            X_aug1 = self.process_image(X_orig.copy())
            X_aug2 = self.process_image(X_orig.copy())
            X_orig = self.process_without_transform(X_orig)

        # Torchify 
        X_orig = torch.tensor(X_orig).float()
        X_aug1 = torch.tensor(X_aug1).float()
        X_aug2 = torch.tensor(X_aug2).float()

        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return {'orig': X_orig, 'aug1': X_aug1, 'aug2': X_aug2}, y


class FaceReconDataset(FaceAugMixDataset):

    def pad_resize_crop(self, X):
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        if self.crop: 
            X = self.crop(image=X)['image']
        return X.transpose(2, 0, 1) # X.shape = (C, H, W)

    def __getitem__(self, i):
        assert self.max_frames == 1, 'Training with reconstruction loss regularization currently only supports training on single frames (max_frames=1)'

        X_orig = self.get(i)
        while type(X_orig) == type(None):
            i = np.random.choice(len(self.vidfiles))
            X_orig = self.get(i)

        if self.max_frames == 1 and not self.test_mode:
            X_orig = X_orig[0]

        y = self.labels[i]
        y = torch.tensor(y).float()

        if self.test_mode:
            X_orig = self.process_video(X_orig)   
            return torch.tensor(X_orig).float(), y        
        else:
            X = self.process_image(X_orig.copy())
            X_orig = self.pad_resize_crop(X_orig)

        # Torchify 
        X_orig = torch.tensor(X_orig).float()
        X = torch.tensor(X).float()

        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return {'image': X_orig/255., 'x': X}, y


class FaceMaskDataset(Dataset):

    def __init__(self, 
                 imgfiles,
                 maskfiles,
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 rescale=False,
                 test_mode=False):

        self.imgfiles = imgfiles
        self.maskfiles = maskfiles
        self.videos = [_.split('/')[-1].split('_')[0].replace('png','mp4') for _ in self.imgfiles]
        self.labels = labels
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.rescale = rescale
        self.test_mode = test_mode

    def __len__(self):
        return len(self.imgfiles)

    def process_image(self, X, y):
        if self.pad: X, y = self.pad(X), self.pad(y)
        if self.resize: 
            resized = self.resize(image=X, mask=y)
            X,y = resized['image'], resized['mask']
        if self.crop: 
            cropped = self.crop(image=X, mask=y)
            X,y = cropped['image'], cropped['mask']
        if self.transform: X = self.transform(image=X)['image']
        if self.preprocessor: X = self.preprocessor.preprocess(X)
        # X.shape = (H, W, C)
        return X.transpose(2,0,1), y.transpose(2,0,1) # X.shape = (C, H, W)

    def get(self, i):
        try:
            X = cv2.imread(self.imgfiles[i])
            if self.maskfiles[i].split('/')[-1] == 'empty_mask': 
                y = np.zeros(X.shape[:2]).astype('float32')
            else:
                y = np.load(self.maskfiles[i])
                if self.rescale: 
                    if np.max(y) > 0:
                        y /= np.max(y)
            y = np.expand_dims(y, axis=-1)

        except Exception as e:
            print('Error loading {}'.format(self.imgfiles[i]))
            print(e)
            X, y = None, None
        return X, y

    def __getitem__(self, i):
        X, y = self.get(i)
        while X is None:
            i = np.random.choice(len(self.imgfiles))
            X, y = self.get(i) 

        # Horizontal flip with probability 0.5
        if not self.test_mode:
            if np.random.binomial(1, 0.5):
                X, y = np.fliplr(X), np.fliplr(y)
        X, y = self.process_image(X, y)

        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, {'seg': y, 'cls': torch.tensor(self.labels[i])}


class FaceVideoTCNDataset(FaceVideoDataset):

    def __init__(self, 
                 vidfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 min_frames,
                 max_frames,
                 test_mode=False,
                 grayscale=False,
                 to_rgb=True):

        self.vidfiles = vidfiles
        self.labels = labels
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.test_mode = test_mode
        self.grayscale = grayscale
        self.badfiles = []
        self.to_rgb = to_rgb

        self.test_frames = max_frames
        self.videos = [_.split('/')[-1].split('_')[1] for _ in vidfiles]
        self.parts  = [int(_.split('/')[-1].split('_')[0]) for _ in vidfiles]

    def get(self, i):
        start_frame = 0 if self.test_mode else None
        num_frames = self.test_frames if self.test_mode else np.random.randint(self.min_frames, self.max_frames+1)
        X = self.load_video_chunk(self.vidfiles[i], num_frames=num_frames, start=start_frame)
        if len(X) < self.max_frames:
            # X.shape = (N, H, W, C)
            X = np.vstack((X, np.zeros((self.max_frames-len(X), X.shape[1], X.shape[2], X.shape[3]))))
        return X

    def __getitem__(self, i):
        # We assume that the chance of sampling 2 consecutive
        # bad files is too small to be relevant
        try:
            X = self.get(i) 
        except:
            self.bad_files.append(i)
            indices = list(range(len(self.vidfiles)))
            indices = list(set(indices) - set(self.badfiles))
            X = self.get(i)

        X = self.process_video(X)
        y = self.labels[i]
        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # TODO: To use .cuda in Dataset, need to set_start_method('spawn')
        # Thus may be better to move .cuda to Trainer class
        return X, y

        return X, y


class BalancedSampler(Sampler):

    def __init__(self,
        dataset):
        super().__init__(data_source=dataset)
        labels = np.asarray(dataset.labels)
        self.reals = np.where(labels == 0)[0]
        self.fakes = np.where(labels == 1)[0]
        # 1 epoch = # of real videos
        # FAKE = 1 for my labels
        self.length = len(self.reals)

    def __iter__(self):
        reals = np.random.permutation(self.reals)
        fakes = np.random.choice(self.fakes, self.length // 2, replace=False)
        indices = np.concatenate([reals,fakes])
        indices = np.random.permutation(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.length

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
        self.length = int(len(dataset) - np.sum(dataset.labels))

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