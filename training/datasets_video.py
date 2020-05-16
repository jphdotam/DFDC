import os
import cv2
import math
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

class RebalancedVideoDataset(Dataset):
    """This ensures that a every epoch classes are balanced without repetition, classes with more examples
    use their full range of videos.
    The requested id maps directly 1:1 with the smallest class, whereas larger classes get a video chosen
    with the corresponding range. E.g. if class1 has videos 1:100, and class 2 has videos 101:500, then
    requesting dataset[5] always results in the 5th video of class one, but dataset[105] will randomly yield
    one of 5 videos in range 125 - 130."""

    def __init__(self, video_dir, train_or_test, label_per_frame, transforms, framewise_transforms, i3d_norm, test_videos=None,
                 test_proportion=0.25, file_ext=".mp4", max_frames=64, bce_labels=False, alt_aug=False):
        self.video_dir = video_dir
        self.train_or_test = train_or_test
        self.label_per_frame = label_per_frame
        self.test_videos = test_videos
        self.test_proportion = test_proportion
        self.file_ext = file_ext
        self.i3d_norm = i3d_norm
        self.max_frames = max_frames
        self.transforms = transforms
        self.framewise_transforms = framewise_transforms
        self.bce_labels = bce_labels
        self.alt_aug = alt_aug
        self.classes = self.get_classes()
        self.n_classes = len(self.classes)
        self.videos_by_class = self.get_videos_by_class()
        self.n_by_class = self.get_n_by_class()
        self.n_smallest_class = self.get_n_smallest_class()
        self.n_balanced = self.get_n_balanced()
        self.n_unbalanced = self.get_n_unbalanced()

        self.c = self.n_classes  # FastAI

        self.summary()

    def get_classes(self):
        return os.listdir(self.video_dir)

    def get_videos_by_class(self):
        videos_by_class = {}
        for cls in self.classes:
            videos_for_class = []
            videopaths = glob(os.path.join(self.video_dir, cls, f"*{self.file_ext}"))
            for videopath in videopaths:
                is_test = self.train_or_test == 'test'

                video_chunk_id = os.path.basename(videopath).split('_', 1)[0]
                in_test = video_chunk_id in self.test_videos
                if is_test == in_test:
                    videos_for_class.append(videopath)

            videos_by_class[cls] = videos_for_class
        return videos_by_class

    def get_n_by_class(self):
        n_by_class = {}
        for cls, videos in self.videos_by_class.items():
            n_by_class[cls] = len(videos)
        return n_by_class

    def get_n_smallest_class(self):
        return min([len(videos) for videos in self.videos_by_class.values()])

    def get_n_balanced(self):
        return self.get_n_smallest_class() * self.n_classes

    def get_n_unbalanced(self):
        return sum([len(videos) for videos in self.videos_by_class.values()])

    def summary(self):
        print(f"{self.train_or_test.upper()}:"
              f"Loaded {self.n_unbalanced} samples across classes '{', '.join(self.classes)}'; effective sample size of {self.n_balanced}")

    def load_video(self, filename, every_n_frames, to_rgb, rescale=None):
        cap = cv2.VideoCapture(filename)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if rescale:
            out_video = np.zeros(
                (math.ceil(frameCount / every_n_frames), int(frameHeight * rescale), int(frameWidth * rescale), 3),
                np.dtype('uint8'))
        else:
            out_video = np.zeros((math.ceil(frameCount / every_n_frames), frameHeight, frameWidth, 3),
                                 np.dtype('uint8'))

        i_frame = 0
        ret = True

        while (i_frame * every_n_frames < frameCount and ret):
            cap.set(cv2.CAP_PROP_FRAME_COUNT, (i_frame * every_n_frames) - 1)
            ret, frame = cap.read()
            if rescale:
                frame = cv2.resize(frame, (0, 0), fx=rescale, fy=rescale)
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_video[i_frame] = frame
            i_frame += 1

        cap.release()
        return out_video

    def __len__(self):
        return self.n_balanced

    def __getitem__(self, idx):
        # Get the class
        id_cls = idx // self.n_smallest_class
        cls = self.classes[id_cls]

        # Get the video within the class
        n_cls = self.n_by_class[cls]
        id_in_cls_bal = idx % self.n_smallest_class
        id_in_cls_from = math.ceil((id_in_cls_bal / self.n_smallest_class) * n_cls)
        id_in_cls_to = max(id_in_cls_from,
                           math.floor((((
                                                    id_in_cls_bal + 1) / self.n_smallest_class) * n_cls) - 0.0001))  # Small epsilon to make sure whole numbers round down (so math.ceil != math.floor)
        id_in_cls = random.randint(id_in_cls_from, id_in_cls_to)

        # Load the video
        videoname = self.videos_by_class[cls][id_in_cls]
        video = self.load_video(filename=videoname, every_n_frames=1, to_rgb=True)

        if self.alt_aug:
            frame_incrementer = random.randint(1, 2)  # 1 for no aug, 2 for 1
        else:
            frame_incrementer = 1
        max_frames = self.max_frames * frame_incrementer

        if self.train_or_test == 'test':
            starting_frame = 0
        elif self.train_or_test == 'train':

            max_starting_frame = len(video) - max_frames
            try:
                starting_frame = random.randint(0, max_starting_frame)
            except ValueError:
                print(f"Problem reading {idx} -> {videoname}")
                raise Exception()
        else:
            raise ValueError(f"train_or_test must be 'train' or 'test', not {self.train_or_test}")

        video = video[starting_frame:starting_frame + max_frames:frame_incrementer]

        label_name = os.path.basename(os.path.dirname(videoname))
        label_id = self.classes.index(label_name)
        if self.label_per_frame:
            label_id = label_id * len(video)  # Label for each frame

        if self.transforms:
            if self.framewise_transforms:
                seed = random.randint(0, 99999)
                video_aug = []
                for frame in video:
                    random.seed(seed)
                    video_aug.append(self.transforms(image=frame)['image'])
                video_aug = np.array(video_aug)
                video = video_aug
            else:
                video = self.transforms(video)

        if type(video) == list:  # Transforms may return a list
            video = np.array(video)

        x = torch.from_numpy(video.transpose([3, 0, 1, 2])).float()

        if self.i3d_norm:
            x = (x / 255.) * 2 - 1

        y = torch.tensor(label_id, dtype=torch.float)
        if self.bce_labels:  # BCEloss expects batch*size * 1 shape, not just batch_size
            y = y.unsqueeze(-1)
        else:
            y = y.long()

        return x, y
