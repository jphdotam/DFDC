The entry points in this file are separated into those regarding the 3D CNN training system, and the 2D CNN training system.

---

## 3D CNN training entry points

### Entry points for preparing the data

These 4 entry points are within `cnn3d\_run\1_export_mp4s` and should be run sequentially

This is the first step of the system.

The idea is these scripts will take us from the original deepfake dataset to ~250,000 MP4 files which are closely cropped to individual people's faces, tracking a bounding box temporally through time.

Examples of such videos are seen in `../data/face_videos_by_real_fake/`

There are 4 steps with corresponding scripts required to generate this dataset

1) `1_export_rois.py` - This analyses every 10th frame of the original DFDC videos and stores bounding box coordinates for
face face present. Only the real videos (non-faked) are analysed, to save time. Fake videos use the ROIs from their
corresponding real partner.

2) `2_export_mp4s.py` - This uses the coordinates produced by the previous script to export MP4s for every unique face
present in a video. These MP4s are cropped down so they are as small as possible, whilst containing the full face even
if it moves across the screen. (We actually analyse each 300 frame videos as 2 smaller 150 frame videos).

3) `3_invalidate_short_videos.py` - MP4s created in the previous step are checked to ensure they have <= 64 frames. If not,
they are renamed to have a different file extension and are not used for training.

4) `4_split_videos_into_real_and_fake.py` - Self-explanatory, we move videos from being shorted by the chunk number they
were in in the original DFDC dataset, to the final training folder (REAL or FAKE).

### Entry points for training the 3D CNNs

Here we will use the dataset we've created to train 7 3D CNNs, which I have organised in the order they are listed
in our inference notebook at https://www.kaggle.com/vaillant/dfdc-3d-2d-inc-cutmix-with-3d-model-fix

These 7 trained models are available in "/data/saved_models/"

In brackets I show their 'type' which is used to prepare data appropriately in this inference notebook:

1) `1_train_model_i3d.py` - Inception 3D trained on 224*224 images (i3d), from https://github.com/piergiaj/pytorch-i3d (Apache license; I call this model I3D, or sometimes J3D as I (James) made some minor tweaks)

2) `2_train_res34.py` - Resnet3D-34 trained on 224*224 images (res34) , from https://github.com/kenshohara/video-classification-3d-cnn-pytorch (MIT license)

3) `3_train_mc3_112.py` - MC3 trained on 112*112 images (mc3_112), from https://github.com/pytorch/vision/blob/master/torchvision/models/video

4) `4_train_mc3_224.py` - MC3 trained on 224*224 images (mc3_224), from https://github.com/pytorch/vision/blob/master/torchvision/models/video

5) `5_train_r2p1d.py` - R2+1D trained on 112*112 images (r2p1_112), from https://github.com/pytorch/vision/blob/master/torchvision/models/video

6) `6_train_i3dcutmix.py` - Inception 3D trained on 224*224 images with cutmix (i3d), again from https://github.com/piergiaj/pytorch-i3d

7) `7_train_r2p1_cutmix.py` - R2+1D trained on 112*112 images with cutmix (r2p1_112), again from https://github.com/pytorch/vision/blob/master/torchvision/models/video

These can be quite time-consuming to train as a batch of 26 videos might be 26 * 64frames * 3 * 224 * 224
I used two computers, one has a single nVidia RTX Titan, and a second has 2 of these cards.
I have left the code as it was, so Kaggle can see which models I trained on single vs dual GPUs.

---

## 2D CNN training entrypoints

Ian's stuff
