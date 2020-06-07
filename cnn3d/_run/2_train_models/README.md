Here we will use the dataset we've created to train 7 3D CNNs, which I have organised in the order they are listed
in our inference notebook at https://www.kaggle.com/vaillant/dfdc-3d-2d-inc-cutmix-with-3d-model-fix

These 7 trained models are available in "/data/saved_models/"

In brackets I show their 'type' which is used to prepare data appropriately in this inference notebook:

1) Inception 3D trained on 224*224 images (i3d), from https://github.com/piergiaj/pytorch-i3d (Apache license; I call this model I3D, or sometimes J3D as I (James) made some minor tweaks)
2) Resnet3D-34 trained on 224*224 images (res34) , from https://github.com/kenshohara/video-classification-3d-cnn-pytorch (MIT license)
3) MC3 trained on 112*112 images (mc3_112), from https://github.com/pytorch/vision/blob/master/torchvision/models/video
4) MC3 trained on 224*224 images (mc3_224), from https://github.com/pytorch/vision/blob/master/torchvision/models/video
5) R2+1D trained on 112*112 images (r2p1_112), from https://github.com/pytorch/vision/blob/master/torchvision/models/video
6) Inception 3D trained on 224*224 images with cutmix (i3d), again from https://github.com/piergiaj/pytorch-i3d
7) R2+1D trained on 112*112 images with cutmix (r2p1_112), again from https://github.com/pytorch/vision/blob/master/torchvision/models/video

These can be quite time-consuming to train as a batch of 26 videos might be 26 * 64frames * 3 * 224 * 224
I used two computers, one has a single nVidia RTX Titan, and a second has 2 of these cards.
I have left the code as it was, so Kaggle can see which models I trained on single vs dual GPUs.

Somewhat confusingly, these networks use different normalisation parameters. These are evident in our kaggle testing
inference notebook.

This is because
1) The I3D model we transfer learned from used its own normalisation method (I call i3d_norm) which is simply:
    x = (x / 255.) * 2 - 1
2) The models from the torchvision repository (MC3, R2+1D) use ImageNet normalisation.
3) The Resnet3D-34 model only seemed to train with non-normalized data.

All videos are trained from a RebalancedVideoDataset which is defined in /training/datasets_video and which ensures
equal balance of real and fake videos are used for training. It randomly subsampled the fake videos which are more
numerous, whilst feeding in each 1 of the rarer real videos per epoch.
