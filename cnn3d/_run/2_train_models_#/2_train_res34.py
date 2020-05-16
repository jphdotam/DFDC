import os
import pickle
from glob import glob

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

from training.resnet import resnet34
from training.datasets_video import RebalancedVideoDataset
from training.utils import Am

"""This is a fairly standard training script which uses:
1) 3D CNNs and their weights taken from https://github.com/kenshohara/video-classification-3d-cnn-pytorch/
- MIT license
- Models from https://github.com/kenshohara/video-classification-3d-cnn-pytorch/blob/master/models/resnet.py
- Weights from https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing (link on github)

2) Note bizarrely for these transforms we didn't normalize: ie we feed in the 0-255 images. This seemed to be the
only way to make it train?!

For all the 3D models I actually use a categorical crossentropy rather than a binary crossentropy because the I3D model
works by having an output convolutional layer per class.
"""

FACE_MP4_DIR = "E:/DFDC/faces2"
BATCH_SIZE = 32
N_WORKERS = 8
DEVICE = "cuda"
EPOCHS = 20

if __name__ == "__main__":

    train_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.25),
        A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.9, 1.1)),
        A.HorizontalFlip()])
    test_transforms = A.Compose([A.CenterCrop(height=224, width=224)])

    test_roots = ['45', '46', '47', '48', '49']

    dataset_train = RebalancedVideoDataset(FACE_MP4_DIR, "train", label_per_frame=False, test_videos=test_roots,
                                           transforms=train_transforms, framewise_transforms=True, i3d_norm=False,
                                           max_frames=32)
    dataset_test = RebalancedVideoDataset(FACE_MP4_DIR, "test", label_per_frame=False, test_videos=test_roots,
                                          transforms=test_transforms, framewise_transforms=True, i3d_norm=False,
                                          max_frames=32)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=N_WORKERS, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS,
                                 pin_memory=True)

    model = resnet34(num_classes=2, shortcut_type='A', sample_size=224, sample_duration=32)  # , last_fc=True)
    model.load_state_dict(
        {k.split('.',1)[1]:v for k,v in torch.load('../../data/external_models/resnet-34-kinetics.pth')['state_dict'].items() if 'fc.' not in k},
        strict=False)
    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    lr = 1e-3

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr/10,
                                                    steps_per_epoch=len(dataloader_train),
                                                    epochs=10)

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {EPOCHS}")
        # TRAIN
        model.train()
        losses, accuracy = Am(), Am()
        for i, (x, y_true) in enumerate(dataloader_train):

            x = x.to(DEVICE)
            t = x.size(2)
            y_true = y_true.to(DEVICE)

            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            losses.update(loss.data, x.size(0))

            # Accuracy
            ps = torch.exp(y_pred)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_true.view(*top_class.shape)
            accuracy.update(torch.mean(equals.type(torch.FloatTensor)).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                print(f'TRAIN\t[{i:03d}/{len(dataloader_train):03d}]\t\t'
                      f'LOSS: {losses.running_average:.4f}\t\t'
                      f'ACCU: {accuracy.running_average:.4f}')

        print(f'TRAIN\tComplete!\t\t'
              f'LOSS: {losses.avg:.4f}\t\t'
              f'ACCU: {accuracy.avg:.4f}')

        # TEST
        model.eval()
        losses, accuracy, reals_correct, fakes_correct = Am(), Am(), Am(), Am()
        for i, (x, y_true) in enumerate(dataloader_test):

            with torch.no_grad():

                x = x.to(DEVICE)
                t = x.size(2)
                y_true = y_true.to(DEVICE)

                y_pred = model(x)

                loss = criterion(y_pred, y_true)
                losses.update(loss.data, x.size(0))

                # Accuracy
                ps = torch.exp(y_pred)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y_true.view(*top_class.shape)
                accuracy.update(torch.mean(equals.type(torch.FloatTensor)).item())

                for cls, eq in zip(top_class, equals):  # Iterate over batch
                    if cls[0] == 0:  # Fake
                        fakes_correct.update(bool(eq[0]))
                    elif cls[0] == 1:
                        reals_correct.update(bool(eq[0]))
                    else:
                        print(f"Unknown class!")

                if i % 10 == 0:
                    print(f'TEST\t[{i:03d}/{len(dataloader_test):03d}]\t\t'
                          f'LOSS: {losses.running_average:.4f}\t\t'
                          f'ACCU: {accuracy.running_average:.4f}\t\t'
                          f'REAL: {reals_correct.running_average:.4f}\t'
                          f'FAKE: {fakes_correct.running_average:.4f}')

        print(f'TEST\tComplete!\t\t'
              f'LOSS: {losses.avg:.4f}\t\t'
              f'ACCU: {accuracy.avg:.4f}\t\t'
              f'REAL: {reals_correct.avg:.4f}\t'
              f'FAKE: {fakes_correct.avg:.4f}')

        torch.save(model.state_dict(), f"./saved_models/res34_1cy_lilaug_nonorm_e{epoch}_l{losses.avg:.4f}.model")
