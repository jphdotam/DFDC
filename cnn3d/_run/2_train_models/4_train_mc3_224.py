import os
import pickle
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from training.datasets_video import RebalancedVideoDataset
from training.utils import Am

"""
In this model we transfer learn, from the saved MC3 model trained on 112*112 images, to 224*224 images. This requires a
batch size to be as small as 10, and is time-consuming. 

For training the prior model, see "3_train_mc3_112.py"
"""

FACE_MP4_DIR = "E:/DFDC/faces2"
BATCH_SIZE = 10
N_WORKERS = 8
DEVICE = "cuda"
EPOCHS = 20

if __name__ == "__main__":
    train_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.25),
        A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.9, 1.1)),
        A.Normalize(),
        A.HorizontalFlip()])
    test_transforms = A.Compose([A.CenterCrop(height=224, width=224),
                                 A.Normalize()
                                 ])

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

    model = mc3_18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=2)
    model.load_state_dict(torch.load("../../data/saved_models/mc3_18_112_1cy_lilaug_nonorm_e9_l0.1905.model"))

    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()

    lr = 5e-5

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr / 10,
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

        torch.save(model.state_dict(), f"./saved_models/mc3_18_112t224_1cy_lilaug_nonorm_e{epoch}_l{losses.avg:.4f}.model")
