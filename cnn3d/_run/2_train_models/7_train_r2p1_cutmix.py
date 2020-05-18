import os
import pickle
from glob import glob

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18

import albumentations as A
from albumentations.pytorch import ToTensor

from cnn3d.training.datasets_video import RebalancedVideoDataset
from cnn3d.training.utils import Am
from cnn3d.training.cutmix import cutmix_apply, MixupBCELoss

FACE_MP4_DIR = "/home/james/Data/dfdc/faces2"
BATCH_SIZE = 36
N_WORKERS = 8
DEVICE = "cuda"
EPOCHS = 10

CUTMIX_ALPHA = 1

train_transforms = A.Compose([
    A.GaussianBlur(p=0.1),
    A.GaussNoise(var_limit=5 / 255, p=0.1),
    A.RandomBrightnessContrast(p=0.25),
    A.RandomResizedCrop(height=112, width=112, scale=(0.5, 1), ratio=(0.9, 1.1)),
    A.Normalize(),
    A.HorizontalFlip()])
test_transforms = A.Compose([A.Resize(height=128, width=128),
                             A.CenterCrop(height=112, width=112),
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

model = r2plus1d_18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=2)
model = nn.DataParallel(model)

model = model.to(DEVICE)

criterion_train = MixupBCELoss()
criterion_test = torch.nn.CrossEntropyLoss()

lr = 1e-3

optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=lr,
                                                steps_per_epoch=len(dataloader_train),
                                                epochs=EPOCHS)

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch + 1} of {EPOCHS}")
    # TRAIN
    model.train()
    losses, accuracy = Am(), Am()
    for i, (x, y_true) in enumerate(dataloader_train):

        x = x.to(DEVICE)
        t = x.size(2)
        y_true = y_true.to(DEVICE)

        x, index, lam = cutmix_apply(x, CUTMIX_ALPHA)
        labels_dict = {'y_true1': y_true, 'y_true2': y_true[index], 'lam': torch.from_numpy(lam).cuda()}

        y_pred = model(x)

        loss = criterion_train(y_pred, labels_dict)
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

            loss = criterion_test(y_pred, y_true)
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

    torch.save(model.module.state_dict(), f"./saved_models/r2plus1dcutmix_112_e{epoch}_l{losses.avg:.4f}.model")
