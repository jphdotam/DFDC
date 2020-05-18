import os
import pickle
from glob import glob

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

from cnn3d.training.i3d import InceptionI3d
from cnn3d.training.datasets_video import RebalancedVideoDataset
from cnn3d.training.utils import Am
from cnn3d.training.cutmix import cutmix_apply, MixupBCELoss

"""
This script is more or less the same as 1_train_model_i3d.py, except we use cutmix augmentation.
I trained this model on my double-GPU set up, hence the larger batch size.
"""

FACE_MP4_DIR = "/home/james/Data/dfdc/faces2"
BATCH_SIZE = 26
N_WORKERS = 8
DEVICE = "cuda"
EPOCHS = 10

CUTMIX_ALPHA = 1

train_transforms = A.Compose([
    A.RandomBrightnessContrast(p=0.25),
    A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.9, 1.1)),
    A.HorizontalFlip()])
test_transforms = A.Compose([A.CenterCrop(height=224, width=224)])

test_roots = ['45', '46', '47', '48', '49']

dataset_train = RebalancedVideoDataset(FACE_MP4_DIR, "train", label_per_frame=False, test_videos=test_roots,
                                       transforms=train_transforms, framewise_transforms=True, i3d_norm=True)
dataset_test = RebalancedVideoDataset(FACE_MP4_DIR, "test", label_per_frame=False, test_videos=test_roots,
                                      transforms=test_transforms, framewise_transforms=True, i3d_norm=True)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=N_WORKERS, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS,
                             pin_memory=True)

model = InceptionI3d(157, in_channels=3, output_method='avg_pool')
model.load_state_dict(torch.load('../../data/external_models/i3d_rgb_charades.pt'))
model.replace_logits(2)
#model = nn.DataParallel(model)

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

    torch.save(model.module.state_dict(), f"./saved_models/i3dcutmix_e{epoch}_l{losses.avg:.4f}.model")
