import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rand_bbox_2d(size, lam):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(np.int)
    cut_h = (H * cut_rat).astype(np.int)
    # uniform
    cx = np.random.randint(0, W, B)
    cy = np.random.randint(0, H, B)
    #
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def rand_bbox_3d(size, lam):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    T = size[2]
    W = size[3]
    H = size[4]
    cut_rat = (1. - lam) ** (1/3.)
    cut_t = (T * cut_rat).astype(np.int)
    cut_w = (W * cut_rat).astype(np.int)
    cut_h = (H * cut_rat).astype(np.int)
    # uniform
    ct = np.random.randint(0, T, B)
    cx = np.random.randint(0, W, B)
    cy = np.random.randint(0, H, B)
    #
    bbt1 = np.clip(ct - cut_t // 2, 0, T)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbt2 = np.clip(ct + cut_t // 2, 0, T)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbt1, bbx1, bby1, bbt2, bbx2, bby2

def cutmix_apply(batch, alpha):
    batch_size = batch.size(0)
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max((lam, 1.-lam), axis=0)
    index = torch.randperm(batch_size)
    if batch.ndim == 5:
        # 3D
        t1, x1, y1, t2, x2, y2 = rand_bbox_3d(batch.size(), lam)
        for b in range(batch.size(0)):
            batch[b, :, t1[b]:t2[b], x1[b]:x2[b], y1[b]:y2[b]] = batch[index[b], :, t1[b]:t2[b], x1[b]:x2[b], y1[b]:y2[b]]
        lam = 1. - ((t2 - t1) * (x2 - x1) * (y2 - y1) / float((batch.size()[-1] * batch.size()[-2] * batch.size()[-3])))

    elif batch.ndim == 4:
        # 2D
        x1, y1, x2, y2 = rand_bbox_2d(batch.size(), lam)
        for b in range(batch.size(0)):
            batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]
        lam = 1. - ((x2 - x1) * (y2 - y1) / float((batch.size()[-1] * batch.size()[-2])))

    return batch, index, lam


def cutmix_double_apply(batch, labels, alpha):
    batch_size = batch.size(0)
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max((lam, 1.-lam), axis=0)
    index = torch.randperm(batch_size)
    # 2D - does not support 3D right now
    x1, y1, x2, y2 = rand_bbox_2d(batch.size(), lam)
    for b in range(batch.size(0)):
        batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]
        labels['seg'][b, :, x1[b]:x2[b], y1[b]:y2[b]] = labels['seg'][index[b], :, x1[b]:x2[b], y1[b]:y2[b]]
    lam = 1. - ((x2 - x1) * (y2 - y1) / float((batch.size()[-1] * batch.size()[-2])))


class MixupBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        if type(y_true) == dict:
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            mix_loss1 = F.cross_entropy(y_pred, y_true1, reduction='none')
            mix_loss2 = F.cross_entropy(y_pred, y_true2, reduction='none')
            return (lam * mix_loss1 + (1. - lam) * mix_loss2).mean()
        else:
            # Validation
            return F.binary_cross_entropy_with_logits(y_pred, y_true)

