import numpy as np
import torch


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

    return batch, labels, index, lam


if __name__ == 'main':

    import torch, numpy as np
    import matplotlib.pyplot as plt
    import cv2, glob

    imgs = glob.glob('*.png')
    images = np.asarray([cv2.imread(i) for i in imgs])
    #images = np.asarray([cv2.resize(i, (224,224)) for i in images])
    thresholded = [(images[i,...,0] < np.mean(images[i,...,0])*0.75) for i in range(len(images))]
    # for t in thresholded:
    #     plt.imshow(t); plt.show()
    batch = images.transpose(0,3,1,2)
    batch = torch.from_numpy(batch).float()

    lam = np.random.beta(1.0,1.0,len(batch))
    x1, y1, x2, y2 = rand_bbox_target(batch, batch.size(), lam)
    #x1, y1, x2, y2 = rand_bbox_vector(batch.size(), lam)
    index = torch.randperm(batch.size(0))

    for b in range(batch.size(0)):
        batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = 255 - batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]

    for bind, b in enumerate(batch):
        print(lam[bind])
        plt.subplot(1,2,1)
        plt.imshow(b.numpy().transpose(1,2,0).astype('uint8')) 
        plt.subplot(1,2,2)
        plt.imshow(images[bind])
        plt.show()



