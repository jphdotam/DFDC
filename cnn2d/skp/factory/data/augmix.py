import albumentations as albu
import numpy as np
import cv2


def cement(aug):
  aug.always_apply = True
  aug.p = 1
  return aug


def augmentations():
  return [cement(albu.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT)),
          cement(albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT)),
          cement(albu.ShiftScaleRotate(scale_limit=0.0625, shift_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT)),
          cement(albu.RandomBrightness(limit=0.3)),
          cement(albu.RandomContrast(limit=0.3)),
          cement(albu.GaussianBlur(blur_limit=(3,7))),
          cement(albu.GaussNoise(var_limit=(0, 30))),
          cement(albu.ChannelShuffle()),
          cement(albu.ToGray())]


def augment_and_mix(image, aug_list, width=3, depth=-1, alpha=1.):
  """
  Perform AugMix augmentations and compute mixture.
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image).astype('float32')
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(1, 4)
    ops = np.random.choice(aug_list, depth, replace=False)
    for op in ops:
      image_aug = op(image=image_aug)['image']
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * image_aug.astype('float32')

  mixed = (1 - m) * image + m * mix
  return {'image': mixed}


if __name__ == '__main__':
  from factory.data.augmix import *
  import matplotlib.pyplot as plt
  aug_list = augmentations()
  img = cv2.imread('/users/ipan/downloads/fake_face.png')
  img = img[...,::-1]
  img_aug = augment_and_mix(img, aug_list)
  plt.subplot(1,2,1); plt.imshow(img)
  plt.subplot(1,2,2); plt.imshow(img_aug.astype('uint8'))
  plt.show()




