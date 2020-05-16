import numpy as np
import cv2

import albumentations as albu

from .gridmask import GridMask


def pad_to_ratio(array, ratio):
    # Default is ratio=1 aka pad to create square image
    ratio = float(ratio)
    # Given ratio, what should the height be given the width? 
    h, w = array.shape[:2]
    desired_h = int(w * ratio)
    # If the height should be greater than it is, then pad height
    if desired_h > h: 
        hdiff = int(desired_h - h) ; hdiff = int(hdiff / 2)
        pad_list = [(hdiff, desired_h-h-hdiff), (0,0), (0,0)]
    # If height should be smaller than it is, then pad width
    elif desired_h < h: 
        desired_w = int(h / ratio)
        wdiff = int(desired_w - w) ; wdiff = int(wdiff / 2)
        pad_list = [(0,0), (wdiff, desired_w-w-wdiff), (0,0)]
    elif desired_h == h: 
        return array 
    return np.pad(array, pad_list, 'constant', constant_values=np.min(array))

def resize(x, y=None):
    if y is None: y = x
    return albu.Compose([
        albu.Resize(x, y, always_apply=True, interpolation=cv2.INTER_CUBIC, p=1)
        ], p=1)

def crop(x, y=None, test_mode=False):
    if y is None: y = x
    if test_mode:
        return albu.Compose([
            albu.CenterCrop(x, y, always_apply=True, p=1)
            ], p=1, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})
    else:
        return albu.Compose([
            albu.RandomCrop(x, y, always_apply=True, p=1)
            ], p=1, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})

def vanilla_transform(p):
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(rotate_limit=30, 
                         scale_limit=0.15,  
                         border_mode=cv2.BORDER_CONSTANT, 
                         value=[0,0,0],
                         p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.JpegCompression(quality_lower=85, p=0.5),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})


def gentle_transform(p):
    return albu.Compose([
        # p=0.5
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        # p=0.2
        albu.ShiftScaleRotate(rotate_limit=30, 
                         scale_limit=0.15,  
                         border_mode=cv2.BORDER_CONSTANT, 
                         value=[0,0,0],
                         p=0.2),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.2),
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})


def flips_only(p):
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})

def color_transform(p):
    return albu.Compose([
        albu.OneOf(
            [   
                albu.IAASharpen(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomContrast(p=1),
                albu.RandomGamma(p=1)
            ],
        p=1)], p=p)

def feather_transform(p):
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(p=0.25, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomBrightnessContrast(p=0.25),
        albu.GaussianBlur(p=0.25)
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,65)})

def grid_mask(p):
    return albu.Compose([
        GridMask(num_grid=(3,7), rotate=(-90, 90), mode=0, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5)
    ], p=p)

class Preprocessor(object):
    '''
    Object to deal with preprocessing.
    Easier than defining a function.
    '''
    def __init__(self, image_range, input_range, mean, sdev):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean 
        self.sdev = sdev

    def preprocess(self, img): 
        ''' 
        Preprocess an input image. 
        '''
        # Assume image is RGB 
        # Unconvinced that RGB<>BGR matters for transfer learning ...
        img = img[..., ::-1].astype('float32')

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])

        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])

        image_range = image_max - image_min
        model_range = model_max - model_min 

        img = (((img - image_min) * model_range) / image_range) + model_min 
        if img.shape[-1] == 3:
            img[..., 0] -= self.mean[0] 
            img[..., 1] -= self.mean[1] 
            img[..., 2] -= self.mean[2] 
            img[..., 0] /= self.sdev[0] 
            img[..., 1] /= self.sdev[1] 
            img[..., 2] /= self.sdev[2] 
        elif img.shape[-1] == 1:
            img -= np.mean(self.mean)
            img /= np.mean(self.sdev)

        return img
