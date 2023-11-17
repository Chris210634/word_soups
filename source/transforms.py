import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import PIL
import numpy as np
import random
from copy import deepcopy

def get_train_transform(opt='simple'):
    if opt=='complex':
        return transforms.Compose( 
            [
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.3, 0.3]),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )
    elif opt=='randaugment':
        return transforms.Compose( 
            [
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )
    else:
        assert opt == 'simple'
    return transforms.Compose( 
        [
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )
# For DomainBed benchmark, add following two transforms:
#         transforms.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.3, 0.3]),
#         transforms.RandomGrayscale(p=0.1),

def get_test_transform():
    return transforms.Compose(
        [transforms.Resize(size=224, max_size=None, antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
    )

# https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    y is one hot.
    '''
    cast_dtype = x.dtype
    lam = torch.tensor(np.random.beta(alpha, alpha, size=len(y))).to(y.device)

    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(y.device)

    mixed_x = lam[:,None,None,None] * x + (1. - lam[:,None,None,None]) * x[index]
    mixed_y = lam[:,None] * y + (1. - lam[:,None]) * y[index]
    return mixed_x.to(cast_dtype), mixed_y, index, lam