# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import pandas as pd
import numpy as np

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


class sle_dataset(Dataset):
    def __init__(self, mode, args):
        """ data_path, dataset_name, task_type, transform, mode"""
        # assert args.data_path == 'docker_prepare/data_input_example/external_test.csv'
        self.df = pd.read_csv(args.data_path).copy()

        self.img_prefix = ''
        
        
        self.label_col = args.task_type+'_label'
        self.transform = build_transform(mode, args)
    
    def __getitem__(self, item):
        img_path = os.path.join(self.img_prefix, self.df.loc[item, 'image_path'])

        image = Image.open(img_path)
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        
        label = np.array(self.df.loc[item,self.label_col])
        label = torch.tensor(label, dtype=torch.long)

        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.df)

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
