# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.my_dataset import *
from utils.utils import read_split_data, train_one_epoch, evaluate

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_mine(args):
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    if args.dataset_name.startswith("PlantDoc"):
        root_train = os.path.join(args.data_path, 'train')
        train_dataset = datasets.ImageFolder(root_train, transform=transform_train)
        root_val = os.path.join(args.data_path, 'val')
        val_dataset = datasets.ImageFolder(root_val, transform=transform_train)

    if args.dataset_name.startswith("plantv"):
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=transform_train)

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=transform_val)
    if args.dataset_name.startswith("IP102"):
        train_dataset = IP102(txt_path=os.path.join(args.data_path, "train_new.txt"), transform=transform_train)
        val_dataset = IP102(txt_path=os.path.join(args.data_path, "test_new.txt"), transform=transform_val)

    if args.dataset_name.startswith("deepweeds"):
        train_dataset = DeepWeeds(csv_path=os.path.join(args.data_path, "train.csv"), transform=transform_train)
        val_dataset = DeepWeeds(csv_path=os.path.join(args.data_path, "test.csv"), transform=transform_val)
    return train_dataset,val_dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
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
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
