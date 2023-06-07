from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
class IP102(Dataset):
    def __init__(self,txt_path,transform=None):
        self.transform=transform
        with open(txt_path,"r") as f:
            lines=f.readlines()
            self.img_list=[ os.path.join(os.path.dirname(txt_path),"images",line.split()[0]) for line in lines]
            self.label_list=[ line.split()[1] for line in lines]
    def __getitem__(self, index):
        img_path=self.img_list[index]
        label=self.label_list[index]
        img=Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)
    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class DeepWeeds(Dataset):
    def __init__(self,csv_path,transform=None):
        self.transform=transform
        import pandas as pd
        data = pd.read_csv(csv_path)
        self.img_list = [os.path.join(os.path.dirname(csv_path), "images", name) for name in data["Filename"].tolist()]
        self.label_list = [label for label in data["Label"].tolist()]
    def __getitem__(self, index):
        img_path=self.img_list[index]
        label=self.label_list[index]
        img=Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)
    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels