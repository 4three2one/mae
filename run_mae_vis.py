import datetime
import os

from tensorboard.backend.event_processing import event_accumulator
import torch
from torchvision import datasets
from torchvision.transforms import transforms
import timm.optim.optim_factory as optim_factory
import models_mae
import argparse
from torch.utils.data import RandomSampler
from util import misc


# coding:utf-8
import os, torchvision
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #自己设置的
    std = [0.229,0.224,0.225]  #自己设置的
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path, size,padding,log_writer):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像保存的路径
        size (int)  --  一行有size张图,最好是2的倍数
    """
    im_grid = torchvision.utils.make_grid(im, size,padding=padding) #将batchsize的图合成一张图
    im_numpy = tensor2im(im_grid) #转成numpy类型并反归一化
    im_array = Image.fromarray(im_numpy)
    #im_array.save(path)
    if log_writer is not None:
        log_writer.add_img("test",im_array)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/media/xjw/ssk_data/plant/PlantVillage_full', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')  #num_samples

    parser.add_argument('--num_samples', default=3, type=int)

    return parser
def reconstruct(x, image_size, patch_size):
    """reconstrcunt [batch_size, num_patches, embedding] -> [batch_size, channels, h, w]"""
    B, N, _ = x.shape  # batch_size, num_patches, dim

    p1, p2 = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
    x = x.reshape([B, p1, p2, -1, patch_size[0], patch_size[1]]).transpose([0, 3, 1, 4, 2, 5]).reshape([B, -1, image_size[0], image_size[1]])
    return x

def main(args):
    now=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_writer = SummaryWriter(log_dir=args.log_dir,filename_suffix=now)
    device = torch.device(args.device)
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model
    checkpoint = torch.load("output_dir/checkpoint-model-420.pth", map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    model.eval()
    with torch.no_grad():
        for epcho in range(2):
            sampler = RandomSampler(dataset_val, replacement=True, num_samples=args.num_samples)
            # val_img = torch.stack([dataset_val[i][0] ford i in range(len(dataset_val))]).to(device)
            val_img = torch.stack([dataset_val[i][0] for i in sampler]).to(device)
            _, pred, mask = model(val_img, mask_ratio=args.mask_ratio)
            pred_img = model.unpatchify(pred)
            mask_img = model.unpatchify(model.patchify(val_img) * (1 - mask).unsqueeze(2))
            compare_img = torch.cat([val_img, mask_img, pred_img], dim=0)
            im_grid = torchvision.utils.make_grid(compare_img, args.num_samples, padding=2)  # 将batchsize的图合成一张图
            im_numpy = tensor2im(im_grid)  # 转成numpy类型并反归一化
            log_writer.add_image(f"test", im_numpy.transpose((2,0,1)),global_step=epcho)
            log_writer.flush()
        pass
    log_writer.close()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)