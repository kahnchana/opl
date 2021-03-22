""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
from typing import Tuple, Any
from PIL import Image
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_network(args):
    """ return given network
    """
    if args.net == 'resnet56':
        from models.resnet_cifar import resnet56_cifar100
        net = resnet56_cifar100(pretrained=False)
        if args.rbf:
            from losses import RBFLogits
            net.output = RBFLogits(feature_dim=64, class_num=100, scale=4, gamma=2)
    elif args.net == 'resnet110':
        from models.resnet_cifar import resnet110_cifar100
        net = resnet110_cifar100(pretrained=True)
        if args.rbf:
            from losses import RBFLogits
            net.output = RBFLogits(feature_dim=64, class_num=100, scale=4, gamma=2)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


class AugCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, mean, std, distortion=1.0, *args, **kwargs):
        super(AugCIFAR100, self).__init__(*args, **kwargs)

        def get_color_distortion(s=distortion):
            # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
            return color_distort

        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            get_color_distortion(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform = transform_train

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, use_aug=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        use_aug:    use contrastive data augmentation
    Returns: train_data_loader:torch dataloader object
    """

    if use_aug:
        cifar100_training = AugCIFAR100(root='./data', train=True, download=True, mean=mean, std=std, distortion=0.8)
    else:

        def cifar_train_transform(mean_rgb=(0.4914, 0.4822, 0.4465), std_rgb=(0.2023, 0.1994, 0.2010),
                                  jitter_param=0.4):
            return transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_rgb, std=std_rgb)
            ])
        transform_train = cifar_train_transform()
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=transform_train)

    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, imbalance=False, use_norm=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = [transforms.ToTensor(), ]
    if use_norm:
        transform_test.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(transform_test)

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = [x for x in os.listdir(weights_folder) if x.endswith(".pth")]
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = [x for x in os.listdir(weights_folder) if x.endswith(".pth")]
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def get_train_valid_loader(batch_size,
                           random_seed=0,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=2,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    cifar100_train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar100_train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    normalize = transforms.Normalize(
        mean=cifar100_train_mean,
        std=cifar100_train_std,
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    def cifar_train_transform(mean_rgb=(0.4914, 0.4822, 0.4465), std_rgb=(0.2023, 0.1994, 0.2010),
                              jitter_param=0.4):
        return transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])

    train_transform = cifar_train_transform(cifar100_train_mean, cifar100_train_std)

    # load the dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(numpy.floor(valid_size * num_train))

    if shuffle:
        numpy.random.seed(random_seed)
        numpy.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader
