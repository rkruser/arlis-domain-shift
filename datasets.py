#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Datasets: MNIST, Cifar10, kaggle planes, etc.
"""

### Imports

import torchvision
import torchvision.transforms as transforms
import torch

import socket

### Datasets
hostname = socket.gethostname()

if 'LV426' in hostname:
    root_dir = '/mnt/linuxshared/phd-research/data/standard_datasets/'
elif 'jacobswks20' in hostname:
    root_dir = '/scratch0/datasets'
else:
    root_dir = './'

'''
Labels = mnist, cifar10, ...

'''

def stack_channels(x):
    return torch.cat((x,x,x),dim=0)

def noise(x):
    return torch.clamp(x+(1.0/256)*torch.randn(x.size()), 0.0, 1.0)

def get_dataset(dataset='mnist'):

    if dataset == 'mnist':
        im_transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                stack_channels, # So it is comparable with cifar
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.MNIST(root_dir, train=True, transform=im_transform, target_transform=None, download=True)
    elif dataset == 'cifar10':
        im_transform = transforms.Compose([
                transforms.ToTensor(),
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.CIFAR10(root_dir, train=True, transform=im_transform, target_transform=None, download=True)
    ## Add other datasets


    return dataset


