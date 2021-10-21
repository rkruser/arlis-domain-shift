#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Datasets: MNIST, Cifar10, kaggle planes, etc.
"""

### Imports

import torchvision
import torchvision.transforms as transforms
import torch

import utils

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


collate_functions = utils.EasyDict(**{
    'tensor_stack': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=stack_cast),
    'tensor_concat': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=concat_cast),
    'list_concat': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=None),
})
keysets = utils.EasyDict(**{
    'image_and_label':['image', 'label'],
})


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, keys=keysets.image_and_label, collate_fn = collate_functions.tensor_stack, index_casts=None):
        self.dataset = dataset
        self.keys = keys
        self.collate_fn = collate_fn

        if len(keys) == 1:
            self.item_cast = lambda x : (x,)
        else:
            self.item_cast = lambda x : x

        if index_casts is None:
            first_point = self.dataset[0]
            first_point = self.item_cast(first_point)
            
            self.index_casts = []
            for item in first_point:
                if isinstance(item, tuple) or isinstance(item, list):
                    self.index_casts.append(torch.Tensor)
                elif isinstance(item, int):
                    self.index_casts.append(lambda x : torch.tensor(x, dtype=torch.int64))
                elif isinstance(item, float):
                    self.index_casts.append(lambda x : torch.tensor(x, dtype=torch.float32))
                else:
                    self.index_casts.append(lambda x : x)
                
        else:
            self.index_casts = index_casts
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item = self.item_cast(item)
        
        return_dict = {'_size':1}
        for i, value in enumerate(item):
            return_dict[self.keys[i]] = self.index_casts[i](value)

        return utils.DataDict(**return_dict)




def stack_channels(x):
    return torch.cat((x,x,x),dim=0)

def noise(x):
    return torch.clamp(x+(1.0/256)*torch.randn(x.size()), 0.0, 1.0)



def stack_cast(x):
    return torch.stack(x)

def concat_cast(x):
    return torch.cat(x)


def get_dataset(dataset='mnist', train=True):

    if dataset == 'mnist':
        im_transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                stack_channels, # So it is comparable with cifar
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.MNIST(root_dir, train=train, transform=im_transform, target_transform=None, download=True)
        keys = keysets.image_and_label        
        collate_fn = collate_functions.tensor_stack
        return DatasetWrapper(dataset)

    elif dataset == 'cifar10':
        im_transform = transforms.Compose([
                transforms.ToTensor(),
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.CIFAR10(root_dir, train=train, transform=im_transform, target_transform=None, download=True)
        return DatasetWrapper(dataset)

    ## Add other datasets


    return dataset


