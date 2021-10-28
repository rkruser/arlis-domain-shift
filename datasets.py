#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Datasets: MNIST, Cifar10, kaggle planes, etc.
"""

### Imports

import os

import torchvision
import torchvision.transforms as transforms
import torch

import utils

import socket





collate_functions = utils.EasyDict(**{
    'tensor_stack': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=stack_cast),
    'tensor_concat': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=concat_cast),
    'list_concat': lambda x : utils.collate_dicts(x, dict_type=utils.DataDict, cast_func=None),
})
keysets = utils.EasyDict(**{
    'image_and_label':['image', 'label'],
})



class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, length=1000, point_size=64, nlabels=10):
        self.length = length
        self.point_size = 64
        self.nlabels = nlabels
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return utils.DataDict(image=torch.randn(self.point_size), label=torch.randint(self.nlabels,size=(1,))[0], _size=1)

class GeneratorOutputLoader(torch.utils.data.Dataset):
    def __init__(self, generator_net, length, batch_size, latent_size, device):
        self.generator_net = generator_net
        self.length = length
        self.device = device
        self.latent_size = latent_size
        self.batch_size = batch_size
    
    def __len__(self):
        return self.length

    def __iter__(self):
        self._itercount = 0
        return self

    def __next__(self):
        if self._itercount < self.length:
            latent = torch.randn(self.batch_size, self.latent_size, device=self.device)
            outputs = self.generator_net(latent)
            self._itercount += 1
            return utils.DataDict(fake_outputs=outputs, latent_codes=latent, _size=self.batch_size)
        else:
            raise StopIteration
             
         

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, keys=keysets.image_and_label, index_casts=None):
        self.dataset = dataset
        self.keys = keys

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


# What to do about collate_fn method?
class SavedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensorfile):
        self.data = torch.load(tensorfile, map_location='cpu')
        self.keys = self.data.keys()
        for key in self.data:
            if not key.startswith('_'):
                self.data[key] = self.data[key].to(torch.float32) #weirdly necessary

    def __len__(self):
        return self.data.data_size()

    def __getitem__(self, index):
        return_dict = {key:self.data[key][index] for key in self.keys if not key.startswith('_')}
        return_dict['_size'] = 1
        return utils.DataDict(**return_dict)


class ConcatDatasets(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.datasets = args

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return_dict = {'_size':1}
        for dataset in self.datasets:
            return_dict.update(dataset[index])

        return utils.DataDict(**return_dict)



def train_val_split(dataset, train_proportion=0.8, random_seed=314159265358979323):
    split_index = int(train_proportion*len(dataset))
    rng = torch.Generator()
    rng.manual_seed(random_seed)

    return torch.utils.data.random_split(dataset, [split_index, len(dataset)-split_index], generator=rng)



def stack_channels(x):
    return torch.cat((x,x,x),dim=0)

def noise(x):
    return torch.clamp(x+(1.0/256)*torch.randn(x.size()), 0.0, 1.0)



def stack_cast(x):
    return torch.stack(x)

def concat_cast(x):
    return torch.cat(x)


def get_dataset(dataset='mnist', dataset_folder='/scratch0/datasets', train=True):
    if dataset == 'mnist':
        im_transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                stack_channels, # So it is comparable with cifar
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.MNIST(dataset_folder, train=train, transform=im_transform, target_transform=None, download=True)
        dataset = DatasetWrapper(dataset)

    elif dataset == 'cifar10':
        im_transform = transforms.Compose([
                transforms.ToTensor(),
                noise,
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
                )
        dataset = torchvision.datasets.CIFAR10(dataset_folder, train=train, transform=im_transform, target_transform=None, download=True)
        dataset = DatasetWrapper(dataset)
    elif dataset == 'testdataset':
        return RandomDataset()


    return dataset

def get_generated_dataset(dataset_file, dataset_folder):
    return SavedTensorDataset(os.path.join(dataset_folder,dataset_file))
