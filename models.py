#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image models for small datasets
"""



### Imports
import argparse
import pickle


import torch
import torch.nn as nn


import os

from stylegan3.stylegan3_ffhqu import get_stylegan_ffhqu
from utils import EasyDict as edict

### Generator DCGAN

# Model for 32 by 32 images
class NetG32(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(opt.nz, opt.ngf * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ngf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*4) x 4 x 4
        nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ngf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*2) x 7 x 7
        nn.ConvTranspose2d(opt.ngf * 2,     opt.ngf, 4, 2, 1, bias=False),
        # for 28 x 28
#        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
        nn.BatchNorm2d(opt.ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf) x 14 x 14
        nn.ConvTranspose2d(    opt.ngf,      opt.nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 32 x 32
    )

  def forward(self, codes):
    codes = codes.view(codes.size(0),codes.size(1),1,1)
    return self.main(codes)






### Discriminator
    
class NetD32(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
       # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        
        nn.Conv2d(opt.ndf*4, opt.ndf*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ndf*4), #state size: (ndf*4)x1x1
        nn.LeakyReLU(0.2, inplace=True)
    )

    # Predictor takes main output and produces a probability
    self.predictor = nn.Sequential(
        nn.Linear(opt.ndf*4, 1)
    )


  def forward(self, images):
    output = self.main(images)
    output = output.reshape(output.size(0),-1)
    prediction = self.predictor(output).squeeze(1)
    return prediction







### Encoder


class NetE32(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
       # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        
        nn.Conv2d(opt.ndf*4, opt.ndf*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ndf*4), #state size: (ndf*4)x1x1
        nn.LeakyReLU(0.2, inplace=True)
    )

    # Predictor takes main output and produces a probability
    self.embedder = nn.Sequential(
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(opt.ndf*4, opt.nz)
    )



  def forward(self, images):
    output = self.main(images)
    output = output.reshape(output.size(0),-1)
    embedding = self.embedder(output)
    return embedding



### Regressor

class NetR32(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
       # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        
        nn.Conv2d(opt.ndf*4, opt.ndf*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ndf*4), #state size: (ndf*4)x1x1
        nn.LeakyReLU(0.2, inplace=True)
    )

    # Predictor takes main output and produces a probability
    self.predictor = nn.Sequential(
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(opt.ndf*4, opt.regressor_out_dims)
    )



  def forward(self, images):
    output = self.main(images)
    output = output.reshape(output.size(0),-1)
    prediction = self.predictor(output).squeeze(1)
    return prediction


# General classifier

class NetC32(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(opt.ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        # for 28 x 28
           # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
        nn.BatchNorm2d(opt.ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        
        nn.Conv2d(opt.ndf*4, opt.ndf*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(opt.ndf*4), #state size: (ndf*4)x1x1
        nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Predictor takes main output and produces a probability
        self.embedder = nn.Sequential(
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(opt.ndf*4, opt.embedding_dims)
        )
        
        self.predictor = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.embedding_dims, opt.output_dims),
            nn.LogSoftmax(dim=1)
            )

    def embed(self, images):
        output = self.main(images)
        output = output.reshape(output.size(0),-1)
        embedding = self.embedder(output)
        return embedding
    
    
    def forward(self, images):
        embedding = self.embed(images)
        prediction = self.predictor(embedding)
        return prediction








def weights_init(m):
    if isinstance(m, torch.nn.Conv2d): 
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            #torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.weight,mean=0.0, std=0.4)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)








        


def load_torch_class(classname, *args, **kwargs):
    kwargs = edict(**kwargs)
    if classname == 'netg32':
        opt = edict()
        opt.nz = kwargs.latent_dimension
        opt.ngf = kwargs.hidden_dimension_base
        opt.nc = kwargs.output_dimension[0]
        return NetG32(opt)
    elif classname == 'netd32':
        opt = edict()
        opt.ndf = kwargs.hidden_dimension_base
        opt.nc = kwargs.input_dimension[0]
        return NetD32(opt)
    elif classname == 'adam':
        return torch.optim.Adam(*args, **kwargs)
    elif classname == 'steplr':
        return torch.optim.lr_scheduler.StepLR(*args, **kwargs)
    elif classname == 'nete32':
        opt = edict()
        opt.nz = kwargs.output_dimension
        opt.ndf = kwargs.hidden_dimension_base
        opt.nc = kwargs.input_dimension[0]
        return NetE32(opt)
    elif classname == 'netr32':
        opt = edict()
        opt.ndf = kwargs.hidden_dimension_base
        opt.nc = kwargs.input_dimension[0]
        opt.regressor_out_dims = kwargs.output_dimension
        return NetR32(opt)
    elif classname == 'netc32':
        opt = edict()
        opt.ndf = kwargs.hidden_dimension_base
        opt.nc = kwargs.input_dimension[0]
        opt.embedding_dims = kwargs.feature_dimension
        opt.output_dims = kwargs.output_dimension
        return NetC32(opt)
    elif classname == 'testg':
        return torch.nn.Linear(8,64)
    elif classname == 'testd':
        return torch.nn.Linear(64,1)
    elif classname == 'teste':
        return torch.nn.Linear(64,8)
    elif classname == 'testr':
        return torch.nn.Linear(64,2)
    elif classname == 'testc':
        return torch.nn.Linear(64,1)
    elif classname == 'stylegan3-ffhqu':
        opt = edict()
        return get_stylegan_ffhqu(opt)

    elif classname == 'stylegan2-ada-cifar10':
        print("Opening pretrained stylegan2")
        with open(kwargs['filename'], 'rb') as f:
            return pickle.load(f)['G_ema']

    






