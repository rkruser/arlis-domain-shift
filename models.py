#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image models for small datasets
"""



### Imports
import argparse


import torch
import torch.nn as nn


import os


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


class NetD32_bigan(nn.Module):
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
        nn.Linear(opt.nz+opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(opt.ndf*4, opt.ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(opt.ndf*4, 1)
    )



  def forward(self, codes, images):
    output = self.main(images)
    output = output.reshape(output.size(0),-1)
    output = torch.cat((codes,output),dim=1)
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
        nn.Linear(opt.ndf*4, 1)
    )



  def forward(self, images):
    output = self.main(images)
    output = output.reshape(output.size(0),-1)
    prediction = self.predictor(output).squeeze(1)
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


# Base class
class Model:
    def __init__(self, opt):
        pass

    def checkpoint(self):
        pass    
    
    # Not sure if should implement this, but maybe
    def load(self):
        pass



class GAN_Model(Model):
    def __init__(self, opt, load_params=None):
        self.opt = opt
        
        self.netg = NetG32(opt).to(opt.device)
        self.netd = NetD32(opt).to(opt.device)
        #self.nete = NetE32(opt).to(opt.device)      

        
        self.netg_optim = torch.optim.Adam(self.netg.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        self.netd_optim = torch.optim.Adam(self.netd.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        #self.nete_optim = torch.optim.Adam(self.nete.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        
        self.netg_scheduler = torch.optim.lr_scheduler.StepLR(self.netg_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        self.netd_scheduler = torch.optim.lr_scheduler.StepLR(self.netd_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        #self.nete_scheduler = torch.optim.lr_scheduler.StepLR(self.nete_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        
        if load_params is not None:
            self.netg.load_state_dict(load_params['netg'])
            self.netd.load_state_dict(load_params['netd'])
            #self.nete.load_state_dict(load_params['nete'])
            
            self.netg_optim.load_state_dict(load_params['optim']['netg_optim'])
            self.netd_optim.load_state_dict(load_params['optim']['netd_optim'])
            #self.nete_optim.load_state_dict(load_params['optim']['nete_optim'])
            
            self.netg_scheduler.load_state_dict(load_params['optim']['netg_scheduler'])
            self.netd_scheduler.load_state_dict(load_params['optim']['netd_scheduler'])
            #self.nete_scheduler.load_state_dict(load_params['optim']['nete_scheduler'])
        else:
            self.netg.apply(weights_init)
            self.netd.apply(weights_init)
            #self.nete.apply(weights_init)
        

    def checkpoint(self, number=None, metrics=None, save_optimizers=True, basename=None, directory=None):
        if number is None:
            number = ''
        else:
            number = '_'+str(number)

        if basename is None:
            basename = self.opt.basename
        
        if directory is None:
            directory = self.opt.modeldir
            
        if save_optimizers:
            optimizers = {
                        'netg_optim':self.netg_optim.state_dict(),
                        'netd_optim':self.netd_optim.state_dict(),
                        #'nete_optim':self.nete_optim.state_dict(), 
                        
                        'netg_scheduler':self.netg_scheduler.state_dict(),
                        'netd_scheduler':self.netd_scheduler.state_dict(),
                        #'nete_scheduler':self.nete_scheduler.state_dict()
                    }
        else:
            optimizers = None
            
        fullpath = os.path.join(directory, basename+number+'.pth')
        data = {
                'opt':self.opt,
                'netg':self.netg.state_dict(),
                'netd':self.netd.state_dict(),
                #'nete':self.nete.state_dict(),
                'optim':optimizers,
                'metrics':metrics
                }
        
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        torch.save(data, fullpath)



class BiGAN_Model(Model):
    def __init__(self, opt, load_params=None):
        self.opt = opt
        
        self.netg = NetG32(opt).to(opt.device)
        self.netd = NetD32_bigan(opt).to(opt.device)
        self.nete = NetE32(opt).to(opt.device)      

        
        self.netg_optim = torch.optim.Adam(self.netg.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        self.netd_optim = torch.optim.Adam(self.netd.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        self.nete_optim = torch.optim.Adam(self.nete.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
        
        self.netg_scheduler = torch.optim.lr_scheduler.StepLR(self.netg_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        self.netd_scheduler = torch.optim.lr_scheduler.StepLR(self.netd_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        self.nete_scheduler = torch.optim.lr_scheduler.StepLR(self.nete_optim, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
        
        if load_params is not None:
            self.netg.load_state_dict(load_params['netg'])
            self.netd.load_state_dict(load_params['netd'])
            self.nete.load_state_dict(load_params['nete'])
            
            self.netg_optim.load_state_dict(load_params['optim']['netg_optim'])
            self.netd_optim.load_state_dict(load_params['optim']['netd_optim'])
            self.nete_optim.load_state_dict(load_params['optim']['nete_optim'])
            
            self.netg_scheduler.load_state_dict(load_params['optim']['netg_scheduler'])
            self.netd_scheduler.load_state_dict(load_params['optim']['netd_scheduler'])
            self.nete_scheduler.load_state_dict(load_params['optim']['nete_scheduler'])
        else:
            self.netg.apply(weights_init)
            self.netd.apply(weights_init)
            self.nete.apply(weights_init)
            
            
            
        

    def checkpoint(self, number=None, metrics=None, save_optimizers=True, basename=None, directory=None):
        if number is None:
            number = ''
        else:
            number = '_'+str(number)

        if basename is None:
            basename = self.opt.basename
        
        if directory is None:
            directory = self.opt.modeldir
            
        if save_optimizers:
            optimizers = {
                        'netg_optim':self.netg_optim.state_dict(),
                        'netd_optim':self.netd_optim.state_dict(),
                        'nete_optim':self.nete_optim.state_dict(), 
                        
                        'netg_scheduler':self.netg_scheduler.state_dict(),
                        'netd_scheduler':self.netd_scheduler.state_dict(),
                        'nete_scheduler':self.nete_scheduler.state_dict()
                    }
        else:
            optimizers = None
            
        fullpath = os.path.join(directory, basename+number+'.pth')
        data = {
                'opt':self.opt,
                'netg':self.netg.state_dict(),
                'netd':self.netd.state_dict(),
                'nete':self.nete.state_dict(),
                'optim':optimizers,
                'metrics':metrics
                }
        
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        torch.save(data, fullpath)
        


#def load_bigan_model(self, directory, basename, number):
#    if number is None:
#        number = ''
#    else:
#        number = '_'+str(number)
#    fullpath = os.path.join(directory, basename+number+'.pth')
#    
#    params = torch.load(fullpath)
#    
#    model = BiGAN_Model(params['opt'])
#    model.netg.load_state_dict()
        

### Testing



### Command line
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=None, help='Name of test to run')
    
    
    
    
    
    opt = parser.parse_args()