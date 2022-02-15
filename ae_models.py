import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os


from utils import EasyDict as edict


################################################################################
################  Neural net helper functions 
################################################################################

def get_fc_layer(nmaps, out_maps=None, downsample=None, upsample=None):
    out_maps = nmaps
    if downsample is not None:
        out_maps = nmaps // downsample
    elif upsample is not None:
        out_maps = nmaps*upsample
    return nn.Linear(nmaps, out_maps)

def get_conv_layer(nmaps, out_maps=None, downsample=None, upsample=None, kernel=3, stride=1, padding=1):
    g_out_maps = nmaps
    conv_constructor = nn.Conv2d
    if downsample is not None:
        stride = downsample
        kernel = 4
        g_out_maps = nmaps*downsample #inverted from fc layer
    elif upsample is not None:
        stride = upsample
        kernel = 4
        g_out_maps = nmaps//upsample
        conv_constructor = nn.ConvTranspose2d
    
    if out_maps is None:
        out_maps = g_out_maps
    return conv_constructor(nmaps, out_maps, kernel, stride, padding)



class block_sample_layer(nn.Module):
    def __init__(self, nmaps, max_out_maps=None, min_out_maps=None,
                 get_layer=get_conv_layer, 
                 nonlinearity=nn.LeakyReLU(0.2,inplace=True),
                 downsample = None,
                 upsample = None,
                 nlayers=2,
                 #omit_last_if_no_dimension_change=True,
                 resblock=True,
                 batchnorm=True, 
                 batchnorm_layer=nn.BatchNorm2d, 
                 pool=False,
                 pool_layer = nn.MaxPool2d,
                 simple_upsample=False,
                 upsample_layer = nn.Upsample):
        super().__init__()
        
        layers = []
        self.resblock = resblock
        
        for _ in range(nlayers-1):
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            layers.append(nonlinearity) #switched order with batchnorm
            layers.append(get_layer(nmaps))
            
        if resblock:
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            layers.append(nonlinearity) # switched order with batchnorm
            layers.append(get_layer(nmaps))
            
        if len(layers) == 0:
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            layers.append(nonlinearity) #switched order with batchnorm
            
        self.last_layer = None
        if downsample or upsample or (not resblock):
            if pool and (downsample is not None):
                self.last_layer = pool_layer(downsample)
            elif simple_upsample and (upsample is not None):
                self.last_layer = upsample_layer(scale_factor=upsample)
            else:
                out_maps = None
                if max_out_maps is not None and (nmaps == max_out_maps):
                    out_maps = max_out_maps
                if min_out_maps is not None and (nmaps==min_out_maps):
                    out_maps = min_out_maps
                self.last_layer = get_layer(nmaps, out_maps=out_maps, downsample=downsample, upsample=upsample)
        else:
            self.last_layer = nn.Sequential()
            
        self.main_layers = nn.Sequential(*layers)
   
        
    def forward(self, x):
        out = self.main_layers(x)
        if self.resblock:
            out += x
        out = self.last_layer(out)
        return out
        

def get_conv_resblock_downsample_layer(nmaps):
    return block_sample_layer(nmaps, downsample=2)
   
def get_conv_resblock_upsample_layer(nmaps):
    return block_sample_layer(nmaps, upsample=2)

def get_fc_resblock_layer(nmaps):
    return block_sample_layer(nmaps, get_layer=get_fc_layer, downsample=None, batchnorm_layer=nn.BatchNorm1d)



################################################################################
################  Main networks
################################################################################
class Encoder(nn.Module):
    def __init__(self, size, ncolors=3, lean=False, very_lean=False, layer_kwargs=None, all_linear=False, add_linear=False):
        super().__init__()
        
        assert((size&(size-1) == 0) and size != 0) # Check if exact power of 2
        
        self.nonlinearity = None
        if all_linear:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
        
        
        
        initial_layers = []
        
        
        exponent = int(np.log2(size)+0.001)
        max_exponent = 9
        
        if exponent == 5:
            start_exponent=5
            final_nfeats = 256
        else:
            start_exponent = max(11-exponent, 5)
            final_nfeats = 512
        # 1024 = 8 feats
        # 512 = 8 feats
        # 256 = 8 feats
        # 128 = 16 feats
        # 64 = 32 feats
        # 32 = 16 feats (breaks the pattern)
        # etc.
        
        count = exponent-2
        
        prev_channels = ncolors
        if exponent >= 10:
            initial_layers.append(nn.Conv2d(prev_channels, 2**start_exponent, 7, 2, 3))
            initial_layers.append(self.nonlinearity)
            prev_channels = 2**start_exponent
            #start_exponent += 1
            count -= 1
        if exponent >= 9:
            initial_layers.append(nn.Conv2d(prev_channels, 2**start_exponent, 7, 2, 3))
            initial_layers.append(self.nonlinearity)
            prev_channels = 2**start_exponent
            #start_exponent += 1
            count -= 1
        if exponent >= 8:
            initial_layers.append(nn.Conv2d(prev_channels, 2**start_exponent, 4, 2, 1))
            count -= 1
        else:
            initial_layers.append(nn.Conv2d(ncolors, 2**start_exponent, 3, 1, 1))
        
        
        if layer_kwargs is None:
            layer_kwargs = edict(downsample=2, max_out_maps=512)
            layer_kwargs.nonlinearity = self.nonlinearity
            if lean:
                layer_kwargs.batchnorm = False
            elif very_lean:
                layer_kwargs.resblock = False
                layer_kwargs.nlayers = 1
                layer_kwargs.batchnorm=False
        
        
        layers = []
        for k in range(count):
            layers.append(block_sample_layer(2**(min(start_exponent+k, max_exponent)), **layer_kwargs))
        # get 512 x 2 x 2
            
        self.initial_layers = nn.Sequential(*initial_layers)
        self.main_layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(self.nonlinearity,
                                         nn.Conv2d(final_nfeats, 32, 1, 1, 0))
        self.add_linear = add_linear
        if add_linear:
            self.linear_postlayer = nn.Linear(512,512)
        
    def forward(self, x):
        x = self.initial_layers(x)
        x = self.main_layers(x)
        x = self.final_layer(x)
        
        if self.add_linear:
            x = x.reshape(x.size(0),-1)
            x = self.linear_postlayer(x)
        #x = x.reshape(x.size(0),-1)
        #x = x.reshape(x.size(0), -1)
        #x = self.final_fc(x)
        return x
        


class Decoder(nn.Module):
    def __init__(self, size, ncolors=3, lean=False, very_lean=False, layer_kwargs=None, all_linear=False, add_linear=False):
        super().__init__()
        
        assert((size&(size-1) == 0) and size != 0) # Check if exact power of 2
        
        self.nonlinearity = None
        if all_linear:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
        
        
        exponent = int(np.log2(size)+0.001)
        count = max(exponent-7, 0)
        min_exponent = 5
        
        if exponent == 5:
            start_exponent=8
        else:
            start_exponent = 9
    
        
        if layer_kwargs is None:
            layer_kwargs = edict(upsample=2, min_out_maps=2**min_exponent)
            layer_kwargs.nonlinearity = self.nonlinearity
            if lean:
                layer_kwargs.batchnorm = False
            elif very_lean:
                layer_kwargs.resblock = False
                layer_kwargs.nlayers = 1
                layer_kwargs.batchnorm=False
        
        
        layers = []
        
        self.add_linear = add_linear
        if add_linear:
            self.linear_prelayer = nn.Linear(512,512)
        
        layers.append(nn.ConvTranspose2d(32,2**start_exponent,1,1,0))
        
        for k in range(exponent-2-count):
            layers.append(block_sample_layer(2**(max(start_exponent-k, min_exponent)), **layer_kwargs))
        last_exponent = max(exponent-2-count, min_exponent)
        last_value = 2**last_exponent
            
        last_layers = []
        if count < 1:
            last_layers += [self.nonlinearity, nn.Conv2d(2**last_exponent, 3, 3, 1, 1)]
        else:
            for i in range(count):
                next_value = 3 if i==(count-1) else 2**max(last_exponent-1-count, min_exponent)
                last_layers += [self.nonlinearity, 
                                   nn.ConvTranspose2d(last_value, next_value, 4, 2, 1)]
                last_value=next_value
                #last_exponent -= 1
        
        
        
        self.main_layers = nn.Sequential(*layers)
        self.last_layers = nn.Sequential(*last_layers)      
        
        
        
    def forward(self, x):
        if self.add_linear:
            x = self.linear_prelayer(x)
            x = x.reshape(x.size(0), 32, 4, 4)
        x = self.main_layers(x)
        x = self.last_layers(x)
        return x
        


class Phi(nn.Module):
    def __init__(self, nblocks=4):
        super().__init__()
        
        self.first_layer = nn.Linear(512,1024)
        
        layers = []
        for _ in range(nblocks):
            layers.append(get_fc_resblock_layer(1024))
            
        self.main_layers = nn.Sequential(*layers)
            
        self.last_layer = nn.Sequential(
                            nn.LeakyReLU(0.2,inplace=True),
                            nn.Linear(1024,512)
                            )
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.main_layers(x)
        x = self.last_layer(x)
        return x
        
        
        
class Phi_regressor(nn.Module):
    def __init__(self, nblocks=4):
        super().__init__()
        
        self.first_layer = nn.Linear(512,1024)
        
        layers = []
        for _ in range(nblocks):
            layers.append(get_fc_resblock_layer(1024))
            
        self.main_layers = nn.Sequential(*layers)
            
        self.last_layer = nn.Sequential(
                            nn.LeakyReLU(0.2,inplace=True),
                            nn.Linear(1024,514) #one extra for log prob, one for norm of vector difference
                            )
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.main_layers(x)
        x = self.last_layer(x)
        return x        
        

        
################################################################################
################  Aggregate models
################################################################################        
class Domain_adversary(nn.Module):
    def __init__(self, linear=True):
        super().__init__()
        
        self.linear = linear
        if linear:
            self.layer1 = nn.Linear(512,1)
            
        else:
            self.layer1 = nn.Linear(512,512)
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
            self.layer2 = nn.Linear(512,1)
        
    def forward(self, x):
        if self.linear:
            return self.layer1(x).squeeze(1)
        else:
            return self.layer2(self.nonlinearity(self.layer1(x))).squeeze(1)

    
    
class Autoencoder_Model:
    def __init__(self, input_size=32, linear=True, very_lean=True, use_adversary=True):
        self.use_adversary = use_adversary
        
        self.encoder = Encoder(input_size, very_lean=very_lean, all_linear=linear, add_linear=True).cuda()
        self.decoder = Decoder(input_size, very_lean=very_lean, all_linear=linear, add_linear=True).cuda()
        
        if self.use_adversary:
            self.domain_adversary = Domain_adversary(linear=linear).cuda()
        
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=0.0001)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=0.0001)
        
        if self.use_adversary:
            self.domain_adversary_optim = torch.optim.Adam(self.domain_adversary.parameters(), lr=0.0001)

class Phi_Model:
    def __init__(self):
        self.e2z = Phi().cuda()
        self.z2e = Phi().cuda()
        
        self.e2z_optim = torch.optim.Adam(self.e2z.parameters(),lr=0.0001)
        self.z2e_optim = torch.optim.Adam(self.z2e.parameters(), lr=0.0001)




################################################################################
################  Training functions
################################################################################

# dataloader assumed to be a multiloader
# keys: real, fake, augmented
def train_autoencoder(model, dataloader, n_epochs, use_adversary=True):
    #reconstruction_loss = torch.nn.BCEWithLogitsLoss()
    #reconstruction_loss = torch.nn.MSELoss()
    #reconstruction_loss = torch.nn.L1Loss()
    
    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    def reconstruction_loss(x,y):
        return l1_lossfunc(x,y)+l2_lossfunc(x,y)
    
    def reg_loss(e):
        return (torch.norm(e.view(e.size(0),-1), dim=1)**2).mean()
    
    adversary_loss = None
    if use_adversary:
        adversary_loss = torch.nn.BCEWithLogitsLoss()
    
    model.encoder.train()
    model.decoder.train()
    
    if use_adversary:
        model.domain_adversary.train()
        lmbda_adv = 1
        lmbda_reg = 0.001
    else:
        lmbda_adv = 0
        lmbda_reg = 0
    
    phase = 0
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        for i, batch in enumerate(dataloader):
            x_real, _ = batch['real']
            x_fake, _ = batch['fake']
            x_augmented, _ = batch['augmented']
            
            x_real = x_real.cuda()
            y_real = torch.zeros(len(x_real),device=x_real.device) #domain labels
            
            x_fake = x_fake.cuda()
            y_fake = torch.ones(len(x_fake), device=x_fake.device) #domain labels
            
            x_augmented = x_augmented.cuda()


            
            
            loss_info = None
            if phase == 0:
                # train encoder and decoder
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                enc_x_aug = model.encoder(x_augmented)
 
                
                if use_adversary:
                    predicted_real_domains = model.domain_adversary(enc_x_real)
                    predicted_fake_domains = model.domain_adversary(enc_x_fake)
                    adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                    regularizing_loss = (1.0/3)*(reg_loss(enc_x_real)+reg_loss(enc_x_fake) + reg_loss(enc_x_aug))
                else:
                    adv_loss = 0
                    regularizing_loss = 0

                recon_real = torch.tanh(model.decoder(enc_x_real)) #add a variant of tanh?
                recon_fake = torch.tanh(model.decoder(enc_x_fake))
                recon_aug = torch.tanh(model.decoder(enc_x_aug))
                
                recon_loss = (1.0/3)*(reconstruction_loss(recon_real, x_real) + reconstruction_loss(recon_fake, x_fake) + reconstruction_loss(recon_aug, x_augmented))

                
                total_loss = recon_loss - lmbda_adv*adv_loss +lmbda_reg*regularizing_loss
                #total_loss = recon_loss#+regularizing_loss
                
                model.encoder.zero_grad()
                model.decoder.zero_grad()
                total_loss.backward()
                model.encoder_optim.step()
                model.decoder_optim.step()
                
                if use_adversary:
                    lossinfo = "recon loss = {0}, adv loss = {1}".format(recon_loss.item(), adv_loss.item())
                else:
                    lossinfo = "recon loss = {0}".format(recon_loss.item())
                #lossinfo = "recon loss = {0}".format(recon_loss.item())
                
                
            if phase == 1:
                # Train adversary
               
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                predicted_real_domains = model.domain_adversary(enc_x_real)
                predicted_fake_domains = model.domain_adversary(enc_x_fake)
                adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                               
                model.domain_adversary.zero_grad()
                adv_loss.backward()
                model.domain_adversary_optim.step()
    
                lossinfo = "adv loss = {0}".format(adv_loss.item())
    
    
            if i%100 == 0 or i%100 == 1:
                print(i, lossinfo)
    
            if use_adversary:
                phase = 1-phase
    
    




def train_invertible(model, dataloader, n_epochs):
    lossfunc = torch.nn.MSELoss()
    lmbda = 0.05
    

    def pressure_loss(x, dim=512, eps = 0.1):
        numerator = dim
        square_norms = x.square().sum(dim=1)
        losses = numerator / (square_norms + eps)
        total_loss = losses.mean()
        return total_loss  


    model.e2z.train()
    model.z2e.train()
  


    for n in range(n_epochs):
        print("Epoch", n)
        for i, batch in enumerate(dataloader):
            z_fake, e_fake, _ = batch['fake']
            e_real, _ = batch['real']
            e_aug, _ = batch['augmented']
            
            
            z_fake = z_fake.cuda()
            e_fake = e_fake.cuda()
            e_real = e_real.cuda()
            e_aug = e_aug.cuda()
            
            
            z_real_pred = model.e2z(e_real)
            z_fake_pred = model.e2z(e_fake)
            z_aug_pred = model.e2z(e_aug)
            
            e_real_cycle = model.z2e(z_real_pred)
            e_fake_cycle = model.z2e(z_fake_pred)
            e_aug_cycle =model.z2e(z_aug_pred)
            
            
            z_fake_loss = lossfunc(z_fake_pred, z_fake)
            e_fake_cycle_loss = lossfunc(e_fake_cycle, e_fake)
            e_real_cycle_loss = lossfunc(e_real_cycle, e_real)
            e_aug_cycle_loss = lossfunc(e_aug_cycle, e_aug)
            z_aug_pressure_loss = pressure_loss(z_aug_pred)
            
            total_loss = z_fake_loss + e_fake_cycle_loss + e_real_cycle_loss + \
                           e_aug_cycle_loss + lmbda*z_aug_pressure_loss
            
            
            

            model.z2e.zero_grad()
            model.e2z.zero_grad()
            total_loss.backward()
            model.z2e_optim.step()
            model.e2z_optim.step()
                

            if i%100 == 0:
                print("  {5}: z_fake_loss={0}, e_fake_cycle_loss={1}, e_real_cycle_loss={2}, e_aug_cycle_loss={3}, z_aug_pressure_loss={4}".format(z_fake_loss.item(), e_fake_cycle_loss.item(), e_real_cycle_loss.item(), e_aug_cycle_loss.item(), z_aug_pressure_loss.item(), i))
