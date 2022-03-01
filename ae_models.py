import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os


from utils import EasyDict as edict



##
global_lr = 0.001



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
        
        
        exponent = int(np.log2(size)+global_lr)
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
        
        
        exponent = int(np.log2(size)+global_lr)
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
    def __init__(self, nblocks=4, hidden=1024, num_in=512, num_out=512):
        super().__init__()
        
        self.first_layer = nn.Linear(num_in,hidden)
        
        layers = []
        for _ in range(nblocks):
            layers.append(get_fc_resblock_layer(hidden))
            
        self.main_layers = nn.Sequential(*layers)
            
        self.last_layer = nn.Sequential(
                            nn.LeakyReLU(0.2,inplace=True),
                            nn.Linear(hidden,num_out)
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
class Small_Classifier(nn.Module):
    def __init__(self, linear=False, in_feats = 512):
        super().__init__()
        
        self.linear = linear
        if linear:
            self.layer1 = nn.Linear(in_feats,1)
            
        else:
            self.layer1 = nn.Linear(in_feats,512)
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
            self.layer2 = nn.Linear(512,1)
        
    def forward(self, x):
        if self.linear:
            return self.layer1(x).squeeze(1)
        else:
            return self.layer2(self.nonlinearity(self.layer1(x))).squeeze(1)


class Small_Decoder(nn.Module):
    def __init__(self, linear=False):
        super().__init__()
        if linear:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
        
        self.block = nn.Sequential(
            nn.Linear(512,512),
            self.nonlinearity,
            nn.Linear(512,512),
            self.nonlinearity,
            nn.Linear(512,512)
            )

        self.predict = nn.Linear(512,1000)

    def forward(self, x):
        block_out = self.block(x)
        predicted = self.predict(block_out)

        return block_out, predicted

   
    
class Autoencoder_Model:
    def __init__(self, input_size=32, linear=True, very_lean=True, use_adversary=False, use_features=False,
                 mixed=False):
        self.use_adversary = use_adversary
        
        encoder_linear = (linear or mixed)
        decoder_linear = (linear and not mixed)

        
        self.encoder = Encoder(input_size, very_lean=very_lean, all_linear=encoder_linear, add_linear=True).cuda()
        self.decoder = Decoder(input_size, very_lean=(very_lean and not mixed), all_linear=decoder_linear, add_linear=True).cuda()
        
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=global_lr)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=global_lr)
        
        if self.use_adversary:
            self.domain_adversary = Small_Classifier(linear=linear).cuda()
            self.domain_adversary_optim = torch.optim.Adam(self.domain_adversary.parameters(linear=linear), lr=global_lr)

        if use_features:
            self.feature_encode = nn.Linear(1000,512).cuda()
            self.feature_decode = Small_Decoder(linear=mixed).cuda()
            self.feature_encode_optim = torch.optim.Adam(self.feature_encode.parameters(), lr=global_lr)
            self.feature_decode_optim = torch.optim.Adam(self.feature_decode.parameters(), lr=global_lr)


class Phi_Model:
    def __init__(self, use_adversary=False, use_friend=False):
        self.e2z = Phi().cuda()
        self.z2e = Phi().cuda()
        
        self.e2z_optim = torch.optim.Adam(self.e2z.parameters(),lr=global_lr)
        self.z2e_optim = torch.optim.Adam(self.z2e.parameters(), lr=global_lr)
        
        if use_adversary:
            self.adversary = Small_Classifier(linear=False).cuda()
            self.adversary_optim = torch.optim.Adam(self.adversary.parameters(),lr=global_lr)

        if use_friend:
            self.friend = Small_Classifier(linear=False).cuda()
            self.friend_optim = torch.optim.Adam(self.friend.parameters(),lr=global_lr)




################################################################################
################  Training functions
################################################################################

# dataloader assumed to be a multiloader
# keys: real, fake, augmented
def train_autoencoder(model, dataloader, n_epochs, use_adversary=True, use_features=False):
    #reconstruction_loss = torch.nn.BCEWithLogitsLoss()
    #reconstruction_loss = torch.nn.MSELoss()
    #reconstruction_loss = torch.nn.L1Loss()
    
    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    def reconstruction_loss(x,y):
        return l1_lossfunc(x,y)+l2_lossfunc(x,y)
    
    def reg_loss(e):
        return (torch.norm(e.view(e.size(0),-1), dim=1)**2).mean()
    
    
    model.encoder.train()
    model.decoder.train()
    
    if use_adversary:
        model.domain_adversary.train()
        adversary_loss = torch.nn.BCEWithLogitsLoss()
        lmbda_adv = 1
        lmbda_reg = 0.001
    else:
        lmbda_adv = 0
        lmbda_reg = 0

    if use_features:
        model.feature_encode.train()
        model.feature_decode.train()

    
    phase = 0
    for epoch in range(n_epochs): 
        print("Epoch", epoch)
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][0].cuda()
            x_aug = batch['augmented'][0].cuda()

            if use_features:
                x_real_feats = batch['real'][1].cuda()
                x_fake_feats = batch['fake'][1].cuda()
                x_aug_feats = batch['augmented'][1].cuda()
            
            if use_adversary:
                y_real = torch.zeros(len(x_real),device=x_real.device) #domain labels
                y_fake = torch.ones(len(x_fake), device=x_fake.device) #domain labels
                       
            loss_info = None
            if phase == 0:
                # train encoder and decoder
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                enc_x_aug = model.encoder(x_aug)
 
                if use_features:
                    enc_x_real = enc_x_real + model.feature_encode(x_real_feats)
                    enc_x_fake = enc_x_fake + model.feature_encode(x_fake_feats)
                    enc_x_aug = enc_x_aug + model.feature_encode(x_aug_feats)
                    
                    real_add, recon_x_real_feats = model.feature_decode(enc_x_real)
                    fake_add, recon_x_fake_feats = model.feature_decode(enc_x_fake)
                    aug_add, recon_x_aug_feats = model.feature_decode(enc_x_aug)

                    loss_real_feats = l2_lossfunc(recon_x_real_feats, x_real_feats)
                    loss_fake_feats = l2_lossfunc(recon_x_fake_feats, x_fake_feats)
                    loss_aug_feats = l2_lossfunc(recon_x_aug_feats, x_aug_feats)
                else:
                    recon_real_feats = 0
                    recon_fake_feats = 0
                    recon_aug_feats = 0
                    
                
                if use_adversary:
                    predicted_real_domains = model.domain_adversary(enc_x_real)
                    predicted_fake_domains = model.domain_adversary(enc_x_fake)
                    adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                    regularizing_loss = (1.0/3)*(reg_loss(enc_x_real)+reg_loss(enc_x_fake) + reg_loss(enc_x_aug))
                else:
                    adv_loss = 0
                    regularizing_loss = 0


                if use_features:
                    # idea is to undo adding the features
                    enc_x_real = enc_x_real + real_add
                    enc_x_fake = enc_x_fake + fake_add
                    enc_x_aug = enc_x_aug + aug_add


                recon_real = torch.tanh(model.decoder(enc_x_real)) #add a variant of tanh?
                recon_fake = torch.tanh(model.decoder(enc_x_fake))
                recon_aug = torch.tanh(model.decoder(enc_x_aug))
                
                recon_loss = (1.0/3)*(reconstruction_loss(recon_real, x_real) + reconstruction_loss(recon_fake, x_fake) + reconstruction_loss(recon_aug, x_aug))
                feats_loss = (1.0/3)*(loss_real_feats + loss_fake_feats + loss_aug_feats)

                
                total_loss = recon_loss - lmbda_adv*adv_loss +lmbda_reg*regularizing_loss + feats_loss
                #total_loss = recon_loss#+regularizing_loss
                
                if use_features:
                    model.feature_encode.zero_grad()
                    model.feature_decode.zero_grad()

                model.encoder.zero_grad()
                model.decoder.zero_grad()
                total_loss.backward()
                model.encoder_optim.step()
                model.decoder_optim.step()

                if use_features:
                    model.feature_encode_optim.step()
                    model.feature_decode_optim.step()
                

                lossinfo = "recon loss = {0}".format(recon_loss.item())
                if use_adversary:
                    lossinfo += ", adv loss = {0}, reg loss = {1}".format(adv_loss.item(), regularizing_loss.item())
                if use_features:
                    lossinfo += ", feats loss = {0}".format(feats_loss.item())
                
                
            if phase == 1:
                # Train adversary
               
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                if use_features:
                    enc_x_real += model.feature_encode(x_real_feats)
                    enc_x_fake += model.feature_encode(x_fake_feats)

                predicted_real_domains = model.domain_adversary(enc_x_real)
                predicted_fake_domains = model.domain_adversary(enc_x_fake)
                adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                               
                model.domain_adversary.zero_grad()
                adv_loss.backward()
                model.domain_adversary_optim.step()
    
                lossinfo = "adv loss = {0}".format(adv_loss.item())
    
    
            if (i > 1) and (i-phase)%100 == 0:
                print(i, lossinfo)
    
            if use_adversary:
                phase = 1-phase
    
    


# 3 is "significance factor" (in 3**4 express)
def pressure_loss(x, lmbda_1 = 0.5*(512**(1.5)), lmbda_2 = 1/(2*(512**0.5)*(3**4)), eps = 1e-5):
    square_norms = x.square().sum(dim=1)
    outward_pressure = lmbda_1 / (square_norms + eps)
    inward_pressure = lmbda_2 * square_norms
    total_loss = (inward_pressure+outward_pressure).mean()
    return total_loss  

def ring_pressure_loss(x, r2=512, w=31, significance_factor = 3, eps=1e-5):
    ring_lmbda = (w**2) / (2*(r2-w))
    reg_lmbda = ring_lmbda / ((significance_factor*w)**2)
    x2 = x.square().sum(dim=1)
    ring = ring_lmbda / ((x2-r2).abs() + eps)
    reg = reg_lmbda*x2
    total_loss = (ring+reg).mean()
    return total_loss


def mse_loss(x,y):
    return torch.nn.functional.mse_loss(x,y)

def clf_loss(x,y):
    return torch.nn.functional.binary_cross_entropy_with_logits(x,y)



def train_invertible(model, dataloader, n_epochs, use_adversary=False, use_friend=False, print_every=200, pressure='ring'):
    lossfunc = mse_loss
    lmbda_a = 1
    lmbda_f = 1
    lmbda_z_fake = 1
    lmbda_e_fake = 1

    lmbda_aug_cycle = 1
    lmbda_cycle = 1

    if pressure == 'ring':
        print("using ring pressure")
        p_lossfunc = ring_pressure_loss
        lmbda_p = 1
    else:
        p_lossfunc = pressure_loss
        lmbda_p = 3

    model.e2z.train()
    model.z2e.train()


    adversary_loss = None
    friend_lossfunc = None
    if use_adversary:
        model.adversary.train()
        adversary_loss = clf_loss
    if use_friend:
        model.friend.train()
        friend_lossfunc = clf_loss


    lossinfo = None
    phase = 0
    for n in range(n_epochs):
        print("Epoch", n)
        if n < 10:
            print("no adversary, no pressure, no cycle, only bidirectinal targets")
            lmbda_a = 0
            lmbda_p = 0
            lmbda_aug_cycle = 0
            lmbda_cycle = 0.05
        elif n < 20:
            print("add cycle loss")
            lmbda_a = 0
            lmbda_p = 0
            lmbda_aug_cycle = 0.05
            lmbda_cycle = 1
        else:
            if (n // 5)%2 == 0:
                # Use adversary but no pressure
                print("adversary, no pressure, aug cycle")
                lmbda_a = 1 
                lmbda_p = 0
                lmbda_aug_cycle = 1
                lmbda_cycle = 1

            else:
                print("pressure, no adversary, aug cycle")
                lmbda_a = 0
                lmbda_p = 0.05
                lmbda_aug_cycle = 1
                lmbda_cycle = 1


        for i, batch in enumerate(dataloader):
            z_fake, e_fake, _ = batch['fake']
            e_real, _ = batch['real']
            e_aug, _ = batch['augmented']
            
            
            z_fake = z_fake.cuda()
            e_fake = e_fake.cuda()
            e_real = e_real.cuda()
            e_aug = e_aug.cuda()

            y_real = None
            y_fake = None
            y_aug = None
            
            if use_adversary:
                y_real = torch.ones(len(e_real), device=e_real.device)
                y_fake = torch.zeros(len(e_fake), device=e_fake.device)
                y_aug = torch.zeros(len(e_aug), device=e_aug.device)
            

            if phase == 0:
                z_real_pred = model.e2z(e_real)
                z_fake_pred = model.e2z(e_fake)
                z_aug_pred = model.e2z(e_aug)




                if use_adversary:
                    y_real_pred = model.adversary(z_real_pred)
                    y_fake_pred = model.adversary(z_fake_pred)
                    adv_loss = 0.5*(adversary_loss(y_real_pred, y_real) + adversary_loss(y_fake_pred, y_fake))
                else:
                    adv_loss = 0


                if use_friend:
                    y_real_pred = model.friend(z_real_pred)
                    y_aug_pred = model.friend(z_aug_pred)
                    friend_loss = 0.5*(friend_lossfunc(y_real_pred, y_real) + friend_lossfunc(y_aug_pred, y_aug))
                else:
                    friend_loss = 0


                
                e_real_cycle = model.z2e(z_real_pred)
                e_fake_cycle = model.z2e(z_fake_pred)
                e_aug_cycle =model.z2e(z_aug_pred)

                
                e_fake_pred = model.z2e(z_fake)
                z_fake_cycle = model.e2z(e_fake_pred)
                
                # z2e loss
                e_fake_loss = lossfunc(e_fake_pred, e_fake)
                z_fake_cycle_loss = lossfunc(z_fake_cycle, z_fake)

                #e2z loss
                z_fake_loss = lossfunc(z_fake_pred, z_fake)

                # cycle losses
                e_fake_cycle_loss = lossfunc(e_fake_cycle, e_fake)
                e_real_cycle_loss = lossfunc(e_real_cycle, e_real)
                e_aug_cycle_loss = lossfunc(e_aug_cycle, e_aug)

                # pressure losses
                z_aug_pressure_loss = p_lossfunc(z_aug_pred)
                
                total_loss = lmbda_z_fake*z_fake_loss + lmbda_e_fake*e_fake_loss + \
                               lmbda_cycle*(e_fake_cycle_loss + e_real_cycle_loss + z_fake_cycle_loss) + \
                               lmbda_aug_cycle*e_aug_cycle_loss - lmbda_a*adv_loss +\
                               lmbda_f*friend_loss + lmbda_p*z_aug_pressure_loss

                if use_friend:
                    model.friend.zero_grad()
                model.z2e.zero_grad()
                model.e2z.zero_grad()
                total_loss.backward()
                model.z2e_optim.step()
                model.e2z_optim.step()
                if use_friend:
                    model.friend_optim.step()


                if i > 1 and (i%print_every == 0 or (i+1)%print_every==0):
                    lossinfo = "  {5}: z_fake_loss={0}, e_fake_cycle_loss={1}, e_real_cycle_loss={2}, e_aug_cycle_loss={3}, z_aug_pressure_loss={4}".format(z_fake_loss.item(), e_fake_cycle_loss.item(), e_real_cycle_loss.item(), e_aug_cycle_loss.item(), z_aug_pressure_loss.item(), i)
    
                    if use_adversary:
                        lossinfo += ", adv_loss={0}".format(adv_loss.item())
                    if use_friend:
                        lossinfo += ", friend_loss={0}".format(friend_loss.item()) 
    
                    z_norms = {'real':torch.norm(z_real_pred,dim=1).square().mean().item(),
                               'fake':torch.norm(z_fake_pred,dim=1).square().mean().item(),
                               'aug':torch.norm(z_aug_pred,dim=1).square().mean().item()}
    
    
    
                    lossinfo += '\n' + str(z_norms)
                

            elif phase == 1:
                z_real_pred = model.e2z(e_real)
                z_fake_pred = model.e2z(e_fake)
                
                y_real_pred = model.adversary(z_real_pred)
                y_fake_pred = model.adversary(z_fake_pred)

                adv_loss = 0.5*(adversary_loss(y_real_pred, y_real) + adversary_loss(y_fake_pred, y_fake))
                model.adversary.zero_grad()
                adv_loss.backward()
                model.adversary_optim.step()

                lossinfo = "adv loss = {0}".format(adv_loss.item())
               
                

            if i>1 and (i%print_every==0 or (i+1)%print_every ==0):
                print(lossinfo)

            if use_adversary:
                phase = 1-phase

