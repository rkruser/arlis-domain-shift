import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt


from utils import EasyDict as edict


"""
out+x --> nonlinearity --> (batch norm) --> conv layer --> nonlin --> (bn) --> conv --> out+x

"""

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
            layers.append(nonlinearity)
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            layers.append(get_layer(nmaps))
            
        if resblock:
            layers.append(nonlinearity)
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            layers.append(get_layer(nmaps))
            
        if len(layers) == 0:
            layers.append(nonlinearity)
            if batchnorm:
                layers.append(batchnorm_layer(nmaps))
            
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
 


class Encoder(nn.Module):
    def __init__(self, size, ncolors=3, lean=False, very_lean=False, layer_kwargs=None):
        super().__init__()
        
        assert((size&(size-1) == 0) and size != 0) # Check if exact power of 2
        
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
        
        
    def forward(self, x):
        x = self.initial_layers(x)
        x = self.main_layers(x)
        x = self.final_layer(x)
        x = x.reshape(x.size(0),-1)
        #x = x.reshape(x.size(0), -1)
        #x = self.final_fc(x)
        return x
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class Decoder(nn.Module):
    def __init__(self, size, ncolors=3, lean=False, very_lean=False, layer_kwargs=None):
        super().__init__()
        
        assert((size&(size-1) == 0) and size != 0) # Check if exact power of 2
        
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
            if lean:
                layer_kwargs.batchnorm = False
            elif very_lean:
                layer_kwargs.resblock = False
                layer_kwargs.nlayers = 1
                layer_kwargs.batchnorm=False
        
        
        layers = []
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
        
  
class domain_adversary(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(512,512)
        self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
        self.layer2 = nn.Linear(512,1)
        
    def forward(self, x):
        return self.layer2(self.nonlinearity(self.layer1(x)))
        


def test1():
    layer = get_conv_resblock_downsample_layer(6)
    print(layer)
    
    layer2 = get_conv_resblock_upsample_layer(6)
    print(layer2)
    
    layer3 = get_fc_resblock_layer(6)
    print(layer3)


def test2():
    net = Encoder(32, lean=False, very_lean=True)
    print(net)
    net = Decoder(32, lean=False, very_lean=True)
    print(net)
    
def test3():
    net = phi()
    print(net)
    

if __name__ == '__main__':
    test3()






        