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
    lmbda = 0.1
    
    
    model.e2z.train()
    model.z2e.train()
    
    alternator = 0
    for n in range(n_epochs):
        print("Epoch", n)
        for i, batch in enumerate(dataloader):
            z_batch, e_batch = batch
            z_batch = z_batch.cuda()
            e_batch = e_batch.cuda()
            
            main_loss = None
            cycled_loss = None
            if alternator == 0:
                predicted_encodings = model.z2e(z_batch)
                cycled_z = model.e2z(predicted_encodings)
                main_loss = lossfunc(predicted_encodings, e_batch)
                cycled_loss = lossfunc(cycled_z, z_batch)
                loss = main_loss +lmbda*cycled_loss
                
                model.z2e.zero_grad()
                model.e2z.zero_grad()
                loss.backward()
                model.z2e_optim.step()
                
            else:
                predicted_z = model.e2z(e_batch)
                cycled_e = model.z2e(predicted_z)
                main_loss = lossfunc(predicted_z, z_batch) 
                cycled_loss = lossfunc(cycled_e, e_batch)
                loss = main_loss+ lmbda*cycled_loss
                
                model.e2z.zero_grad()
                model.z2e.zero_grad()
                loss.backward()
                model.e2z_optim.step()


            if i>0 and i%100 == 0 or i%101 == 0:
                msg = None
                if alternator == 0:
                    msg = "z2e"
                else:
                    msg = "e2z"
                print(i, msg, main_loss.item(), cycled_loss.item())
                
            alternator = 1-alternator
            




################################################################################
################  Dataset functions
################################################################################    
class BlendedDataset(torch.utils.data.Dataset):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        
    def __getitem__(self, i):
        if i%2 == 0:
            return self.d1[(i//2)]
        else:
            return self.d2[((i-1)//2)]
    
    def __len__(self):
        return 2*min(len(self.d1),len(self.d2))    


def two_domain_dataset():
    fake_data = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
    #z_values = data['z_values']
    images = fake_data['images']
    #images = ((images+1)/2).clamp(0,1)
    fake_labels = torch.ones(len(images))
    fake_dataset = torch.utils.data.TensorDataset(images, fake_labels)
    
    im_transform = tv.transforms.ToTensor()
    #real_data = tv.datasets.CIFAR10('/scratch0/datasets', train=True, transform=im_transform,
    #                                target_transform=None, download=True)
    
#     filtered_real = []
#     for pt in real_data:
#         x,y = pt
#         if y == 1:
#             filtered_real.append(x)
            
#     real_class_1 = torch.stack(filtered_real)
    
#     torch.save({'images':real_class_1}, './models/autoencoder/cifar_real_class_1.pth')
    
    real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
    real_class_1 = 2*real_class_1 - 1
    real_labels = torch.zeros(len(real_class_1))
    real_dataset = torch.utils.data.TensorDataset(real_class_1, real_labels)
    
    
    
    combined_dataset = BlendedDataset(real_dataset, fake_dataset)
    
    return combined_dataset



def get_cifar_class():
#     fake_data = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
#     #z_values = data['z_values']
#     images = fake_data['images']
#     #images = ((images+1)/2).clamp(0,1)
#     fake_labels = torch.ones(len(images))
#     fake_dataset = torch.utils.data.TensorDataset(images, fake_labels)
    
    im_transform = tv.transforms.ToTensor()
    real_data = tv.datasets.CIFAR10('/fs/vulcan-datasets/CIFAR', train=True, transform=im_transform,
                                   target_transform=None, download=False)
    
    filtered_real = []
    for pt in real_data:
        x,y = pt
        if y == 0:
            filtered_real.append(x)
            
    real_class_0 = torch.stack(filtered_real)
    real_class_0 = 2*real_class_0 - 1
    
    torch.save({'images':real_class_0}, './models/autoencoder/cifar_real_class_0.pth')
    
#     real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
#     real_class_1 = 2*real_class_1 - 1
#     real_labels = torch.zeros(len(real_class_1))
#     real_dataset = torch.utils.data.TensorDataset(real_class_1, real_labels)
    
    
    
#     combined_dataset = BlendedDataset(real_dataset, fake_dataset)
    
#     return combined_dataset



def preprocess_cifar(dataset_dir, save_dir):
    print("Dataset_dir", dataset_dir)
    print("Save_dir", save_dir)
    
    im_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    cifar_train = tv.datasets.CIFAR10(dataset_dir, train=True, transform=im_transform,
                               target_transform=None, download=False)
    cifar_test = tv.datasets.CIFAR10(dataset_dir, train=False, transform=im_transform,
                               target_transform=None, download=False)
    
    cifar_sorted = {i:{'train':[], 'test':[]} for i in range(10)}
    for i in range(len(cifar_train)):
        x, y = cifar_train[i]
        cifar_sorted[y]['train'].append(x)
    for i in range(len(cifar_test)):
        x, y = cifar_test[i]
        cifar_sorted[y]['test'].append(x)        
        
        
    for key in cifar_sorted:
        cifar_sorted[key]['train'] = torch.stack(cifar_sorted[key]['train'])
        cifar_sorted[key]['test'] = torch.stack(cifar_sorted[key]['test'])
        
    torch.save(cifar_sorted, os.path.join(save_dir, 'cifar_sorted.pth'))


    
def sample_cifar_stylegan(savedir):
    #encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
    
    from models import load_torch_class
    
    #cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '../repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()
    

    
    
    stylegan_sorted = {i:[] for i in range(10)}
    for const in range(10):
        print("Class", const)
        class_constant = torch.zeros(10, device='cuda')
        class_constant[const] = 1

        z_values = torch.randn(6000, 512, device='cuda')
        
        images = []
        w_vals = []
        for i, batch in enumerate(torch.chunk(z_values, 6000//128 + 1)):
            classes = class_constant.repeat(batch.size(0),1)
            w_values = cifar_stylegan_net.mapping(batch, classes)
            
            image_outputs = cifar_stylegan_net.synthesis(w_values, noise_mode='const', force_fp32=True)
            
            if i==0:
                print(image_outputs[0].min(), image_outputs[0].max())
                
            image_outputs = normalize_to_range(image_outputs)
            
            if i==0:
                print(image_outputs[0].min(), image_outputs[0].max())
            
            images.append(image_outputs.detach().cpu())
            w_vals.append(w_values[:,0,:].cpu())
            


            if i%10 == 0:
                print(i)


            
        all_z = z_values.cpu()
        all_w = torch.cat(w_vals)
        all_ims = torch.cat(images)
        stylegan_sorted[const] = {
            'train': {
                'z_values':all_z[:5000], 
                'w_values':all_w[:5000], 
                'images':all_ims[:5000]
            },
            'test': {
                'z_values':all_z[5000:],
                'w_values':all_w[5000:],
                'images':all_ims[5000:]
            }
        }
        
    torch.save(stylegan_sorted, os.path.join(savedir, 'cifar_stylegan_samples_renorm.pth'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def normalize_to_range(x, minval=-1, maxval=1, element_wise = True):
    if element_wise:
        original_size = x.size()
        x = x.view(x.size(0), -1)
        mins, _ = x.min(dim=1)
        maxes, _ = x.max(dim=1)        
        mins = mins.unsqueeze(1)
        maxes = maxes.unsqueeze(1)
        x = (x-mins)/(maxes-mins)
        x = (maxval-minval)*x + minval
        x = x.view(original_size)
        return x
    else:
        allmin = x.min()
        allmax = x.max()
        x = (x-allmin)/(allmax-allmin)
        x = (maxval-minval)*x + minval
        return x
    
def add_noise_and_renormalize(x, range_proportion=1.0/256, minval=-1, maxval=1):
    coeff = range_proportion*(maxval-minval)
    noise = 2*coeff*torch.rand(x.size()) - coeff
    x += noise
    x = normalize_to_range(x, minval=minval, maxval=maxval, element_wise=True)
    return x
    
    
class Domain_Adversarial_Dataset(torch.utils.data.Dataset):
    def __init__(self, real, fake, real_classes=None):
        self.real = real
        self.fake = fake
        self.real_classes = real_classes
                
    def __getitem__(self, i):
        if i < len(self.real):
            x, y = self.real[i]
            z = -1
            if y in self.real_classes:
                z = 0
            return x, y, z
        else:
            x = self.fake[i-len(self.real)]
            y = -1
            z = 1
            return x, y, z
    
    def __len__(self):
        return len(self.real)+len(self.fake)  

    
class Sorted_Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, train=True, include_keys=['images'], include_labels=None, keep_separate=False):
        data = torch.load(filename)

        self.keep_separate = keep_separate
        self.include_keys = include_keys
        

        if include_labels is None:
            include_labels = data.keys()
        self.include_labels = include_labels
            
            
        if train:
            partition_key = 'train'
        else:
            partition_key = 'test'
        
  
        if keep_separate:
            pass
#             self.data = {}
                
#             self.lengths = []
#             cumulative_length = 0
#             for label in include_labels:
#                 label_data = []
#                 for key in include_keys:
#                     label_data.append(data[label][partition_key][key])
#                 self.data[label] = label_data

#                 self.lengths.append((label, len(label_data[0]), cumulative_length))
#                 cumulative_length += len(label_data[0])

#             self.length = cumulative_length
        
        else:
            self.data = []
            for key in include_keys:
                key_label_data = []
                #label_data = []
                for label in include_labels:
                    key_label_data.append(data[label][partition_key][key])
                #    label_data.append(torch.tensor(label).repeat(len(data[label][key]))
                
                self.data.append(torch.cat(key_label_data))
                
            labels = []
            key0 = include_keys[0]
            for label in include_labels:
                labels.append(torch.tensor(label).repeat( len(data[label][partition_key][key0]) ))
            self.labels = torch.cat(labels)
            self.length = len(self.labels)
                              
            
    def __getitem__(self, i):
        if self.keep_separate:
            pass
        else:
            return [ self.data[k][i] for k in range(len(self.data)) ] + [ self.labels[i] ]

        
    def __len__(self):
        return self.length

    
    
def test_sorted_dataset():
    path0 = './models/autoencoder/cifar_sorted.pth'
    path1 = './models/autoencoder/cifar_sorted_stylegan.pth'
    include_labels = [0, 1]
    
    cifar = Sorted_Dataset(path0, train=True, include_keys=['images'], include_labels=include_labels)
    
    stylegan = Sorted_Dataset(path1, train = True, include_keys=['z_values', 'images'], include_labels = include_labels)
                           
    print(len(cifar))
    print(len(stylegan))
    
    x, y = cifar[500]
    print(x.size(), y)
    
    z, x, y = stylegan[500]
    print(z.size(), x.size(), y)

    
    
class Multi_Dataset_Loader:
    def __init__(self, dset_dict, batch_size=64, loader_length = 'sum', shuffle=False, drop_last=False, all_at_once=True):
        self.dset_dict = dset_dict
        self.loaders = {}
        self.iterators = {}
        self.batch_size = batch_size
        self.length = 0
        self.all_at_once = all_at_once
        
        for key in self.dset_dict:
            dset = self.dset_dict[key]
            loader = torch.utils.data.DataLoader(dset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last)
            self.loaders[key] = loader
            self.iterators[key] = iter(loader)
            if loader_length == 'max':
                self.length = max(self.length, len(loader))
            elif loader_length == 'sum':
                self.length += len(loader)
                
        if isinstance(loader_length, int):
            self.length = loader_length
      
    def keys(self):
        return self.loaders.keys()
    
    def get_next_batch(self, dset):
        batch = next(self.iterators[dset], None)
        if batch is None:
            self.iterators[dset] = iter(self.loaders[dset])
            batch = next(self.iterators[dset], None)
        return batch
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        if not self.all_at_once:
            self.iter_state = 0
        self.iter_count = 0
        return self
        
    def __next__(self):
        if self.iter_count == self.length:
            raise StopIteration
        
        batches = None
        if self.all_at_once:
            batches = {}
            for key in self.dset_dict:
                batches[key] = self.get_next_batch(key)
        else:
            key = self.dset_dict.keys()[self.iter_state]
            batch = self.get_next_batch(key)
            self.iter_state = (self.iter_state + 1)%len(self.dset_dict)
            batches = {key:batch}
        
        self.iter_count += 1
        return batches

def test_multi_loader():
    path0 = './models/autoencoder/cifar_sorted.pth'
    path1 = './models/autoencoder/cifar_sorted_stylegan.pth'
    
    cifar_1 = Sorted_Dataset(path0, train=True, include_keys=['images'], include_labels=[1])
    cifar_rest = Sorted_Dataset(path0, train=True, include_keys=['images'], include_labels=[0,2,3,4,5,6,7,8,9])
    
    stylegan = Sorted_Dataset(path1, train = True, include_keys=['z_values', 'images'], include_labels = [1])
    
    
    loader = Multi_Dataset_Loader({'cifar_1':cifar_1, 'cifar_rest':cifar_rest, 'stylegan':stylegan}, shuffle=True)
    print(len(loader))
    batches = next(iter(loader))
    print(batches.keys())
    for key in batches:
        print(key)
        for item in batches[key]:
            print(item.size())
        print(batches[key][-1])
        
    for i, batch in enumerate(loader):
        if i > 3:
            sys.exit()
        for key in batch:
            print(key)
            for item in batch[key]:
                if len(item.size()) == 4:
                    view_tensor_images(item)


def test_stylegan_norm():
    path0 = './models/autoencoder/cifar_sorted_stylegan.pth'
    path1 = './models/autoencoder/cifar_stylegan_samples_renorm.pth'
    
    style0 = Sorted_Dataset(path0, train=True, include_keys=['images'], include_labels=[1])
    stylenorm = Sorted_Dataset(path1, train=True, include_keys=['images'], include_labels=[1])
    
    
    
    loader = Multi_Dataset_Loader({'clamped':style0, 'normed':stylenorm}, shuffle=False)
    print(len(loader))
        
    for i, batch in enumerate(loader):
        if i > 3:
            sys.exit()
        for key in batch:
            print(key)
            for item in batch[key]:
                if len(item.size()) == 4:
                    view_tensor_images(item)                    
                    
    
def get_dataloaders(cfg, stage):
    # Load all of cifar, optionally priveliging real_classes (None or a tuple)
    # mode=None, real=None, fake=None, real_classes=None
    
    if stage in cfg:
        print("Getting dataloaders for stage", stage)
        cfg = cfg[stage]
        
    mode = cfg.mode
    print("...dataloader mode", mode)
    
    if mode == 'threeway':
        cfg.real
        cfg.fake
        cfg.augmented
        cfg.real_classes
        cfg.fake_classes
        cfg.augmented_classes
        
        real_dset = Sorted_Dataset(cfg.real, train=True, include_keys=['images'], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=True, include_keys=['images'], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=True, include_keys=['images'], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=128, shuffle=True, drop_last=True)
        
        return dataloader
    
    elif mode == 'phi_training':
        pass #?? how to change for new training procedure?
    

    elif mode == 'extract_probs':
        pass
#     real_class_0 = torch.load('./models/autoencoder/cifar_real_class_0.pth')['images'].cpu()
#     real_class_0 = real_class_0*2-1
#     real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images'].cpu()
#     real_class_1 = real_class_1*2-1
    
    
#     class_0_dataloader = torch.utils.data.DataLoader(real_class_0,
#                                          batch_size=256,
#                                          shuffle=False,
#                                          drop_last=False)
#     class_1_dataloader = torch.utils.data.DataLoader(real_class_1,
#                                      batch_size=256,
#                                      shuffle=False,
#                                      drop_last=False)
 

#     fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images'].cpu()
#     fake_class_1 = fake_class_1[:real_class_0.size(0)]
#     fake_class_1_dataloader = torch.utils.data.DataLoader(fake_class_1,
#                                      batch_size=256,
#                                      shuffle=False,
#                                      drop_last=False)
#     print(real_class_0[0].min(), real_class_1[0].min(), fake_class_1[0].min())
    
    #fake_class_1_codes = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['z_values']

    
    elif mode == 'visualize':
        pass
    ##### Fake batches ########
#     fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images']
#     fake_class_1_codes = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['z_values']    
#     fake_dataloader = torch.utils.data.DataLoader(fake_class_1_codes,
#                                          batch_size=64,
#                                          shuffle=False,
#                                          drop_last=False)
#     fake_image_dataloader = torch.utils.data.DataLoader(fake_class_1,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)    

#     fake_batch = next(iter(fake_dataloader)).cpu()
#     fake_image_batch = next(iter(fake_image_dataloader)).cpu()
    
#     fake_batches = [("fake_class_1", fake_batch, fake_image_batch)]
    
#     ##### Real batches ########
#     real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
#     real_class_1 = real_class_1*2-1    
#     real_dataloader_1 = torch.utils.data.DataLoader(real_class_1,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)    
    
#     real_class_0 = torch.load('./models/autoencoder/cifar_real_class_0.pth')['images']
#     real_class_0 = real_class_0*2-1    
#     real_dataloader_0 = torch.utils.data.DataLoader(real_class_0,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)
#     real_batch_class_0 = next(iter(real_dataloader_0)).cpu()
#     real_batch_class_1 = next(iter(real_dataloader_1)).cpu()
    
#     real_batches = [("real_class_0",real_batch_class_0), ("real_class_1", real_batch_class_1)]
   
    

    
    
    
    
    
    
    
    
    
    


    
################################################################################
################  Model building, training, applying, and visualizing
################################################################################     

    
def view_tensor_images(t, scale=True):
    if scale:
        t = ((t+1)/2).clamp(0,1)
    grid = tv.utils.make_grid(t.detach().cpu())
    grid = grid.permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()

    
    
def build_and_train_autoencoder(model_path, model_name_prefix, autoencoder_cfg, data_cfg, train_cfg):
    model = Autoencoder_Model(**autoencoder_cfg)
    dataloader = get_dataloaders(data_cfg, 'ae_stage')
    
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    
    train_autoencoder(model, dataloader, **train_cfg.ae_stage)

    pickle.dump(model, open(model_fullpath, 'wb'))


def visualize_autoencoder(model_path, model_name_prefix, data_cfg):
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    model = pickle.load(open(model_fullpath,'rb'))
    
    dataloader = get_dataloaders(data_cfg, 'ae_stage')
    for i, batch in enumerate(dataloader):
        for key in batch:
            print(key)
            print("original")
            ims = batch[key][0][:64]
            view_tensor_images(ims)
            print("reconstructed")
            view_tensor_images(torch.tanh(model.decoder(model.encoder(ims.cuda()))))
        if i > 2:
            break
    
    
def build_and_train_invertible(model_path, model_name_prefix, phi_cfg, data_cfg, train_cfg):
    model = Phi_Model(**phi_cfg)
    
    #model = pickle.load(open('./models/autoencoder/ae_phi_model_second_recon_adv.pkl','rb'))[1]
    
#     data = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
#     dataset = torch.utils.data.TensorDataset(data['z_values'].cuda(), data['encodings_exp_4'].cuda())
#     print(len(dataset))
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                          batch_size=256,
#                                          shuffle=True,
#                                          drop_last=True)
    
    dataloader = get_dataloaders(data_cfg, 'phi_stage')
    
    model_fullpath = os.path.join(model_path, model_name_prefix+'_phi.pkl')
    train_invertible(model, dataloader, **train_cfg.phi_stage)
    
    
    pickle.dump(model, open(model_fullpath,'wb'))
 
    
def encode_samples(model_path, model_name_prefix, data_cfg):
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    model = pickle.load(open(model_fullpath,'rb'))

    
    samples = torch.load(data_cfg.encode_stage.fake_sample_file)
    
#     samples = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
#     print(samples.keys())
#     #sys.exit()
    
    ims = samples['images']
    
    
    fake_dataloader = torch.utils.data.DataLoader(ims,
                                     batch_size=128,
                                     shuffle=False,
                                     drop_last=False)
    
    model.encoder.eval()
    all_samples = []
    for i,batch in enumerate(fake_dataloader):
        if i%10==0:
            print(i)
        batch = batch.cuda()
        encodings = model.encoder(batch).detach().cpu()
        all_samples.append(encodings)
        
    samples['encodings_'+model_name_prefix] = torch.cat(all_samples)
    print(samples.keys())
    
    torch.save(samples, data_cfg.encode_stage.fake_sample_file)

def extract_probabilities(model_path, model_name_prefix, data_cfg):
#     real_class_0 = torch.load('./models/autoencoder/cifar_real_class_0.pth')['images'].cpu()
#     real_class_0 = real_class_0*2-1
#     real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images'].cpu()
#     real_class_1 = real_class_1*2-1
    
    
#     class_0_dataloader = torch.utils.data.DataLoader(real_class_0,
#                                          batch_size=256,
#                                          shuffle=False,
#                                          drop_last=False)
#     class_1_dataloader = torch.utils.data.DataLoader(real_class_1,
#                                      batch_size=256,
#                                      shuffle=False,
#                                      drop_last=False)
 

#     fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images'].cpu()
#     fake_class_1 = fake_class_1[:real_class_0.size(0)]
#     fake_class_1_dataloader = torch.utils.data.DataLoader(fake_class_1,
#                                      batch_size=256,
#                                      shuffle=False,
#                                      drop_last=False)
#     print(real_class_0[0].min(), real_class_1[0].min(), fake_class_1[0].min())
    
    #fake_class_1_codes = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['z_values']
    
    
    multi_loader = get_dataloaders(data_cfg, 'prob_stage')
    
    
    phi_model_fullpath = os.path.join(model_path, model_name_prefix+'_phi.pkl')
    ae_model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    
    phi_model = pickle.load(open(model_fullpath,'rb'))    
    model = pickle.load(open(ae_model_fullpath,'rb'))
    model.encoder.eval()
    model.decoder.eval()
    phi_model.e2z.eval()
    phi_model.z2e.eval()
    
    

    from model_functions import jacobian, log_priors

    
    stats = edict()
    #for name, dataloader in [("class_0", class_0_dataloader), ("class_1", class_1_dataloader),("fake_class_1",fake_class_1_dataloader)]:
        
    for name in multi_loader.keys():
        dataloader = multi_loader.loaders[name]
        
        print(name)
        e_codes = []
        e_differences = []
        e_norms = []
        logpriors = []
        z2e_jacobian_probs = []
        e2z_jacobian_probs = []
        total_z2e_probs = []
        total_e2z_probs = []
        for i, batch in enumerate(dataloader):
            print("  {0} of {1}".format(i,len(dataloader)))
            batch = batch.cuda()
            e_c = model.encoder(batch).detach()
            e_codes.append(e_c.cpu())
            
            e_c.requires_grad_(True)
            z_predicted = phi_model.e2z(e_c)

            forward_jacobians = -jacobian(e_c, z_predicted).detach().cpu()
            e2z_jacobian_probs.append(forward_jacobians)

            
            z_predicted = z_predicted.detach()
            z_log_priors = log_priors(z_predicted.cpu())
            logpriors.append(z_log_priors)
            
            z_predicted.requires_grad_(True)
            e_reconstructed = phi_model.z2e(z_predicted)
            inv_jacobians = jacobian(z_predicted, e_reconstructed).detach().cpu()
            z2e_jacobian_probs.append(inv_jacobians)
            
            diffs = (e_c - e_reconstructed).detach().cpu()
            e_differences.append(diffs)
            e_norms.append(torch.norm(diffs,dim=1))
            
            
            total_z2e_probs.append(z_log_priors+inv_jacobians)
            total_e2z_probs.append(z_log_priors+forward_jacobians)
        
        stats[name] = edict()
        stats[name].e_codes = torch.cat(e_codes)
        stats[name].e_differences = torch.cat(e_differences)
        stats[name].e_norms = torch.cat(e_norms)
        stats[name].log_priors = torch.cat(logpriors)
        stats[name].z2e_jacobian_probs = torch.cat(z2e_jacobian_probs)
        stats[name].e2z_jacobian_probs = torch.cat(e2z_jacobian_probs)
        stats[name].total_z2e_probs = torch.cat(total_z2e_probs)
        stats[name].total_e2z_probs = torch.cat(total_e2z_probs)
        
        
    
    # 1. Encode the real data, detach encodings from graph
    # 2. Run encodings through e2z and z2e, get logprobs and log priors
    # 3. Plot (3 graphs: jacobian dets, priors, and combined)
    
    print(stats.keys())
    for key in stats:
        print(stats[key].keys())
        
    save_path = os.path.join(model_path, model_name_prefix+'_extracted.pkl')
    pickle.dump(stats, open(save_path,'wb'))    
        
    
def visualize_model(model_path, model_name_prefix, data_cfg):
    ##### Fake batches ########
#     fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images']
#     fake_class_1_codes = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['z_values']    
#     fake_dataloader = torch.utils.data.DataLoader(fake_class_1_codes,
#                                          batch_size=64,
#                                          shuffle=False,
#                                          drop_last=False)
#     fake_image_dataloader = torch.utils.data.DataLoader(fake_class_1,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)    

#     fake_batch = next(iter(fake_dataloader)).cpu()
#     fake_image_batch = next(iter(fake_image_dataloader)).cpu()
    
#     fake_batches = [("fake_class_1", fake_batch, fake_image_batch)]
    
#     ##### Real batches ########
#     real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
#     real_class_1 = real_class_1*2-1    
#     real_dataloader_1 = torch.utils.data.DataLoader(real_class_1,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)    
    
#     real_class_0 = torch.load('./models/autoencoder/cifar_real_class_0.pth')['images']
#     real_class_0 = real_class_0*2-1    
#     real_dataloader_0 = torch.utils.data.DataLoader(real_class_0,
#                                      batch_size=64,
#                                      shuffle=False,
#                                      drop_last=False)
#     real_batch_class_0 = next(iter(real_dataloader_0)).cpu()
#     real_batch_class_1 = next(iter(real_dataloader_1)).cpu()
    
#     real_batches = [("real_class_0",real_batch_class_0), ("real_class_1", real_batch_class_1)]

    fake_code_loader, fake_image_loader, real_multi_loader = get_dataloaders(data_cfg, 'visualize_stage')
    fake_code_batch = next(iter(fake_code_dataloader)).cpu()
    fake_image_batch = next(iter(fake_image_dataloader)).cpu()
    fake_batches = [("fake", fake_code_batch, fake_image_batch)]
    
    real_batches = []
    for key in real_multi_loader.keys():
        batch = real_multi_loader.get_next_batch(key).cpu()
        real_batches.append( (key, batch) )
    
    
    
    ###### Models ########
    from models import load_torch_class
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()    
    phi_model = pickle.load(open('./models/autoencoder/phi_model_exp_4.pkl','rb')) #note exp2
    model = pickle.load(open('./models/autoencoder/ae_model_exp_4.pkl','rb'))
    
    model.encoder.eval()
    model.decoder.eval()
    phi_model.e2z.eval()
    phi_model.z2e.eval()
    cifar_stylegan_net.eval()
    
    with torch.no_grad():
        ##### Loop over fake batches #####
        for name, z_codes, fake_ims in fake_batches:

            
            reconstructed_fake = model.decoder(model.encoder(fake_ims.cuda())).detach().cpu()
            fake_encoded = phi_model.z2e(fake_batch.cuda()).detach()
            fake2real = model.decoder(fake_encoded).detach().cpu()
            
            print(name, "original")
            view_tensor_images(fake_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(reconstructed_fake)
            print(name, "z_codes decoded via autoencoder")
            view_tensor_images(fake2real)
            
            # Store these somehow

        ##### Loop over real batches #####
        for name, real_ims in real_batches:
            # Stylegan beauracracy
            class_constant = torch.zeros(10, device='cuda')
            class_constant[1] = 1
            classes = class_constant.repeat(real_ims.size(0),1)
            
            reconstructed_real = model.decoder(model.encoder(real_ims.cuda())).detach().cpu()
            real_encoded = model.encoder(real_ims.cuda())
            real2stylegan_w = cifar_stylegan_net.mapping(phi_model.e2z(real_encoded), classes)
            real2stylegan = cifar_stylegan_net.synthesis(real2stylegan_w, noise_mode='const', force_fp32=True)
            real2stylegan = real2stylegan.detach().cpu()
            
            print(name, "original")
            view_tensor_images(real_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(reconstructed_real)
            print(name, "Encodings decoded via stylegan")
            view_tensor_images(real2stylegan)            


    

    
    
def view_extracted_probabilities(model_path, model_name_prefix, data_cfg):
    # note the off-manifold scores are norms here, not squared norms
    
    #data = pickle.load(open('./models/autoencoder/extracted_info_exp_4.pkl', 'rb'))
    data = pickle.load(open(data_cfg.plot_stage.prob_sample_file,'rb'))
    
    
    plt.title("logpriors")
    plt.hist(data.class_1.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()
    
    plt.title("e2z jacobians")
    plt.hist(data.class_1.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()    
    
    
    plt.title("e2z combined")
    plt.hist(data.class_1.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()       
    
    plt.title("z2e jacobians")
    plt.hist(data.class_1.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()     
    
    
    plt.title("z2e combined")
    plt.hist(data.class_1.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()     
    
    plt.title("difference norms")
    plt.hist(data.class_1.e_norms.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake_class_1.e_norms.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.class_0.e_norms.numpy(), bins=50, density=True, alpha=0.3, label="Real airplanes")
    plt.legend()
    plt.show()      
    
    
    
    
    
    
    
    
    
################################################################################
################  Test and experiment functions
################################################################################    
    
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
    
def test4():
    dset = two_domain_dataset()
    print(len(dset))
    
    for i in range(10):
        x, y = dset[i]
        print(y, x.size(), x.min(), x.max())


        

        
        
        
        

    
    

    
def dataset_config(key, dataset_directory, model_path, model_name_prefix):
    dset_config = edict(
        {
            'cifar_1_all': {
                'ae_stage': {
                    'mode': 'threeway',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9]
                },
                'encode_stage': {
                    'mode': 'encode',
                    'fake_sample_file': os.path.join(model_path, 'cifar_class_1_generated.pth'),
                },
                'phi_stage': {
                    'mode': 'cifar_all_phi',
                    'real': dataset_directory, # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_class_1_generated.pth'), # preprocessed fake data
                    'fake_encoding_key': 'encodings_' + model_name_prefix,
                    'real_classes':[ 1 ]                                   
                },
                'prob_stage': {
                      
                },
                'visualize_stage': {
                    
                },
                'plot_stage': {
                    
                },
            }
        }
    )
    
    return dset_config[key]


def train_config(key):
    train_cfg = edict(
        {
            'cifar_1_all': {
                'ae_stage': {
                    'n_epochs':30,
                    'use_adversary':True
                },
                'phi_stage': {
                    'n_epochs':200,
                    
                }
            }
        }
    )
    
    return train_cfg[key]



def autoencoder_config(key):
    cfg = {
        'linear_ae': {
            'input_size': 32,
            'linear': True,
            'very_lean': True,
            'use_adversary': True
        }
    }
    
    return cfg[key]


def phi_config(key):
    cfg = {
        'linear_ae': {}
    }
    
    return cfg[key]


def run_experiment(model_path, model_name_prefix, autoencoder_cfg, phi_cfg, data_cfg, train_cfg):

    build_and_train_autoencoder(model_path, model_name_prefix, autoencoder_cfg, data_cfg, train_cfg)
    
    
    #encode_samples(model_path, model_name_prefix, data_cfg)    
    #build_and_train_invertible(model_path, model_name_prefix, phi_cfg, data_cfg, train_cfg)
    #extract_probabilities(model_path, model_name_prefix, data_cfg)

    
    
    
def visualize_experiment(model_path, model_name_prefix, data_cfg):
    view_extracted_probabilities(model_path, model_name_prefix, data_cfg)    
    visualize_model(model_path, model_name_prefix, data_cfg)
    
    

##
# New pressure loss function (to be moved upward)
def pressure_loss(x, lmbda, dim=512, eps = 0.1):
    numerator = np.sqrt(dim)
    norms = torch.norm(x,dim=1)
    losses = numerator / (norms + eps)
    total_loss = lmbda*losses.mean()
    return total_loss

##
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='run')
    parser.add_argument('--save_directory', type=str, default='./models/autoencoder')
    parser.add_argument('--dataset_directory', type=str, default='/mnt/linuxshared/phd-research/data/standard_datasets')
    parser.add_argument('--experiment_prefix', type=str, default='linear_ae_model')
    parser.add_argument('--experiment_suffix', type=str, default=None)
    parser.add_argument('--experiment_number', type=int, default=None)
    parser.add_argument('--model_name_prefix', type=str, default=None)
    parser.add_argument('--autoencoder_config_key', default='linear_ae')
    parser.add_argument('--phi_config_key', default='linear_ae')
    parser.add_argument('--dataset_config_key', default='cifar_1_all')
    parser.add_argument('--train_config_key', default='cifar_1_all')
    opt = parser.parse_args()
    
    if opt.experiment_suffix is None:
        if opt.experiment_number is None:
            opt.experiment_suffix = ''
        else:
            opt.experiment_suffix = '_exp_'+str(opt.experiment_number)
    
    if opt.model_name_prefix is None:
        opt.model_name_prefix = opt.experiment_prefix + opt.experiment_suffix
    
    data_cfg = dataset_config(opt.dataset_config_key, opt.dataset_directory, opt.save_directory, opt.model_name_prefix)
    autoencoder_cfg = autoencoder_config(opt.autoencoder_config_key)
    phi_cfg = phi_config(opt.phi_config_key)
    train_cfg = train_config(opt.train_config_key)
    
    
    if opt.mode == 'run':
        run_experiment(opt.save_directory, opt.model_name_prefix, autoencoder_cfg, phi_cfg, data_cfg, train_cfg)
    elif opt.mode == 'visualize_ae':
        visualize_autoencoder(opt.save_directory, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'visualize':
        visualize_experiment(opt.save_directory, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'preprocess_cifar':
        preprocess_cifar(opt.dataset_directory, opt.save_directory)
    elif opt.mode == 'sample_stylegan':
        sample_cifar_stylegan(opt.save_directory)
    elif opt.mode == 'test_sorted':
        test_sorted_dataset()
    elif opt.mode == 'test_multi':
        test_multi_loader()
    elif opt.mode == 'test_norm':
        test_stylegan_norm()    
# Next to do:
#  - Finish redoing the training procedures
#  - Finish implementing the get_loaders function and make sure it works with everything
#  - Train with the new method
    
# Notes:
#  is clamping the stylegan outputs a bad idea? Perhaps rescale them instead?
#  also, add noise to all the images and rescale?
    
    
    
    
    
