import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


from utils import EasyDict as edict


"""
out+x --> nonlinearity --> (batch norm) --> conv layer --> nonlin --> (bn) --> conv --> out+x

"""

# /scratch0/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl


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
        
  
class Domain_adversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512,512)
        self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)
        self.layer2 = nn.Linear(512,1)
        
    def forward(self, x):
        return self.layer2(self.nonlinearity(self.layer1(x))).squeeze(1)

    
    
class Autoencoder_Model:
    def __init__(self, input_size=32):
        self.encoder = Encoder(input_size, very_lean=False).cuda()
        self.decoder = Decoder(input_size, very_lean=False).cuda()
        self.domain_adversary = Domain_adversary().cuda()
        
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters())
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters())
        self.domain_adversary_optim = torch.optim.Adam(self.domain_adversary.parameters())

class Phi_Model:
    def __init__(self):
        self.e2z = Phi().cuda()
        self.z2e = Phi().cuda()
        
        self.e2z_optim = torch.optim.Adam(self.e2z.parameters(),lr=0.0001)
        self.z2e_optim = torch.optim.Adam(self.z2e.parameters(), lr=0.0001)
    
def train_autoencoder(model, dataloader, n_epochs):
    #reconstruction_loss = torch.nn.BCEWithLogitsLoss()
    reconstruction_loss = torch.nn.MSELoss()
    #reconstruction_loss = torch.nn.L1Loss()
    adversary_loss = torch.nn.BCEWithLogitsLoss()
    
    model.encoder.train()
    model.decoder.train()
    model.domain_adversary.train()
    
    phase = 0
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        for i, batch in enumerate(dataloader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            
            z = model.encoder(x)
            noise = 0.0001*torch.randn(z.size(), device=z.device)
            z = z+noise
            
            loss_info = None
            if phase == 0:
                # train encoder and decoder


                
                predicted_domains = model.domain_adversary(z)
                reconstruction = torch.sigmoid(model.decoder(z))
                
                neg_adv_loss = -0.05*adversary_loss(predicted_domains,y)
                recon_loss = reconstruction_loss(reconstruction, x)
                regularizing_loss = 0.001*(torch.norm(z.view(z.size(0),-1),dim=1)**2).mean()
                total_loss = recon_loss+neg_adv_loss+regularizing_loss
                #total_loss = recon_loss#+regularizing_loss
                
                model.encoder.zero_grad()
                model.decoder.zero_grad()
                model.domain_adversary.zero_grad()
                total_loss.backward()
                model.encoder_optim.step()
                model.decoder_optim.step()
                
                lossinfo = "recon loss = {0}, adv loss = {1}".format(recon_loss.item(), -neg_adv_loss.item())
                #lossinfo = "recon loss = {0}".format(recon_loss.item())
                
                
            if phase == 1:
                predicted_domains = model.domain_adversary(z)
                adv_loss = adversary_loss(predicted_domains,y)
                
                model.encoder.zero_grad()
                model.decoder.zero_grad()                
                model.domain_adversary.zero_grad()
                adv_loss.backward()
                model.domain_adversary_optim.step()
    
                lossinfo = "adv loss = {0}".format(-neg_adv_loss.item())
    
    
            if i%50 == 0:
                print(i, lossinfo)
    
            phase = 1-phase
    
    
    
    pickle.dump(model, open('./models/autoencoder/ae_model.pkl','wb'))
    
    
    
    
    
    
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
    images = ((images+1)/2).clamp(0,1)
    fake_labels = torch.ones(len(images))
    fake_dataset = torch.utils.data.TensorDataset(images, fake_labels)
    
    im_transform = tv.transforms.ToTensor()
    real_data = tv.datasets.CIFAR10('/scratch0/datasets', train=True, transform=im_transform,
                                    target_transform=None, download=True)
    
#     filtered_real = []
#     for pt in real_data:
#         x,y = pt
#         if y == 1:
#             filtered_real.append(x)
            
#     real_class_1 = torch.stack(filtered_real)
    
#     torch.save({'images':real_class_1}, './models/autoencoder/cifar_real_class_1.pth')
    
    real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
    real_labels = torch.zeros(len(real_class_1))
    real_dataset = torch.utils.data.TensorDataset(real_class_1, real_labels)
    
    
    
    combined_dataset = BlendedDataset(real_dataset, fake_dataset)
    
    return combined_dataset
    
    
def build_and_train_autoencoder():
    dataset = two_domain_dataset()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128,
                                             shuffle=True,
                                             drop_last=True)
    
    model = Autoencoder_Model()
    print(model.encoder)
    print(model.decoder)
    print(model.domain_adversary)
    train_autoencoder(model, dataloader, 200)
    
    
    
def view_tensor_images(t):
    grid = tv.utils.make_grid(t.detach().cpu())
    grid = grid.permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()
    
def visualize_autoencoder():
    fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images']
    fake_class_1 = ((fake_class_1+1)/2).clamp(0,1)
    real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
    fake_dataloader = torch.utils.data.DataLoader(fake_class_1,
                                         batch_size=64,
                                         shuffle=False,
                                         drop_last=False)
    real_dataloader = torch.utils.data.DataLoader(real_class_1,
                                     batch_size=64,
                                     shuffle=False,
                                     drop_last=False)
    model = pickle.load(open('./models/autoencoder/ae_model.pkl','rb'))
    model.encoder.eval()
    model.decoder.eval()
    
    fake_batch = next(iter(fake_dataloader))
    real_batch = next(iter(real_dataloader))
    
    reconstructed_fake = torch.sigmoid(model.decoder(model.encoder(fake_batch.cuda())))
    reconstructed_real = torch.sigmoid(model.decoder(model.encoder(real_batch.cuda())))
    
    print("fake")
    view_tensor_images(fake_batch)
    print("reconstructed fake")
    view_tensor_images(reconstructed_fake)
    print("real")
    view_tensor_images(real_batch)
    print("reconstructed real")
    view_tensor_images(reconstructed_real)

    
    
def encode_samples():
    model = pickle.load(open('./models/autoencoder/ae_model.pkl','rb'))
    samples = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
    print(samples.keys())
    #sys.exit()
    
    ims = samples['images']
    ims = ((ims+1)/2).clamp(0,1) # Maybe clamping isn't the best... investigate in the future
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
        
    samples['encodings'] = torch.cat(all_samples)
    print(samples.keys())
    
    torch.save(samples, './models/autoencoder/cifar_class_1_generated.pth')
    

def train_invertible(model, dataloader, n_epochs):
    lossfunc = torch.nn.MSELoss()
    lmbda = 0.25
    
    
    model.e2z.train()
    model.z2e.train()
    
    alternator = 0
    for n in range(n_epochs):
        print("Epoch", n)
        for i, batch in enumerate(dataloader):
            z_batch, e_batch = batch
            z_batch = z_batch.cuda()
            e_batch = e_batch.cuda()
            
            if alternator == 0:
                predicted_encodings = model.z2e(z_batch)
                cycled_z = model.e2z(predicted_encodings)
                loss = lossfunc(predicted_encodings, e_batch)+lmbda*lossfunc(cycled_z, z_batch)
                
                model.z2e.zero_grad()
                model.e2z.zero_grad()
                loss.backward()
                model.z2e_optim.step()
                
            else:
                predicted_z = model.e2z(e_batch)
                cycled_e = model.z2e(predicted_z)
                loss = lossfunc(predicted_z, z_batch) + lmbda*lossfunc(cycled_e, e_batch)
                
                model.e2z.zero_grad()
                model.z2e.zero_grad()
                loss.backward()
                model.e2z_optim.step()


            if i>0 and i%100 == 0 or i%101 == 0:
                print(i, alternator, loss.item())
                
            alternator = 1-alternator
            
    pickle.dump(model, open('./models/autoencoder/phi_model.pkl','wb'))
    
    

def build_and_train_invertible():
    model = Phi_Model()
    data = torch.load('./models/autoencoder/cifar_class_1_generated.pth')
    dataset = torch.utils.data.TensorDataset(data['z_values'].cuda(), data['encodings'].cuda())
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=256,
                                         shuffle=True,
                                         drop_last=True)
    train_invertible(model, dataloader, 100)
    
    #print(e2z)

    
def visualize_invertible():
    fake_class_1 = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['images']
    fake_class_1_codes = torch.load('./models/autoencoder/cifar_class_1_generated.pth')['z_values']
    fake_class_1 = ((fake_class_1+1)/2).clamp(0,1)
    real_class_1 = torch.load('./models/autoencoder/cifar_real_class_1.pth')['images']
    fake_dataloader = torch.utils.data.DataLoader(fake_class_1_codes,
                                         batch_size=64,
                                         shuffle=False,
                                         drop_last=False)
    fake_image_dataloader = torch.utils.data.DataLoader(fake_class_1,
                                     batch_size=64,
                                     shuffle=False,
                                     drop_last=False)
    real_dataloader = torch.utils.data.DataLoader(real_class_1,
                                     batch_size=64,
                                     shuffle=False,
                                     drop_last=False)
    
    encoder_model = pickle.load(open('./models/autoencoder/ae_model.pkl','rb'))
    phi_model = pickle.load(open('./models/autoencoder/phi_model.pkl','rb'))
    
    
    with torch.no_grad():
        encoder_model.encoder.eval()
        encoder_model.decoder.eval()
        phi_model.e2z.eval()
        phi_model.z2e.eval()

        fake_batch = next(iter(fake_dataloader)).cpu()
        real_batch = next(iter(real_dataloader)).cpu()
        fake_image_batch = next(iter(fake_image_dataloader)).cpu()

        reconstructed_fake = torch.sigmoid(encoder_model.decoder(encoder_model.encoder(fake_image_batch.cuda()))).detach().cpu()
        reconstructed_real = torch.sigmoid(encoder_model.decoder(encoder_model.encoder(real_batch.cuda()))).detach().cpu()



        from models import load_torch_class
        cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/scratch0/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()
        class_constant = torch.zeros(10, device='cuda')
        class_constant[1] = 1
        classes = class_constant.repeat(real_batch.size(0),1)

        real_encoded = encoder_model.encoder(real_batch.cuda()).reshape(real_batch.size(0),-1)
        real2stylegan_w = cifar_stylegan_net.mapping(phi_model.e2z(real_encoded), classes)
        real2stylegan = cifar_stylegan_net.synthesis(real2stylegan_w, noise_mode='const', force_fp32=True)
        real2stylegan = ((real2stylegan+1)/2).clamp(0,1)
        real2stylegan = real2stylegan.detach().cpu()

        fake_encoded = phi_model.z2e(fake_batch.cuda()).reshape(fake_batch.size(0),32,4,4).detach()
        stylegan2real = torch.sigmoid(encoder_model.decoder(fake_encoded)).detach().cpu()


        print("real")
        view_tensor_images(real_batch)
        print("fake")
        view_tensor_images(fake_image_batch)
        print("reconstructed real")
        view_tensor_images(reconstructed_real)
        print("reconstructed fake")
        view_tensor_images(reconstructed_fake)
        print("real2stylegan")
        view_tensor_images(real2stylegan)
        print("stylegan2real")
        view_tensor_images(stylegan2real)
    
# regularize autoencoder with l2 and domain adversary, and maybe add small amounts of noise
        

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
    
    
if __name__ == '__main__':
    #test4()
    #build_and_train_autoencoder()
    #visualize_autoencoder()
    #build_and_train_invertible()
    #encode_samples()
    #build_and_train_invertible()
    visualize_invertible()
        