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
    
    cifar_sorted = {i:{'train':{'images':[]}, 'test':{'images':[]}} for i in range(10)}
    for i in range(len(cifar_train)):
        x, y = cifar_train[i]
        cifar_sorted[y]['train']['images'].append(x)
    for i in range(len(cifar_test)):
        x, y = cifar_test[i]
        cifar_sorted[y]['test']['images'].append(x)        
        
        
    for key in cifar_sorted:
        cifar_sorted[key]['train']['images'] = torch.stack(cifar_sorted[key]['train']['images'])
        cifar_sorted[key]['test']['images'] = torch.stack(cifar_sorted[key]['test']['images'])
        
    torch.save(cifar_sorted, os.path.join(save_dir, 'cifar_sorted.pth'))


    
def sample_cifar_stylegan(savedir, stylegan_file):
    #encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
    
    from models import load_torch_class
    
    #cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= stylegan_file).cuda()
    

    
    
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
        
    torch.save(stylegan_sorted, os.path.join(savedir, 'cifar_sorted_stylegan.pth'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
                    
    
def get_dataloaders(cfg, stage, include_keys = None, shuffle=None):
    # Load all of cifar, optionally priveliging real_classes (None or a tuple)
    # mode=None, real=None, fake=None, real_classes=None
    

    if stage in cfg:
        print("Getting dataloaders for stage", stage)
        cfg = cfg[stage]

    if include_keys is not None:
        cfg.data_keys = include_keys

    mode = cfg.mode
    print("...dataloader mode", mode)
    
    if mode == 'threeway':
#         cfg.real
#         cfg.fake
#         cfg.augmented
#         cfg.real_classes
#         cfg.fake_classes
#         cfg.augmented_classes
        
        real_dset = Sorted_Dataset(cfg.real, train=True, include_keys=cfg.data_keys, include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=True, include_keys=cfg.data_keys, include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=True, include_keys=cfg.data_keys, include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}

        shuffle_threeway = True
        if shuffle is not None:
            print("Manual shuffling spec")
            shuffle_threeway = shuffle
        dataloader = Multi_Dataset_Loader(dsets, batch_size=128, shuffle=shuffle_threeway, drop_last=True)
        
        return dataloader
    
    elif mode == 'threeway_encodings':
        real_dset = Sorted_Dataset(cfg.real, train=True, include_keys=[cfg.encoding_key], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=True, include_keys=['z_values', cfg.encoding_key], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=True, include_keys=[cfg.encoding_key], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=128, shuffle=True, drop_last=True)
        return dataloader

    elif mode == 'threeway_combined':
        print("augmented classes", cfg.augmented_classes)
        real_dset = Sorted_Dataset(cfg.real, train=True, include_keys=cfg.data_keys, include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=True, include_keys=['z_values']+cfg.data_keys, include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=True, include_keys=cfg.data_keys, include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}

        shuffle_threeway = True
        if shuffle is not None:
            print("Manual shuffling spec")
            shuffle_threeway = shuffle
        dataloader = Multi_Dataset_Loader(dsets, batch_size=128, shuffle=shuffle_threeway, drop_last=True)
        
        return dataloader


    elif mode == 'extract_probs':
        real_dset = Sorted_Dataset(cfg.real, train=False, include_keys=[cfg.encoding_key], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=False, include_keys=[cfg.encoding_key], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=False, include_keys=[cfg.encoding_key], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=64, shuffle=False, drop_last=False)
        return dataloader

    elif mode == 'extract_probs_combined':
        real_dset = Sorted_Dataset(cfg.real, train=False, include_keys=['images','encodings_vgg16'], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=False, include_keys=['images','encodings_vgg16'], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=False, include_keys=['images','encodings_vgg16'], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=64, shuffle=False, drop_last=False)
        return dataloader


    
    elif mode == 'visualization':
        real_dset = Sorted_Dataset(cfg.real, train=False, include_keys=[cfg.encoding_key, 'images'], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=False, include_keys=['z_values', cfg.encoding_key, 'images'], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=False, include_keys=[cfg.encoding_key, 'images'], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=64, shuffle=False, drop_last=False)
        return dataloader        

    elif mode == 'visualize_combined':
        real_dset = Sorted_Dataset(cfg.real, train=False, include_keys=['images', 'encodings_vgg16'], include_labels=cfg.real_classes)
        fake_dset = Sorted_Dataset(cfg.fake, train=False, include_keys=['z_values', 'images', 'encodings_vgg16'], include_labels=cfg.fake_classes)
        aug_dset = Sorted_Dataset(cfg.augmented, train=False, include_keys=['images', 'encodings_vgg16'], include_labels=cfg.augmented_classes)
        dsets = {'real':real_dset, 'fake':fake_dset, 'augmented':aug_dset}
        dataloader = Multi_Dataset_Loader(dsets, batch_size=64, shuffle=False, drop_last=False)
        return dataloader        
       
