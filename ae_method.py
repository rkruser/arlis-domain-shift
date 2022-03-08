import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os





from utils import EasyDict as edict



from ae_models import *
from ae_data import *
from ae_features import *

   
################################################################################
################  Model building, training, applying, and visualizing
################################################################################     

    
def view_tensor_images(t, scale=True, resize=False, new_size=224, nrow=8):
    if resize:
        resize_func = tv.transforms.Resize(new_size)
        t = resize_func(t)
    if scale:
        t = ((t+1)/2).clamp(0,1)
    grid = tv.utils.make_grid(t.detach().cpu(), nrow=nrow)
    grid = grid.permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()

    
def test_vgg(model_path, model_name_prefix, data_cfg):
#    from ae_features import obtain_vgg_features

    dataloader = get_dataloaders(data_cfg, 'ae_stage')
    batch_0 = dataloader.get_next_batch('augmented')
    print(len(batch_0))
    print(batch_0[0].size())
    print(batch_0[1].size())
    #view_tensor_images(batch_0)
    #view_tensor_images(batch_0[:36], resize=True, nrow=6)

    feats = obtain_vgg_features(None, data_batch = batch_0[0][:64], device='cuda:1')
    
    print(torch.argmax(feats, dim=1))
    print(torch.unique(torch.argmax(feats,dim=1)))
    print(batch_0[1][:64])
    

    
def build_and_train_autoencoder(model_path, model_name_prefix, autoencoder_cfg, data_cfg, train_cfg):
    model = Autoencoder_Model(**autoencoder_cfg)
    dataloader = get_dataloaders(data_cfg, 'ae_stage')
    
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    
    train_autoencoder(model, dataloader, **train_cfg.ae_stage)

    pickle.dump(model, open(model_fullpath, 'wb'))

def build_and_train_combined_autoencoder(model_path, model_name_prefix, autoencoder_cfg, data_cfg, train_cfg):
    import ae_combined

    model = ae_combined.Combined_Autoencoder(**autoencoder_cfg)
    dataloader = get_dataloaders(data_cfg, 'ae_stage')
    
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    
    ae_combined.train_combined(model, dataloader, **train_cfg.ae_stage)

    pickle.dump(model, open(model_fullpath, 'wb'))


def visualize_autoencoder(model_path, model_name_prefix, data_cfg):
    model_fullpath = os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
    model = pickle.load(open(model_fullpath,'rb'))

    use_features = False
    include_keys = ['images']
    if hasattr(model, 'feature_encode') and model.feature_encode is not None:
        print("Using features")
        use_features = True
        include_keys.append('encodings_vgg16')

    
    dataloader = get_dataloaders(data_cfg, 'ae_stage', include_keys = include_keys, shuffle=False)
    for i, batch in enumerate(dataloader):
        if i > 0:
            break
        for key in batch:
            print(key)
            print("original")
            ims = batch[key][0][:64]
            view_tensor_images(ims)
            print("reconstructed")
            
            ims = ims.cuda()
            encoded = model.encoder(ims)
            if use_features:
                feats = batch[key][1][:64].cuda()
                encoded_feats = model.feature_encode(feats)
                encoded = encoded + encoded_feats
                add_vals, _ = model.feature_decode(encoded)
                encoded = encoded + add_vals

            view_tensor_images(torch.tanh(model.decoder(encoded)))

    
    
def build_and_train_invertible(model_path, model_name_prefix, phi_cfg, data_cfg, train_cfg, load_model=False):
    model = None
    model_fullpath = os.path.join(model_path, model_name_prefix+'_phi.pkl')
    if load_model:
        print("loading model")
        print(model_fullpath)
        model = pickle.load(open(model_fullpath, 'rb'))
    else:
        model = Phi_Model(**phi_cfg)
    
    dataloader = get_dataloaders(data_cfg, 'phi_stage')
    

    train_invertible(model, dataloader, **train_cfg.phi_stage, use_adversary=phi_cfg.get('use_adversary',False),
                     use_friend = phi_cfg.get('use_friend',False))
    
    
    pickle.dump(model, open(model_fullpath,'wb'))
 


# Need to rewrite some of this to deal with how the data files are structured
def encode_samples(model_path, model_name_prefix, data_cfg, encode_vgg=False):
    use_features = False

    model = None
    if encode_vgg:
        print("Applying vgg")
        model = tv.models.vgg16(pretrained=True).to('cuda:1')
        model.eval()
        #from ae_features import apply_vgg
        model_name_prefix = 'vgg16'
    else:
        model = pickle.load(open(data_cfg.encode_stage.model_file,'rb'))
        model.encoder.eval()
        if hasattr(model, 'feature_encode') and model.feature_encode is not None:
            print("Using features")
            use_features=True
    
    # { label0: {train: { z_values:[], images:[], encodings:[] }, test: { z_values, images, encodings},
    #   label1: {train: {...}, test:{...}, ...}
    #    
    

    include_keys = ['images']
    if use_features:
        include_keys.append('encodings_vgg16')
    
    ims_fake = Sorted_Dataset(data_cfg.encode_stage.fake_sample_file, include_keys=include_keys, train=True)
    ims_real = Sorted_Dataset(data_cfg.encode_stage.real_sample_file, include_keys=include_keys, train=True)
    
    ims_fake_test = Sorted_Dataset(data_cfg.encode_stage.fake_sample_file, include_keys=include_keys, train=False)
    ims_real_test = Sorted_Dataset(data_cfg.encode_stage.real_sample_file, include_keys=include_keys, train=False)    
    
    loaders = Multi_Dataset_Loader({'ims_real_train':ims_real, 'ims_real_test':ims_real_test, 
                                    'ims_fake_train':ims_fake, 'ims_fake_test':ims_fake_test},
                                   batch_size=128, shuffle=False, drop_last=False)
    
    
    
    info = {
            'ims_real_train':('train', data_cfg.encode_stage.real_sample_file),
            'ims_real_test':('test', data_cfg.encode_stage.real_sample_file),
            'ims_fake_train':('train', data_cfg.encode_stage.fake_sample_file),
            'ims_fake_test':('test', data_cfg.encode_stage.fake_sample_file)
           }
    
    with torch.no_grad():
        for key in loaders.loaders:
            print(key)


            all_samples = []
            all_labels = []
            for i, batch in enumerate(loaders.loaders[key]):
                if i%10==0:
                    print(i)
                x = batch[0].cuda()
                y = batch[-1]
                
                encodings = None
                if encode_vgg:
                    encodings = apply_vgg(model, x, device='cuda:1').detach().cpu()
                else:
                    encodings = model.encoder(x)
                    if use_features:
                        x_vgg = batch[1].cuda()
                        x_vgg_encoded = model.feature_encode(x_vgg)
                        encodings = encodings + x_vgg_encoded
                        
                    encodings = encodings.detach().cpu()


                all_samples.append(encodings)
                all_labels.append(y)

            all_samples = torch.cat(all_samples)
            all_labels = torch.cat(all_labels)
            print(all_samples.size())
            print(all_labels.size())
            
            

            data_file_contents = torch.load(info[key][1])
            lbls = torch.unique(all_labels)
            print(lbls)
            for lbl in lbls:
                lbl = lbl.item()
                data_file_contents[lbl][info[key][0]]['encodings_'+model_name_prefix] = all_samples[all_labels == lbl]

            print("Saving in", info[key])
            torch.save(data_file_contents, info[key][1])
            
            
            

def extract_probabilities(model_path, model_name_prefix, data_cfg):
    multi_loader = get_dataloaders(data_cfg, 'prob_stage')   
    
    phi_model = pickle.load(open(data_cfg.prob_stage.phi_model_file,'rb'))    
    model = pickle.load(open(data_cfg.prob_stage.model_file,'rb'))
    
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
        #e_codes = []
        e_differences = []
        e_norms = []
        z_norms = []
        logpriors = []
        z2e_jacobian_probs = []
        e2z_jacobian_probs = []
        total_z2e_probs = []
        total_e2z_probs = []
        for i, batch in enumerate(dataloader):
            print("  {0} of {1}".format(i,len(dataloader)))
            e_c, _ = batch
            e_c = e_c.cuda() #e codes
            
            e_c.requires_grad_(True)
            z_predicted = phi_model.e2z(e_c)

            forward_jacobians = -jacobian(e_c, z_predicted).detach().cpu()
            e2z_jacobian_probs.append(forward_jacobians)

            
            z_predicted = z_predicted.detach()
            z_log_priors = log_priors(z_predicted)
            logpriors.append(z_log_priors)
            z_norms.append(torch.norm(z_predicted.cpu(),dim=1))
    
            
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
        #stats[name].e_codes = torch.cat(e_codes)
        stats[name].e_differences = torch.cat(e_differences)
        stats[name].e_norms = torch.cat(e_norms)
        stats[name].log_priors = torch.cat(logpriors)
        stats[name].z2e_jacobian_probs = torch.cat(z2e_jacobian_probs)
        stats[name].e2z_jacobian_probs = torch.cat(e2z_jacobian_probs)
        stats[name].total_z2e_probs = torch.cat(total_z2e_probs)
        stats[name].total_e2z_probs = torch.cat(total_e2z_probs)
        stats[name].z_norms = torch.cat(z_norms)
        
        
    
    # 1. Encode the real data, detach encodings from graph
    # 2. Run encodings through e2z and z2e, get logprobs and log priors
    # 3. Plot (3 graphs: jacobian dets, priors, and combined)
    
    print(stats.keys())
    for key in stats:
        print(stats[key].keys())
        
    save_path = os.path.join(model_path, model_name_prefix+'_extracted.pkl')
    pickle.dump(stats, open(save_path,'wb'))    
        
    
def visualize_model(model_path, model_name_prefix, data_cfg, class_constant_stylegan=1):
    multi_loader = get_dataloaders(data_cfg, 'visualize_stage')
    
    fake_batch = multi_loader.get_next_batch('fake')
    fake_images = fake_batch[2]
    fake_encoded = fake_batch[1]
    fake_z = fake_batch[0]
    fake_batches = [("fake", fake_z, fake_encoded, fake_images)]
    
    real_batches = []
    for key in multi_loader.keys():
        if key != 'fake':
            batch = multi_loader.get_next_batch(key)
            real_ims = batch[1]
            real_encoded = batch[0]
            real_batches.append( (key, real_encoded, real_ims) )
    
    
    
    ###### Models ########
    from models import load_torch_class
#     cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()    
#     phi_model = pickle.load(open('./models/autoencoder/phi_model_exp_4.pkl','rb')) #note exp2
#     model = pickle.load(open('./models/autoencoder/ae_model_exp_4.pkl','rb'))
    
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= data_cfg.visualize_stage.stylegan_file).cuda()    
    phi_model = pickle.load(open(data_cfg.visualize_stage.phi_model_file,'rb')) #note exp2
    model = pickle.load(open(data_cfg.visualize_stage.model_file,'rb'))  
    
    
    
    model.encoder.eval()
    model.decoder.eval()
    phi_model.e2z.eval()
    phi_model.z2e.eval()
    cifar_stylegan_net.eval()
    
    with torch.no_grad():
        ##### Loop over fake batches #####
        for name, z_codes, fake_encoded, fake_ims in fake_batches:

            
            reconstructed_fake = torch.tanh(model.decoder(fake_encoded.cuda())).detach().cpu()
            z2e_codes = phi_model.z2e(z_codes.cuda()).detach()
            fake2real = torch.tanh(model.decoder(z2e_codes)).detach().cpu()
            
            print(name, "original")
            view_tensor_images(fake_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(reconstructed_fake)
            print(name, "z_codes decoded via autoencoder")
            view_tensor_images(fake2real)
            
            # Store these somehow

        ##### Loop over real batches #####
        for name, real_encoded, real_ims in real_batches:
            # Stylegan beauracracy
            class_constant = torch.zeros(10, device='cuda')
            class_constant[class_constant_stylegan] = 1
            classes = class_constant.repeat(real_ims.size(0),1)
            
            real_encoded = real_encoded.cuda()
            reconstructed_real = torch.tanh(model.decoder(real_encoded)).detach().cpu()
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
    plt.hist(data.real.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()
    
    plt.title("e2z jacobians")
    plt.hist(data.real.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()    
    
    
    plt.title("e2z combined")
    plt.hist(data.real.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()       
    
    plt.title("z2e jacobians")
    plt.hist(data.real.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()     
    
    
    plt.title("z2e combined")
    plt.hist(data.real.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()     
    
    plt.title("z norms")
    plt.hist(data.real.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Real aug")
    plt.legend()
    plt.show()      
    
    
    
    
def view_extracted_probabilities_combined(model_path, model_name_prefix, data_cfg, aug_label="Real planes", aug_class=0):
    # note the off-manifold scores are norms here, not squared norms
    
    #data = pickle.load(open('./models/autoencoder/extracted_info_exp_4.pkl', 'rb'))
    data = pickle.load(open(data_cfg.plot_stage.prob_sample_file,'rb'))

    ######### View top and bottom of each ###########3
    real_images = torch.load(data_cfg.visualize_stage.real)[1]['test']['images']
    fake_images = torch.load(data_cfg.visualize_stage.fake)[1]['test']['images']
    aug_images = torch.load(data_cfg.visualize_stage.augmented)[aug_class]['test']['images']

    real_znorms = data['real']['z_norms']
    real_e2z = data['real']['e2z_jacobian_probs']
    real_z2e = data['real']['z2e_jacobian_probs']

    fake_znorms = data['fake']['z_norms']
    fake_e2z = data['fake']['e2z_jacobian_probs']
    fake_z2e = data['fake']['z2e_jacobian_probs']

    aug_znorms = data['augmented']['z_norms']
    aug_e2z = data['augmented']['e2z_jacobian_probs']
    aug_z2e = data['augmented']['z2e_jacobian_probs']

    view_top_and_bottom(real_znorms, real_images, 'Real cars, z norms')
    view_top_and_bottom(real_e2z, real_images, 'Real cars, e2z')
    view_top_and_bottom(real_z2e, real_images, 'Real cars, z2e')

    view_top_and_bottom(fake_znorms, fake_images, 'Fake cars, z norms')
    view_top_and_bottom(fake_e2z, fake_images, 'Fake cars, e2z')
    view_top_and_bottom(fake_z2e, fake_images, 'Fake cars, z2e')

    view_top_and_bottom(aug_znorms, aug_images, aug_label+', z norms')
    view_top_and_bottom(aug_e2z, aug_images, aug_label+', e2z')
    view_top_and_bottom(aug_z2e, aug_images, aug_label+', z2e')

    ###################

    
    
    plt.title("logpriors")
    plt.hist(data.real.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.log_priors.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()
    
    plt.title("e2z jacobians")
    plt.hist(data.real.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()    
    
    
    plt.title("e2z combined")
    plt.hist(data.real.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()       
    
    plt.title("z2e jacobians")
    plt.hist(data.real.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()     
    
    
    plt.title("z2e combined")
    plt.hist(data.real.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()     
    
    plt.title("z norms")
    plt.hist(data.real.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
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


        

        
        
        
        

    
    

    
def dataset_config(key, dataset_directory, model_path, model_name_prefix, stylegan_file):
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
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'data_keys': ['images']
                },
                'encode_stage': {
                    'mode': 'encode',
                    'fake_sample_file': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_sample_file': os.path.join(model_path, 'cifar_sorted_stylegan.pth'),
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
                },
                'phi_stage': {
                    'mode': 'threeway_encodings',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'encoding_key': 'encodings_' + model_name_prefix
                },
                'prob_stage': {
                    'mode': 'extract_probs',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'phi_model_file':  os.path.join(model_path, model_name_prefix+'_phi.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                    'encoding_key': 'encodings_' + model_name_prefix                     
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'phi_model_file':  os.path.join(model_path, model_name_prefix+'_phi.pkl'),
                    'mode': 'visualization',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'encoding_key': 'encodings_' + model_name_prefix                    
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
                },
            },

            'cifar_1_all_variation_2': {
                'ae_stage': {
                    'mode': 'threeway',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'data_keys': ['images', 'encodings_vgg16']
                },
                'encode_stage': {
                    'mode': 'encode',
                    'fake_sample_file': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_sample_file': os.path.join(model_path, 'cifar_sorted_stylegan.pth'),
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl')
                },
                'phi_stage': {
                    'mode': 'threeway_encodings',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'encoding_key': 'encodings_' + model_name_prefix
                },
                'prob_stage': {
                    'mode': 'extract_probs',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'phi_model_file':  os.path.join(model_path, model_name_prefix+'_phi.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0, 2, 3, 4, 5],
                    'encoding_key': 'encodings_' + model_name_prefix                     
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'phi_model_file':  os.path.join(model_path, model_name_prefix+'_phi.pkl'),
                    'mode': 'visualization',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8,9],
                    'encoding_key': 'encodings_' + model_name_prefix                    
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
                },
            },
            'cifar_1_all_combined': {
                'ae_stage': {
                    'mode': 'threeway_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [2,3,4,5,6,7,8,9],
                    'data_keys': ['images', 'encodings_vgg16']
                },
                'prob_stage': {
                    'mode': 'extract_probs_combined',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'mode': 'visualize_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
                },

            },
            'cifar_1_all_combined_no_train_augment': {
                'ae_stage': {
                    'mode': 'threeway_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [],
                    'data_keys': ['images', 'encodings_vgg16']
                },
                'prob_stage': {
                    'mode': 'extract_probs_combined',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'mode': 'visualize_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
                },

            },
            'cifar_1_all_combined_noise': {
                'ae_stage': {
                    'mode': 'threeway_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [],
                    'use_noise': True,
                    'data_keys': ['images', 'encodings_vgg16']
                },
                'prob_stage': {
                    'mode': 'extract_probs_combined',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [],
                    'use_noise':True
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'mode': 'visualize_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0],
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
                },

            },
            'exp_2_cifar_1_all_combined': {
                'ae_stage': {
                    'mode': 'threeway_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'), # preprocessed standard data
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), # preprocessed fake data
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [0,2,3,4,5,6,7,8],
                    'data_keys': ['images', 'encodings_vgg16']
                },
                'prob_stage': {
                    'mode': 'extract_probs_combined',
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [9],
                },
                'visualize_stage': {
                    'stylegan_file': stylegan_file,
                    'model_file': os.path.join(model_path, model_name_prefix+'_autoencoder.pkl'),
                    'mode': 'visualize_combined',
                    'real': os.path.join(model_path, 'cifar_sorted.pth'),
                    'fake': os.path.join(model_path, 'cifar_sorted_stylegan.pth'), 
                    'augmented': os.path.join(model_path, 'cifar_sorted.pth'),
                    'real_classes':[ 1 ],
                    'fake_classes':[ 1 ],
                    'augmented_classes': [9],
                },
                'plot_stage': {
                    'prob_sample_file': os.path.join(model_path, model_name_prefix+'_extracted.pkl')
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
                    'n_epochs':40,
                    'use_adversary':True
                },
                'phi_stage': {
                    'n_epochs':40,
                    
                }
            },
            'cifar_1_all_variation_2': {
                'ae_stage': {
                    'n_epochs':40,
                    'use_adversary':False,
                    'use_features':True
                },
                'phi_stage': {
                    'n_epochs':80,
                    
                }
            },
            'combined': {
                'ae_stage': {
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.05,
                    'significance':5,
                }
            },
            'combined_ae_simple': {
                'ae_stage': {
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer': {
                'ae_stage': {
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer_no_train_augment': {
                'ae_stage': {
                    'use_augmented':False,
                    'print_every':10,
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer_no_train_augment_no_adversary': {
                'ae_stage': {
                    'use_augmented':False,
                    'use_adversary':False,
                    'print_every':10,
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer_no_features': {
                'ae_stage': {
                    'n_epochs':200,
                    'print_every':20,
                    'use_features':False,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer_no_train_augment_no_adversary_no_features': {
                'ae_stage': {
                    'use_augmented':False,
                    'use_adversary':False,
                    'print_every':10,
                    'n_epochs':200,
                    'use_features':False,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': True
                }
            },
            'combined_ae_layer_simpler_loss': {
                'ae_stage': {
                    'n_epochs':200,
                    'use_features':True,
                    'ring_loss_after':10, #start using ring loss after this many epochs
                    'ring_loss_max':10000, # stop using ring loss after this many
                    'lmbda_norm': 1,
                    'lmbda_cosine': 1,
                    'lmbda_recon': 0.5,
                    'lmbda_feat': 0.5,
                    'lmbda_adv': 1,
                    'lmbda_ring': 0.02,
                    'significance':5,
                    'use_simple_ring_loss': False,
                    'use_simpler_ring_loss': True
                }
            },

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
        },
        'nonlinear_ae': {
            'input_size':32,
            'linear': False,
            'very_lean':False,
            'use_adversary':True
        },
        'mixed_feature_ae': {
            'input_size':32,
            'linear': True,
            'mixed': True,
            'very_lean':True,
            'use_adversary':False,
            'use_features':True
        },
        'combined_ae': {
            'global_lr':0.0001,
            'use_features':True,
            'device':'cuda:0',
        },
        'combined_ae_simple': {
            'global_lr':0.0001,
            'use_features':True,
            'device':'cuda:0',
            'use_simple_nets':True
        },
        'combined_ae_layer': {
            'global_lr':0.0001,
            'use_features':True,
            'use_linear_nets':False,
            'use_layer_norm':True,
            'device':'cuda:0',
            'use_simple_nets':False
        },
        'combined_ae_layer_no_features': {
            'global_lr':0.0001,
            'use_features':False,
            'use_linear_nets':False,
            'use_layer_norm':True,
            'device':'cuda:0',
            'use_simple_nets':False
        },

    }
    
    return cfg[key]


def phi_config(key):
    cfg = {
        'linear_ae': {},
        'adversarial_phi': {
            'use_adversary':False,
            'use_friend':False
            }
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
    
    
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    dataset_dir = None
    stylegan_file = None
    
    if 'jacobswks20' in hostname:
        dataset_dir = '../../datasets'
        stylegan_file = '../../repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl'
    elif 'LV426' in hostname:
        dataset_dir = '/mnt/linuxshared/phd-research/data/standard_datasets'
        stylegan_file = '../repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl'

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_ae')
    parser.add_argument('--model_path', type=str, default='./models/ae2')
    parser.add_argument('--dataset_directory', type=str, default=dataset_dir)
    parser.add_argument('--stylegan_file', type=str, default=stylegan_file)
    parser.add_argument('--experiment_prefix', type=str, default='linear_ae_model')
    parser.add_argument('--experiment_suffix', type=str, default=None)
    parser.add_argument('--experiment_number', type=int, default=None)
    parser.add_argument('--model_name_prefix', type=str, default=None)
    parser.add_argument('--autoencoder_config_key', default='linear_ae')
    parser.add_argument('--phi_config_key', default='linear_ae')
    parser.add_argument('--dataset_config_key', default='cifar_1_all')
    parser.add_argument('--train_config_key', default='cifar_1_all')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--aug_label', default='real planes')
    parser.add_argument('--aug_class', type=int, default=0)

    opt = parser.parse_args()
    
    if opt.experiment_suffix is None:
        if opt.experiment_number is None:
            opt.experiment_suffix = ''
        else:
            opt.experiment_suffix = '_exp_'+str(opt.experiment_number)
    
    if opt.model_name_prefix is None:
        opt.model_name_prefix = opt.experiment_prefix + opt.experiment_suffix
    
    data_cfg = dataset_config(opt.dataset_config_key, opt.dataset_directory, opt.model_path, opt.model_name_prefix, opt.stylegan_file)
    autoencoder_cfg = autoencoder_config(opt.autoencoder_config_key)
    phi_cfg = phi_config(opt.phi_config_key)
    train_cfg = train_config(opt.train_config_key)
    

#    print("Data config")
#    print(data_cfg)
    
    if opt.mode == 'train_ae':
        build_and_train_autoencoder(opt.model_path, opt.model_name_prefix, autoencoder_cfg, data_cfg, train_cfg)        
    elif opt.mode == 'visualize_ae':
        visualize_autoencoder(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'visualize':
        visualize_experiment(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'preprocess_cifar':
        preprocess_cifar(opt.dataset_directory, opt.model_path)
    elif opt.mode == 'sample_stylegan':
        sample_cifar_stylegan(opt.model_path, opt.stylegan_file)
    elif opt.mode == 'test_sorted':
        test_sorted_dataset()
    elif opt.mode == 'test_multi':
        test_multi_loader()
    elif opt.mode == 'test_norm':
        test_stylegan_norm()
    elif opt.mode == 'encode_samples':
        encode_samples(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'encode_samples_vgg':
        encode_samples(opt.model_path, opt.model_name_prefix, data_cfg, encode_vgg=True)
    elif opt.mode == 'train_phi':
        build_and_train_invertible(opt.model_path, opt.model_name_prefix, phi_cfg, data_cfg, train_cfg, load_model=opt.load_model)
    elif opt.mode == 'extract_probs':
        extract_probabilities(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'plot_probs':
        view_extracted_probabilities(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'visualize_model':
        visualize_model(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'test_vgg':
        test_vgg(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'train_combined':
        build_and_train_combined_autoencoder(opt.model_path, opt.model_name_prefix, autoencoder_cfg, data_cfg, train_cfg)
    elif opt.mode == 'extract_probs_combined':
        from ae_combined import extract_probabilities_combined
        extract_probabilities_combined(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'visualize_model_combined':
        from ae_combined import visualize_model_combined
        visualize_model_combined(opt.model_path, opt.model_name_prefix, data_cfg)
    elif opt.mode == 'plot_probs_combined':
#        from ae_combined import view_extracted_probabilities_combined
        from ae_combined import view_top_and_bottom
        view_extracted_probabilities_combined(opt.model_path, opt.model_name_prefix, data_cfg, aug_label=opt.aug_label, aug_class=opt.aug_class)

           

# Next:
# classifier comparison
# autoencoder reconstruction errors
# realnvp phi nets?
# clean up code

