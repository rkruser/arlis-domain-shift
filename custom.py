import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import numpy as np
import pickle
import sys

import matplotlib.pyplot as plt


from models import load_torch_class
from model_functions import jacobian, log_priors


def sample_cifar_stylegan():
    encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
    
    from models import load_torch_class
    
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()
    
    z_values = torch.randn(50000, 512, device='cuda')
    class_constant = torch.zeros(10, device='cuda')
    class_constant[0] = 1

    encodings = []
    images = []
    for i, batch in enumerate(torch.chunk(z_values, 50000//128 + 1)):
        classes = class_constant.repeat(batch.size(0),1)
        
        w_values = cifar_stylegan_net.mapping(batch, classes)
        image_outputs = cifar_stylegan_net.synthesis(w_values, noise_mode='const', force_fp32=True)
        images.append(image_outputs.detach().cpu())
        rescaled_outputs = ((image_outputs+1)/2).clamp(0,1)
        encoded = encoder(rescaled_outputs.view(-1,3072))
        encodings.append(encoded.detach().cpu())
        
        if i%10 == 0:
            print(i)
        
        """
        print(rescaled_outputs[0])
        npvers = rescaled_outputs.view(-1).detach().cpu().numpy()
        print(np.percentile(npvers,1))
        print(np.percentile(npvers,10))
        print(np.percentile(npvers,90))
        print(np.percentile(npvers,99))
        break
        """
        
        
    all_encodings = torch.cat(encodings)
    all_images = torch.cat(images)

    torch.save({'z_values':z_values.cpu(), 'images':all_images, 'encodings':all_encodings}, './generated/custom_encodings/cifar_class_0_encoded.pth')




def view_tensor_images(t):
    grid = tv.utils.make_grid(t.detach().cpu())
    grid = grid.permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()
    

def view_samples():
    data = torch.load('./generated/custom_encodings/cifar_class_0_encoded.pth')
    print(data['z_values'].size())
    print(data['images'].size())
    print(data['encodings'].size())
    
    
#     encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
#     images_to_encode = data['images'].cuda()
#     images_to_encode = ((images_to_encode+1)/2).clamp(0,1)
#     new_encodings = []
#     for batch in torch.chunk(images_to_encode, 50000//256):
#         new_encoding = encoder(batch.view(-1,3072))
#         new_encodings.append(new_encoding)
#     all_new = torch.cat(new_encodings)
#     data['encodings'] = all_new.detach().cpu()
#     torch.save(data,'./generated/custom_encodings/cifar_class_0_encoded.pth')
#     sys.exit()
    
    
    
    
    
    decoder = pickle.load(open('./models/small_linear_cifar10_encoder/decoder.pkl','rb'))
    print(decoder)
    
    imgrid = tv.utils.make_grid(((data['images'][:64]+1)/2).clamp(0,1)).permute(1,2,0).numpy()
    plt.imshow(imgrid)
    plt.show()
    
    to_decode = data['encodings'][:64].cuda()
    decoded = torch.sigmoid(decoder(to_decode)).detach().cpu().view(-1,3,32,32)
    decoded_grid = tv.utils.make_grid(decoded).permute(1,2,0).numpy()
    plt.imshow(decoded_grid)
    plt.show()
    

    

#     encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
#     decoder = pickle.load(open('./models/small_linear_cifar10_encoder/decoder.pkl','rb'))    
#     ims_initial = ((data['images'][:64]+1)/2).clamp(0,1).cuda()
#     ims_final = torch.sigmoid(decoder(encoder(ims_initial.view(-1,3072)))).reshape(-1,3,32,32)
#     view_tensor_images(ims_initial)
#     view_tensor_images(ims_final)



def train_bidirectional():
#     z_to_encodings = nn.Sequential(
#         nn.Linear(512,1024),
#         nn.LeakyReLU(0.2,inplace=True),
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,512)
#     ).cuda()

#     encodings_to_z = nn.Sequential(
#         nn.Linear(512,1024),
#         nn.LeakyReLU(0.2,inplace=True),
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,1024),
#         nn.LeakyReLU(0.2,inplace=True),        
#         nn.Linear(1024,512)
#     ).cuda()


    print("loading")
    z_to_encodings = pickle.load(open('./models/small_linear_cifar10_encoder/z_to_e.pkl','rb')).cuda()
    encodings_to_z = pickle.load(open('./models/small_linear_cifar10_encoder/e_to_z.pkl','rb')).cuda()
    
    
    

    data = torch.load('./generated/custom_encodings/cifar_class_0_encoded.pth')    
    z_values = data['z_values'].cuda()
    encodings = data['encodings'].cuda()
    
    dataset = torch.utils.data.TensorDataset(z_values, encodings)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size = 128, 
                                          shuffle=True,
                                          drop_last=True)
    
    
    
    lossfunc = torch.nn.MSELoss()
    
    z_to_e_optim = torch.optim.Adam(z_to_encodings.parameters())
    e_to_z_optim = torch.optim.Adam(encodings_to_z.parameters())  
    
    lmbda = 0.25
    
    alternator = 0
    for n in range(50):
        print("Epoch", n)
        for i, batch in enumerate(dataloader):
            z_batch, e_batch = batch
            
            if alternator == 0:
                predicted_encodings = z_to_encodings(z_batch)
                cycled_z = encodings_to_z(predicted_encodings)
                loss = lossfunc(predicted_encodings, e_batch)+lmbda*lossfunc(cycled_z, z_batch)
                
                z_to_encodings.zero_grad()
                loss.backward()
                z_to_e_optim.step()
                
            else:
                predicted_z = encodings_to_z(e_batch)
                cycled_e = z_to_encodings(predicted_z)
                loss = lossfunc(predicted_z, z_batch) + lmbda*lossfunc(cycled_e, e_batch)
                
                encodings_to_z.zero_grad()
                loss.backward()
                e_to_z_optim.step()


            if i%100 == 0:
                print(i, alternator, loss.item())
                
            alternator = 1-alternator


    pickle.dump(z_to_encodings, open('./models/small_linear_cifar10_encoder/z_to_e.pkl','wb'))
    pickle.dump(encodings_to_z, open('./models/small_linear_cifar10_encoder/e_to_z.pkl','wb'))    



def show_bidirectional_results():
    print("loading")
    z_to_encodings = pickle.load(open('./models/small_linear_cifar10_encoder/z_to_e.pkl','rb')).cuda()
    encodings_to_z = pickle.load(open('./models/small_linear_cifar10_encoder/e_to_z.pkl','rb')).cuda()
    
    encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb'))
    decoder = pickle.load(open('./models/small_linear_cifar10_encoder/decoder.pkl','rb'))

    data = torch.load('./generated/custom_encodings/cifar_class_0_encoded.pth')
    
    first_64_images = ((data['images'][:64]+1)/2).clamp(0,1)
    view_tensor_images(first_64_images)
    
    first_64_z = data['z_values'][:64].cuda()
    first_64_predicted_e = z_to_encodings(first_64_z)
    first_64_predicted_decoded = torch.sigmoid(decoder(first_64_predicted_e)).reshape(-1,3,32,32)
    view_tensor_images(first_64_predicted_decoded)
    
    
    
    
    
    
    dataset_folder = '/fs/vulcan-datasets/CIFAR'
    im_transform = tv.transforms.ToTensor()
    dataset = tv.datasets.CIFAR10(dataset_folder, train=True, transform=im_transform, target_transform=None, download=False)
    first_64_real_class_0 = []
    pos = 0
    #permutation = torch.randperm(len(dataset))
    while len(first_64_real_class_0) < 64:
        im, label = dataset[pos]
        if label == 1:
            first_64_real_class_0.append(im)
        pos += 1
    first_64_real_class_0 = torch.stack(first_64_real_class_0)#.reshape(-1,3072).cuda()
    view_tensor_images(first_64_real_class_0)
    first_64_real_encoded = encoder(first_64_real_class_0.reshape(-1,3072).cuda())
    
    predicted_z = encodings_to_z(first_64_real_encoded) #nonlinear map back to z-space
    class_constant = torch.zeros(10, device='cuda')
    class_constant[0] = 1
    classes = class_constant.repeat(64,1)
    
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda() 
    w_values = cifar_stylegan_net.mapping(predicted_z, classes)
    image_outputs = cifar_stylegan_net.synthesis(w_values, noise_mode='const', force_fp32=True)
    image_outputs = ((image_outputs+1)/2).clamp(0,1)
    view_tensor_images(image_outputs)
 

# def apply_forward_jacobian(inputs, outputs):
#     indices = torch.arange(len(inputs))
#     acc = []
#     for i, ch in enumerate(torch.chunk(indices, len(inputs)//64 + 1)):
#         print("Jacobians", i)
#         logjacobians = -jacobian(inputs[ch], outputs[ch])
#         acc.append(logjacobians)
        
#     return torch.cat(acc)
        
def jacobians():
    dataset_folder = '/fs/vulcan-datasets/CIFAR'
    im_transform = tv.transforms.ToTensor()
    dataset = tv.datasets.CIFAR10(dataset_folder, train=True, transform=im_transform, target_transform=None, download=False)
    real_class_0 = []
    class_0_inds = []
    real_class_1 = []
    class_1_inds = []
    pos = 0
    #permutation = torch.randperm(len(dataset))
    while min(len(real_class_0),len(real_class_1)) < 1000:
        im, label = dataset[pos]
        if label == 0:
            real_class_0.append(im)
            class_0_inds.append(pos)
        elif label == 1:
            real_class_1.append(im)
            class_1_inds.append(pos)
        pos += 1
    real_class_0 = torch.stack(real_class_0).cuda()
    real_class_1 = torch.stack(real_class_1).cuda()
    
    encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb')).cuda()
    encodings_to_z = pickle.load(open('./models/small_linear_cifar10_encoder/e_to_z.pkl','rb')).cuda()
    
    
    class_0_encoded = encoder(real_class_0.reshape(-1,3072)).detach()
    class_0_encoded.requires_grad_(True)
    class_0_inverted = encodings_to_z(class_0_encoded)
    class_1_encoded = encoder(real_class_1.reshape(-1,3072)).detach()
    class_1_encoded.requires_grad_(True)
    class_1_inverted = encodings_to_z(class_1_encoded)
    
    
    
    class_0_log_jacobians = -jacobian(class_0_encoded,class_0_inverted) #apply_forward_jacobian(class_0_encoded, class_0_inverted)
    class_1_log_jacobians = -jacobian(class_1_encoded,class_1_inverted) #apply_forward_jacobian(class_1_encoded, class_1_inverted)
    
    class_0_priors = log_priors(class_0_inverted)
    class_1_priors = log_priors(class_1_inverted)
    
    
    
    data = torch.load('./generated/custom_encodings/cifar_class_0_encoded.pth')    
    fake_class_0_encodings = data['encodings'][:1000].cuda()
    fake_class_0_encodings.requires_grad_(True)
    fake_class_0_inverted = encodings_to_z(fake_class_0_encodings)
    
    fake_class_0_log_jacobians = -jacobian(fake_class_0_encodings, fake_class_0_inverted)
    fake_class_0_priors = log_priors(fake_class_0_inverted)
    
    
    
    torch.save({'class_0_log_jacob':class_0_log_jacobians, 'class_1_log_jacob':class_1_log_jacobians,
                'class_0_priors':class_0_priors, 'class_1_priors':class_1_priors, 
                'class_0_inds':class_0_inds, 'class_1_inds':class_1_inds,
                'fake_class_0_log_jacob':fake_class_0_log_jacobians, 'fake_class_0_priors':fake_class_0_priors}, 
               './generated/custom_encodings/cifar_encoder_logprobs.pth')
    
    
def jacobians_other_direction():
    dataset_folder = '/fs/vulcan-datasets/CIFAR'
    im_transform = tv.transforms.ToTensor()
    dataset = tv.datasets.CIFAR10(dataset_folder, train=True, transform=im_transform, target_transform=None, download=False)
    real_class_0 = []
    class_0_inds = []
    real_class_1 = []
    class_1_inds = []
    pos = 0
    #permutation = torch.randperm(len(dataset))
    while min(len(real_class_0),len(real_class_1)) < 1000:
        im, label = dataset[pos]
        if label == 0:
            real_class_0.append(im)
            class_0_inds.append(pos)
        elif label == 1:
            real_class_1.append(im)
            class_1_inds.append(pos)
        pos += 1
    real_class_0 = torch.stack(real_class_0).cuda()
    real_class_1 = torch.stack(real_class_1).cuda()
    
    encoder = pickle.load(open('./models/small_linear_cifar10_encoder/encoder.pkl','rb')).cuda()
    encodings_to_z = pickle.load(open('./models/small_linear_cifar10_encoder/e_to_z.pkl','rb')).cuda()
    z_to_encodings = pickle.load(open('./models/small_linear_cifar10_encoder/z_to_e.pkl','rb')).cuda()
    
    class_0_encoded = encoder(real_class_0.reshape(-1,3072))
    class_0_inverted = encodings_to_z(class_0_encoded).detach()
    class_0_inverted.requires_grad_(True)
    class_0_reencoded = z_to_encodings(class_0_inverted)

    
    class_1_encoded = encoder(real_class_1.reshape(-1,3072))
    class_1_inverted = encodings_to_z(class_1_encoded).detach()
    class_1_inverted.requires_grad_(True) 
    class_1_reencoded = z_to_encodings(class_1_inverted)
  
    
    
#     class_0_log_jacobians_re = jacobian(class_0_inverted,class_0_reencoded) #no negative since reverse jacobian
#     class_1_log_jacobians_re = jacobian(class_1_inverted,class_1_reencoded) #no negative since reverse jacobian
    
#     class_0_priors_re = log_priors(class_0_inverted)
#     class_1_priors_re = log_priors(class_1_inverted)
    
    
    
    data = torch.load('./generated/custom_encodings/cifar_class_0_encoded.pth')    
    fake_class_0_encodings = data['encodings'][:1000].cuda()
    fake_class_0_z = data['z_values'][:1000].cuda()
    fake_class_0_z.requires_grad_(True)
    fake_class_0_reencoded = z_to_encodings(fake_class_0_z)
    
#     fake_class_0_log_jacobians_re = jacobian(fake_class_0_z, fake_class_0_reencoded)
#     fake_class_0_priors_re = log_priors(fake_class_0_z)
    
    
    class_0_recon_errors = torch.linalg.norm(class_0_encoded-class_0_reencoded,dim=1)**2
    class_1_recon_errors = torch.linalg.norm(class_1_encoded-class_1_reencoded,dim=1)**2
    fake_class_0_recon_errors = torch.linalg.norm(fake_class_0_encodings-fake_class_0_reencoded,dim=1)**2
    print(class_0_recon_errors.size())
    
    
    torch.save({'class_0_recon_errors':class_0_recon_errors, 'class_1_recon_errors':class_1_recon_errors, 
                'fake_class_0_recon_errors':fake_class_0_recon_errors}, 
               './generated/custom_encodings/cifar_encoder_recon_errors.pth')
#     torch.save({'class_0_log_jacob_re':class_0_log_jacobians_re, 'class_1_log_jacob_re':class_1_log_jacobians_re,
#                 'class_0_priors_re':class_0_priors_re, 'class_1_priors_re':class_1_priors_re, 
#                 'class_0_inds':class_0_inds, 'class_1_inds':class_1_inds,
#                 'fake_class_0_log_jacob_re':fake_class_0_log_jacobians_re, 'fake_class_0_priors_re':fake_class_0_priors_re,
#                 'class_0_recon_errors':class_0_recon_errors, 'class_1_recon_errors':class_1_recon_errors, 
#                 'fake_class_0_recon_errors':fake_class_0_recon_errors}, 
#                './generated/custom_encodings/cifar_encoder_logprobs_reverse.pth')

    
def view_jacobian_histogram():
#     jacobian_data = torch.load('./generated/custom_encodings/cifar_encoder_logprobs.pth')
    
#     total_logprobs_0 = jacobian_data['class_0_log_jacob']+jacobian_data['class_0_priors']
#     total_logprobs_1 = jacobian_data['class_1_log_jacob']+jacobian_data['class_1_priors']
#     total_logprobs_fake_0 = jacobian_data['fake_class_0_log_jacob']+jacobian_data['fake_class_0_priors']
    
#     print("Total logprob histogram")
#     plt.hist(total_logprobs_0.cpu().numpy(), bins=50, density=True, alpha=1, label="Airplanes", range=[-2500,-1700])
#     plt.show()
#     plt.hist(total_logprobs_1.cpu().numpy(), bins=50, density=True, alpha=1, label="Cars",range=[-2500,-1700])
#     plt.show()
#     plt.hist(total_logprobs_fake_0.cpu().numpy(), bins=50, density=True, alpha=1, label="Fake airplanes", range=[-2500,-1700])
#     #plt.legend(loc='upper left')
#     plt.show()
    
    
    
#     print("Prior histogram")
#     plt.hist(jacobian_data['class_0_priors'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()
#     plt.hist(jacobian_data['class_1_priors'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()
#     plt.hist(jacobian_data['fake_class_0_priors'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()  
    
    
    
#     print("Jacobian histogram")
#     plt.hist(jacobian_data['class_0_log_jacob'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()
#     plt.hist(jacobian_data['class_1_log_jacob'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()
#     plt.hist(jacobian_data['fake_class_0_log_jacob'].cpu().numpy(), bins=50, density=True, alpha=1)
#     plt.show()     
    
    
    
    
    
    
    reverse_jacobian_data = torch.load('./generated/custom_encodings/cifar_encoder_logprobs_reverse.pth')
    
    print("Total logprob histograms reverse (z to e)")
    total_reverse_0 = reverse_jacobian_data['class_0_log_jacob_re']+reverse_jacobian_data['class_0_priors_re']
    total_reverse_1 = reverse_jacobian_data['class_1_log_jacob_re']+reverse_jacobian_data['class_1_priors_re']
    total_reverse_fake_0 = reverse_jacobian_data['fake_class_0_log_jacob_re']+reverse_jacobian_data['fake_class_0_priors_re']    
    plt.hist(total_reverse_0.cpu().numpy(), bins=50, density=True, alpha=1)
    plt.show()
    plt.hist(total_reverse_1.cpu().numpy(), bins=50, density=True, alpha=1)
    plt.show()
    plt.hist(total_reverse_fake_0.cpu().numpy(), bins=50, density=True, alpha=1)
    plt.show()     
        
        
        
    recon_losses_data = torch.load('./generated/custom_encodings/cifar_encoder_recon_errors.pth')    
    
    print("Reconstruction losses histograms")
    plt.hist(recon_losses_data['class_0_recon_errors'].detach().cpu().numpy(), bins=50, density=True, alpha=0.5, label="Real airplanes")
    #plt.show()
    plt.hist(recon_losses_data['class_1_recon_errors'].detach().cpu().numpy(), bins=50, density=True, alpha=0.5, label="Car ims")
    #plt.show()
    plt.hist(recon_losses_data['fake_class_0_recon_errors'].detach().cpu().numpy(), bins=50, density=True, alpha=0.5, label="Stylegan airplanes")
    #plt.show()       
    plt.legend()
    plt.show()
    

def run_custom_command(command):
    if command == 'sample_cifar_stylegan':
        sample_cifar_stylegan()
    elif command == 'view_samples':
        view_samples()
    elif command == 'train_bidirectional':
        train_bidirectional()
    elif command == 'show_bidirectional_results':
        show_bidirectional_results()
    elif command == 'jacobian':
        jacobians()
    elif command == 'reverse_jacobians':
        jacobians_other_direction()
    elif command == 'view_jacobian_histogram':
        view_jacobian_histogram()




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        