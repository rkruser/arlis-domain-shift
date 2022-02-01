import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import numpy as np
import pickle
import sys

import matplotlib.pyplot as plt


from models import load_torch_class


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
    


def run_custom_command(command):
    if command == 'sample_cifar_stylegan':
        sample_cifar_stylegan()
    elif command == 'view_samples':
        view_samples()
    elif command == 'train_bidirectional':
        train_bidirectional()
    elif command == 'show_bidirectional_results':
        show_bidirectional_results()




