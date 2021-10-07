#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regressor sampling/testing
"""


### Imports

import argparse
import os

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import models
import datasets




### Command line arguments


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Sample, train, or test')
parser.add_argument('--loadfrom', type=str, default=None, help='Model to load from')
parser.add_argument('--modeldir', type=str, default='models', help='Directory where models are located')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
parser.add_argument('--basename', type=str, default='encoder', help='Name to save model under')

# parameter arguments
parser.add_argument('--encoder_iters', type=int, default=5000, help='Training iterations')
parser.add_argument('--checkpoint_every', type=int, default=1000, help='How often to checkpoint the model')
parser.add_argument('--print_every', type=int, default=20, help='How often to print encoder training loss')
parser.add_argument('--visualize_every', type=int, default=100, help='How often to visualize encoder training process')


parser.add_argument('--nc', type=int, default=3, help="Number of color channels")
parser.add_argument('--ndf', type=int, default=64, help="Number of hidden features in encoder")
parser.add_argument('--encoder_lr', type=float, default=0.0001, help="Encoder learning rate")
parser.add_argument('--nz', type=int, default=100, help="Encoder target dimension")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--encoder_step_lr', type=int, default=20000, help='How often to step down encoder learning rate')
parser.add_argument('--encoder_lr_gamma', type=float, default=0.1, help='How much to scale down encoder learning rate')
parser.add_argument('--encoder_betas', type=tuple, default=(0.9, 0.999), help="Adam betas")
parser.add_argument('--encoder_weight_decay', type=float, default=0.0, help="Encoder weight decay")
parser.add_argument('--encoder_lambda', type=float, default=1.0, help='Encoder latent space loss coefficient')
parser.add_argument('--encoder_recon_lambda', type=float, default=1.0, help='Encoder image space reconstruction loss')
parser.add_argument('--encoder_discriminator_lambda', type=float, default=1.0, help='Encoder discriminator reconstruction loss')
parser.add_argument('--encoder_regularization_lambda', type=float, default=1.0, help='Penalty for large norms of encoded points')

parser.add_argument('--encoder_guidance_lambda', type=float, default=0.1, help="Weight on encoder consistency loss")
parser.add_argument('--latent_optim_check_every', type=int, default=100, help="How often to check on individual optimization iters")
parser.add_argument('--latent_lr', type=float, default=0.001, help="Latent code optim learning rate")
parser.add_argument('--latent_betas', type=tuple, default=(0.9, 0.999), help="Latent code optim betas")


parser.add_argument('--regressor_out_dims', type=int, default=2, help="Number of outputs for regressor")
parser.add_argument('--regressor_step_lr', type=int, default=15, help='How often to step down encoder learning rate')
parser.add_argument('--regressor_lr_gamma', type=float, default=0.1, help='How much to scale down encoder learning rate')
parser.add_argument('--regressor_betas', type=tuple, default=(0.9, 0.999), help="Adam betas")
parser.add_argument('--regressor_weight_decay', type=float, default=0.0, help="Encoder weight decay")
parser.add_argument('--regressor_lr', type=float, default=0.0001, help="Encoder learning rate")
parser.add_argument('--regressor_epochs', type=int, default=30, help="Number of training epochs for regressor")


opt=parser.parse_args()




def visualize_image_grid(ims, nrow=4):
    sample_grid = torchvision.utils.make_grid(ims, nrow=nrow)

    samples_processed = sample_grid.detach().permute(1,2,0).cpu().numpy()
    samples_processed = (samples_processed+1)/2
    plt.imshow(samples_processed)
    plt.show()



### Jacobian sampling and saving

def sample_jacobian(generator, reconstruction_dataset, batch_size, device, method='analytical'):
    dataloader = torch.utils.data.DataLoader(reconstruction_dataset, batch_size, shuffle=False, drop_last=False)
    
    generator.eval()
    
    if method == 'analytical':
        
        all_logprobs = []
        all_errors = []
        for i, batch in enumerate(dataloader):
            print("Batch {0} of {1}".format(i, len(dataloader)))
            z_codes, ims = batch[0], batch[1]   
            z_codes = z_codes.to(device)
            ims = ims.to(device)
            z_codes.requires_grad_(True)
            output = generator(z_codes)
            
            #print("Image grids:")
            #visualize_image_grid(ims[:16])
            #visualize_image_grid(output[:16])
            #print(ims.min(), ims.max())
            #print(output.min(), output.max())

            
            num_points = z_codes.size(0)
            output_size = output.numel() // num_points
            input_size = z_codes.numel() // num_points
            
            output = output.view(num_points, output_size) #flatten outputs
            ims = ims.view(num_points, output_size)
            error_vals = (output-ims).norm(dim=1).detach().cpu() #torch.sqrt(((output-ims)**2).mean(dim=1))
            
            #print("Norm differences")
            #print(error_vals[:16])
            #input("Press a key")
            
            #print("Starting gradients")
            gradients = [torch.autograd.grad(outputs=output[:,k], inputs=z_codes,
                                      grad_outputs=torch.ones(num_points, device=device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0] for k in range(output_size)]
            #print("End gradients")
            
            gradients = torch.cat(gradients,dim=1)#.detach()
            gradients = gradients.reshape(num_points, output_size, input_size)
            
            gradients = gradients.detach().cpu().numpy()
    
    
            dim = input_size
            gauss_constant = (-dim / 2)*torch.log(4*torch.acos(torch.tensor(0.0))) # -dim/2 * log(2*pi)
            z_log_likelihood = gauss_constant + (-0.5)*(z_codes**2).sum(dim=1)
            
            z_log_likelihood = z_log_likelihood.detach().cpu().numpy()

            # Compute determinants with QR decomp:
            logprobs = np.zeros(num_points)
            #print("Starting QR")
            for k in range(num_points):
                R = np.linalg.qr(gradients[k], mode='r')
                jacobian_score = -np.log(np.abs(R.diagonal())).sum() #negative because inverting the diagonal elements
                log_prob = z_log_likelihood[k] + jacobian_score
                logprobs[k] = log_prob
            #print("End QR")
            
            logprobs = torch.from_numpy(logprobs)
            
            all_logprobs.append(logprobs)
            all_errors.append(error_vals)
            
        logprobs = torch.cat(all_logprobs)
        error_vals = torch.cat(all_errors)
        
        mean_error = error_vals.mean()
        lmbda = 1.0/mean_error
        log_penalties = torch.log(lmbda) - lmbda*error_vals
        
        combined = logprobs + log_penalties
        
        
        return logprobs, error_vals, log_penalties, combined
            
### Loading saved samples into dataset




def map_images_to_latent(generator, encoder, dataset, opt):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    
    generator.eval()
    encoder.eval()
    all_codes = []
    all_metrics = []
    for i, batch in enumerate(dataloader): 
        batch = batch[0].to(opt.device)
        batch_size = batch.size(0)
        codes = encoder(batch).detach() #initial guesses
        
        if i%opt.latent_optim_check_every == 0:
            print("Initial code norms", codes[:16].norm(dim=1))
            
            
        codes.requires_grad_(True)
        
        optim = torch.optim.Adam([codes], lr=opt.latent_lr, betas=opt.latent_betas)
        grad_norms = torch.ones(batch_size)
        iters = 0
        while grad_norms.max() > 0.001 and iters<100: #This is the wrong approach; need to test z convergence directly
            fake_ims = generator(codes)
            reconstruction_losses = ((fake_ims.view(batch_size,-1) - batch.view(batch_size, -1))**2).mean(dim=1)
            reconstruction_loss = reconstruction_losses.mean()
            encoder_guidance_loss = ((encoder(fake_ims)-codes)**2).mean()
            
            loss = reconstruction_loss + opt.encoder_guidance_lambda*encoder_guidance_loss
            
            if codes.grad is not None:
                codes.grad.fill_(0.0)
            loss.backward()
            grad_norms = torch.norm(codes.grad, dim=1)
            optim.step()
            iters += 1
        
        
        
        if i%opt.latent_optim_check_every == 0:
            print("Batch {0} of {2}, num_iters {1}, losses:".format(i, iters, len(dataloader)), reconstruction_losses[:16])
            print("Grad norms", grad_norms[:16])
            print("Code norms", codes[:16].norm(dim=1))
            reconstruction_guess = generator(codes[:16])
            print("Real batch:")
            visualize_image_grid(batch[:16])
            print("Reconstructions:")
            visualize_image_grid(reconstruction_guess[:16])
            
            #input("Press any key to continue")
        
        
        all_codes.append(codes.detach().cpu())
        all_metrics.append(reconstruction_losses)
        print("Batch {0} of {3}, iters {1}, max loss {2}".format(i, iters, loss.max(), len(dataloader)))
        
    all_codes = torch.cat(all_codes,dim=0)
    all_metrics = torch.cat(all_metrics)
    torch.save( (all_codes, all_metrics) , os.path.join(opt.modeldir, opt.basename)+'.pth')




# Procedure: train encoder to invert GAN samples, using mixed reconstruction loss / discriminator objective
# For dataset points: Run through encoder to initialize z values, then use mixed objective to optimize individual z points
def train_encoder(encoder, generator, discriminator, opt, feature_embedding_net=None):
    if feature_embedding_net is None:
        feature_embedding_net = lambda x: x
    
    generator.eval()
    discriminator.eval()
    encoder.train()
    
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=opt.encoder_lr, betas=opt.encoder_betas, weight_decay=opt.encoder_weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optim, step_size=1, gamma=opt.encoder_lr_gamma)
    
    all_losses = []
    for i in range(opt.encoder_iters):
        codes = torch.randn(opt.batch_size, opt.nz, device=opt.device)
        images = generator(codes).detach()
        
        predictions = encoder(images)
        regularization_loss = (predictions**2).mean()
        encoder_loss = ((predictions-codes)**2).mean()
        generated_from_encoding = generator(predictions)
        reconstruction_loss = ((feature_embedding_net(generated_from_encoding) - feature_embedding_net(images))**2).mean()
        discriminator_loss = (-1)*discriminator(generated_from_encoding).mean() #change if discriminator has sigmoid or not?
        loss = opt.encoder_lambda*encoder_loss + opt.encoder_recon_lambda*reconstruction_loss + opt.encoder_discriminator_lambda*discriminator_loss+opt.encoder_regularization_lambda*regularization_loss
        
        encoder.zero_grad()
        loss.backward()
        encoder_optim.step()
        
        all_losses.append(loss.item())
        
        if opt.encoder_step_lr == i:
            encoder_scheduler.step()
        
        if i%opt.print_every == 0:
            print("Iter: {0}, Encoder loss = {1}".format(i, loss.item()))
            
        if (i+1)%opt.checkpoint_every == 0:
            torch.save((encoder.state_dict(), encoder_optim.state_dict(), encoder_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'.pth')
        
        
        if (i+1)%opt.visualize_every == 0:
            codes = torch.randn(16, opt.nz, device=opt.device)
            images = generator(codes)
            encoder.eval()
            predicted_codes = encoder(images)
            encoder.train()
            reconstructed_images = generator(predicted_codes)
            print("Sampled images")
            visualize_image_grid(images)
            print("Reconstructed images")
            visualize_image_grid(reconstructed_images)
            
        
        
    torch.save((encoder.state_dict(), encoder_optim.state_dict(), encoder_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'.pth')
    







### Training regressor model, checkpointing, extracting metrics


def train_regressor(regressor, generator, dataset, opt):
    generator.eval()
    regressor.train()
    
    regressor_optim = torch.optim.Adam(regressor.parameters(), lr=opt.regressor_lr, betas=opt.regressor_betas, weight_decay=opt.regressor_weight_decay)
    regressor_scheduler = torch.optim.lr_scheduler.StepLR(regressor_optim, step_size=1, gamma=opt.regressor_lr_gamma)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    all_losses = []
    for epoch in range(opt.regressor_epochs):
        print("Epoch", epoch)
        for i, batch in enumerate(dataloader):
            ims, _, on_manifold_score, off_manifold_score = batch
            ims = ims.to(opt.device)
            on_manifold_score = on_manifold_score.to(opt.device)
            off_manifold_score = off_manifold_score.to(opt.device)
            
            predictions = regressor(ims)
            on_manifold_loss = ((predictions[:,0]-on_manifold_score)**2).mean()
            off_manifold_loss = ((predictions[:,1]-off_manifold_score)**2).mean()
            loss = on_manifold_loss + off_manifold_loss

            
            regressor.zero_grad()
            loss.backward()
            regressor_optim.step()
            
            all_losses.append((on_manifold_loss.item(), off_manifold_loss.item(), loss.item()))

            if i%opt.print_every == 0:
                print("Iter: {0}, total loss = {1}, on_manifold = {2}, off_manifold = {3}".format(i, loss.item(), on_manifold_loss.item(), off_manifold_loss.item()))
                
            if (i+1)%opt.checkpoint_every == 0:
                torch.save((regressor.state_dict(), regressor_optim.state_dict(), regressor_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'_regressor.pth')
            
            
        if (epoch+1)%opt.regressor_step_lr == 0:
            print("Stepping regressor lr")
            regressor_scheduler.step()
        if (epoch+1)%opt.checkpoint_every == 0:
            print("Checkpointing at", epoch)
            torch.save((regressor.state_dict(), regressor_optim.state_dict(), regressor_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'_regressor.pth')
        
        
    torch.save((regressor.state_dict(), regressor_optim.state_dict(), regressor_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'_regressor.pth')













"""
def workflow():
    generator, discriminator = load_gan(file_location)
    feature_embedding_network = load_embedding_network(embedding_net_file_location) #optional
    encoder = train_encoder(generator, discriminator, feature_embedding_network, opt)
    inverse_gan_dataset, reconstruction_losses = invert_dataset(generator, discriminator, dataset, encoder, feature_embedding_network, subsample_jacobian = False)
    regressor = train_regressor(inverse_gan_dataset)
    test_regressor(regressor)
    


For inversions of large images:
    inverse encoder and regressor will both be resnets
    feature embedding network can be a pretrained resnet perhaps
    
The encoder dataset:
    will include both generated points and generated points with added noise
    
    
The regressor dataset:
    will include generated points, generated points with added noise, inverted real images, (real images with noise?)














"""





def train_encoder_mnist_gan(opt):
    fname = os.path.join(opt.modeldir, opt.loadfrom)
    params = torch.load(fname)
    loaded_opt = params['opt']
    loaded_opt.device = opt.device
    print("Loaded options:", loaded_opt)
    model = models.GAN_Model(loaded_opt, load_params = params)
    generator = model.netg
    discriminator = model.netd
    
    encoder = models.NetE32(opt).to(opt.device)
    encoder.apply(models.weights_init)
    
    
    train_encoder(encoder, generator, discriminator, opt)



def invert_mnist_gan_with_encoder(opt):
    gan_fname = os.path.join(opt.modeldir, "wgan_mnist_successful_test.pth")
    params = torch.load(gan_fname)
    loaded_opt = params['opt']
    loaded_opt.device = opt.device
    print("Loaded options:", loaded_opt)
    model = models.GAN_Model(loaded_opt, load_params = params)
    generator = model.netg
    
    
    encoder_fname = os.path.join(opt.modeldir, "encoder.pth")
    encoder_state_dict, _, _, _, encoder_loaded_opt = torch.load(encoder_fname)
    encoder_loaded_opt.device = opt.device
    encoder = models.NetE32(encoder_loaded_opt)
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.to(opt.device)
    

    dataset = datasets.get_dataset(dataset='mnist')
    
    
    map_images_to_latent(generator, encoder, dataset, opt)
    
 
    
    
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, cutoff=None):
        super().__init__()
        self.datasets = datasets
        assert(len(datasets) > 0)
        length = len(self.datasets[0])
        for d in datasets:
            assert(len(d) == length)
        
        if cutoff is not None:
            self.length = cutoff
        else:
            self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        output = ()
        for d in self.datasets:
            d_out = d[index]
            if not isinstance(d_out, tuple):
                d_out = (d_out,)
            output += d_out
        
        return output  
    
    
    
def sample_jacobian_mnist_gan(opt):
    gan_fname = os.path.join(opt.modeldir, "wgan_mnist_successful_test.pth")
    params = torch.load(gan_fname)
    loaded_opt = params['opt']
    loaded_opt.device = opt.device
    print("Loaded options:", loaded_opt)
    model = models.GAN_Model(loaded_opt, load_params = params)
    generator = model.netg
    generator.eval()

    mnist_dataset = datasets.get_dataset(dataset='mnist')
    
    encoded_mnist, _ = torch.load('models/saved_mnist_codes.pth')
            
    
    combined_dataset = CombinedDataset([encoded_mnist, mnist_dataset])
    
    
    # Try sending through generator samples
    #sampled_codes = torch.randn(36,100, device=opt.device)
    #fake_ims = generator(sampled_codes).detach()
    #combined_dataset = torch.utils.data.TensorDataset(sampled_codes, fake_ims)
    
    
    logprobs, error_vals, log_penalties, combined = sample_jacobian(generator, combined_dataset, opt.batch_size, opt.device, method='analytical')
    
    #print("Logprobs:",logprobs, "error vals:", error_vals, "Log_penalties", log_penalties)
    torch.save((logprobs, error_vals, log_penalties, combined), 'models/saved_mnist_logprobs.pth')
    
    
# (encoder.state_dict(), encoder_optim.state_dict(), encoder_scheduler.state_dict(), all_losses, opt)




def train_regressor_mnist_gan(opt):
    gan_fname = os.path.join(opt.modeldir, "wgan_mnist_successful_test.pth")
    params = torch.load(gan_fname)
    loaded_opt = params['opt']
    loaded_opt.device = opt.device
    print("Loaded options:", loaded_opt)
    model = models.GAN_Model(loaded_opt, load_params = params)
    generator = model.netg
    generator.eval()


    mnist_dataset = datasets.get_dataset(dataset='mnist')
    
    logprobs, error_vals, log_penalties, combined = torch.load('models/saved_mnist_logprobs.pth')
    logprobs = logprobs.float() # Because of how it was saved

    
    squared_norm_error = error_vals**2
    mean_squared_norm_error = squared_norm_error.mean()
    mean_log_prob = logprobs.mean() #Negative
    penalty_lambda = -mean_squared_norm_error / mean_log_prob #Positive
    
    error_penalties = -squared_norm_error / penalty_lambda #This is log exponential error = log exp ( -||x||^2 / penalty_lambda)
    mean_error_penalty = error_penalties.mean()
    
    centered_log_probs = logprobs - mean_log_prob
    centered_error_penalties = error_penalties - mean_error_penalty
    combined_std = torch.cat([centered_log_probs, centered_error_penalties]).std()
    combined_scale = combined_std * (1.0/3) # Set the normalized standard deviation to 3, somewhat arbitrarily, but to preserve differences between high and low probability regions
    
    scaled_centered_log_probs = centered_log_probs / combined_scale
    scaled_centered_error_penalties = centered_error_penalties / combined_scale
    
    regressor_dataset = CombinedDataset([mnist_dataset, scaled_centered_log_probs, scaled_centered_error_penalties])
    
    regressor = models.NetR32(opt)
    regressor.apply(models.weights_init)
    regressor = regressor.to(opt.device)
    
    train_regressor(regressor, generator, regressor_dataset, opt)
    
    


def regressor_histogram(opt):
    saved_regressor = torch.load('models/regressor_model_60_epochs_regressor.pth')
    regressor_dict = saved_regressor[0]
    saved_opt = saved_regressor[-1]
    regressor = models.NetR32(saved_opt)
    regressor.load_state_dict(regressor_dict)
    regressor = regressor.to(opt.device)
    regressor.eval()
    
    mnist = datasets.get_dataset(dataset='mnist')
    cifar = datasets.get_dataset(dataset='cifar10')
    
    
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=256, shuffle=False)
    cifar_loader = torch.utils.data.DataLoader(cifar, batch_size=256, shuffle=False)
    
    print("Sampling MNIST")
    all_mnist = []
    for i,batch in enumerate(mnist_loader):
        if i%20 == 0:
            print("Batch {0} of {1}".format(i, len(mnist_loader)))
        ims, _ = batch
        ims = ims.to(opt.device)
        outputs = regressor(ims).detach()
        all_mnist.append(outputs)
    
    print("Sampling Cifar")
    all_cifar = []
    for i,batch in enumerate(cifar_loader):
        if i%20 == 0:
            print("Batch {0} of {1}".format(i, len(cifar_loader)))        
        ims, _ = batch
        ims = ims.to(opt.device)
        outputs = regressor(ims).detach()
        all_cifar.append(outputs)
        
    all_mnist = torch.cat(all_mnist, dim=0).cpu().numpy()
    all_cifar = torch.cat(all_cifar, dim=0).cpu().numpy()
    
    
    print("MNIST on-manifold")
    plt.hist(all_mnist[:,0], bins=100, density=True)
    
    print("Cifar on-manifold")
    plt.hist(all_cifar[:,0], bins=100, density=True)
    plt.show()
    
    print("MNIST off-manifold")
    plt.hist(all_mnist[:,1], bins=100, density=True)

    
    print("Cifar off-manifold")
    plt.hist(all_cifar[:,1], bins=100, density=True)
    plt.show()


    print("Total mnist")
    plt.hist(all_mnist[:,0]+all_mnist[:,1], bins=100, density=True)
    
    
    print("Total cifar")
    plt.hist(all_cifar[:,0]+all_cifar[:,1], bins=100, density=True)
    plt.show()















if __name__ == '__main__':
    opt = parser.parse_args()
    
    if opt.mode == 'train':
        train_encoder_mnist_gan(opt)
    elif opt.mode == 'invert':
        invert_mnist_gan_with_encoder(opt)
    elif opt.mode == 'jacobian':
        sample_jacobian_mnist_gan(opt)
    elif opt.mode == 'regressor':
        train_regressor_mnist_gan(opt)
    elif opt.mode == 'histogram':
        regressor_histogram(opt)




















