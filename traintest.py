#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and test image models
"""





### Imports

import argparse


import torch
import torchvision


import matplotlib.pyplot as plt


import models
import datasets




parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Train or test')
parser.add_argument('--method', type=str, default='wgan', help='The GAN model type being used')
parser.add_argument('--loadfrom', type=str, default=None, help='Model to load from')
parser.add_argument('--basename', type=str, default='model', help='Base name to save model to for checkpointing')
parser.add_argument('--modeldir', type=str, default='models', help='Directory where models are located')
parser.add_argument('--dataset', type=str, default='mnist', help='Which datset to train on')


parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help="Adam betas")
parser.add_argument('--eps', type=float, default=1e-8, help="Adam epsilon")
parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay parameter for optimizer")
parser.add_argument('--amsgrad', action='store_true', help="Adam amsgrad parameter")
parser.add_argument('--lr_step_size', type=int, default=1, help="LR scheduler step size")
parser.add_argument('--lr_gamma', type=float, default=0.1, help="LR scheduler scale factor")
parser.add_argument('--nc', type=int, default=3, help="Number of input channels in datasets")
parser.add_argument('--ndf', type=int, default=64, help="Base number of hidden features in discriminator and encoder")
parser.add_argument('--ngf', type=int, default=64, help="Base number of hidden features in generator")
parser.add_argument('--nz', type=int, default=100, help="Size of generator input space")

parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
#parser.add_argument('--drop_last', action='store_true', help="Drop last batch if not full size")
parser.add_argument('--step_threshold', type=int, default=10000, help="Number of iters before stepping learning rate")
parser.add_argument('--num_iters', type=int, default=100000, help="Number of discriminator/generator iters")
parser.add_argument('--checkpoint_every', type=int, default=1000, help="How many iters to checkpoint after")
parser.add_argument('--critic_iters', type=int, default=5, help="Number of discriminator updates at once")
parser.add_argument('--gen_iters', type=int, default=1, help="Number of generator updates at once")

parser.add_argument('--device', type=str, default='cuda:0', help='Where to train the model')
parser.add_argument('--gp_lambda', type=float, default=10.0, help="Gradient penalty coefficient for wasserstein loss")

parser.add_argument('--print_every', type=int, default=100, help="How often to print loss updates")
parser.add_argument('--image_every', type=int, default=1000, help="How often to save sample images")







def calc_gradient_penalty(discriminator_net, real_data, fake_data, gp_lambda):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size))#.contiguous()
    alpha = alpha.reshape(real_data.size())
    
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    disc_interpolates = discriminator_net(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty    


            
def calc_bigan_gradient_penalty(discriminator_net, encoded_real, real_data, z_codes, fake_data, gp_lambda):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    alpha_ims = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha_ims = alpha_ims.reshape(real_data.size())
    alpha_codes = alpha.expand(batch_size, int(encoded_real.nelement()/batch_size)).contiguous()
    
    
    # change in order to capture both the codes and actual images
    interpolates_ims = alpha_ims*real_data + (1-alpha_ims)*fake_data
    interpolates_ims = interpolates_ims.detach()
    interpolates_ims.requires_grad_(True)
    
    interpolates_codes = alpha_codes*encoded_real + (1-alpha_codes)*z_codes
    interpolates_codes = interpolates_codes.detach()
    interpolates_codes.requires_grad_(True)

    disc_interpolates = discriminator_net(interpolates_codes, interpolates_ims)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=[interpolates_codes, interpolates_ims],
                              grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)

    gradients_codes = gradients[0].view(gradients[0].size(0), -1) 
    gradients_ims = gradients[1].view(gradients[1].size(0), -1)
    gradients = torch.cat([gradients_codes, gradients_ims], dim=1)                             
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty




def train_regular_gan(model, dataset, opt):
    model.netg.train()
    model.netd.train()
    
    
    #if opt.pretraining:
    #    pass
        ## TODO: pretraining code to initialize the model
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    
    stepthreshold = opt.step_threshold

    
    discriminator_losses = []
    generator_losses = []
    sample_store = {}
    
    
    epoch = 0
    dataiter = iter(dataloader)
    for iternum in range(opt.num_iters):
        for i in range(opt.critic_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
                
            batch = batch[0] #batch[1] is the label    
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)           

            gen_fake = model.netg(z_codes).detach()
            
            disc_real = model.netd(batch)
            disc_fake = model.netd(gen_fake)
            
            disc_cost = torch.nn.functional.binary_cross_entropy_with_logits(disc_real, torch.ones(disc_real.size(),device=opt.device)) +\
                                torch.nn.functional.binary_cross_entropy_with_logits(disc_fake, torch.zeros(disc_fake.size(),device=opt.device))
            
            discriminator_losses.append(disc_cost.item())
            
            model.netd.zero_grad()
            disc_cost.backward()
            model.netd_optim.step()
            
        for i in range(opt.gen_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
            
            batch = batch[0]
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)          

            fake_data = model.netg(z_codes)
            
            disc_fake_labels = model.netd(fake_data)
            
            gen_fake_cost = torch.nn.functional.binary_cross_entropy_with_logits(disc_fake_labels, torch.ones(disc_fake_labels.size(),device=opt.device))
            
            generator_losses.append(gen_fake_cost.item())
            
            total_cost = gen_fake_cost
            
            
            
            model.netg.zero_grad()
            total_cost.backward()
            model.netg_optim.step()
            
            
            if (iternum+1)%opt.image_every== 0:
                samples = torchvision.utils.make_grid(fake_data, nrow=8)
                sample_store[iternum] = samples
                samples_processed = samples.detach().permute(1,2,0).cpu().numpy()
                samples_processed = (samples_processed+1)/2
                print("Showing samples at iter", iternum)
                plt.imshow(samples_processed)
                plt.show()
            
        if (iternum+1)%opt.print_every == 0:
            print("Epoch: {0}, batch_iter: {1}, disc_loss: {2}, gen_loss: {3}".format(epoch, iternum, discriminator_losses[-1], generator_losses[-1]))

        
        # step learning rates
        if iternum == stepthreshold:
            print("Stepping learning rate")
            model.netg_scheduler.step()
            model.netd_scheduler.step()
            
            #stepthreshold += 1000 ###
        
        if (iternum+1)%opt.checkpoint_every == 0:
            print("Checkpointing at iter", iternum)
            model.checkpoint(number=iternum+1, metrics=(sample_store, discriminator_losses, generator_losses))
        
            
    model.checkpoint(metrics=(sample_store, discriminator_losses, generator_losses)) 




def train_wgan(model, dataset, opt):
    model.netg.train()
    model.netd.train()
    
    
    #if opt.pretraining:
    #    pass
        ## TODO: pretraining code to initialize the model
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    
    stepthreshold = opt.step_threshold
    # assume optimizers and schedulers are part of the model I guess
    
    
    # if cuda, model should already be cuda
    
    discriminator_losses = []
    generator_losses = []
    sample_store = {}
    
    
    epoch = 0
    dataiter = iter(dataloader)
    for iternum in range(opt.num_iters):
        for i in range(opt.critic_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
                
            batch = batch[0] #batch[1] is the label    
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)        

            gen_fake = model.netg(z_codes).detach()
            
            disc_real = model.netd(batch).mean()
            disc_fake = model.netd(gen_fake).mean()
            gradient_penalty = calc_gradient_penalty(model.netd, batch, gen_fake, opt.gp_lambda)
            
            disc_cost = disc_fake - disc_real + gradient_penalty
            
            discriminator_losses.append(disc_cost.item())
            
            model.netd.zero_grad()
            disc_cost.backward()
            model.netd_optim.step()
            
        for i in range(opt.gen_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
            
            batch = batch[0]
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)
            z_codes = z_codes.to(opt.device)            

            fake_data = model.netg(z_codes)
            
            gen_fake_cost = (-1)*model.netd(fake_data).mean()
            
            generator_losses.append(gen_fake_cost.item())
            
            total_cost = gen_fake_cost
            
            
            
            model.netg.zero_grad()
            total_cost.backward()
            model.netg_optim.step()
            
            
            if (iternum+1)%opt.image_every == 0 and i==0:
                samples = torchvision.utils.make_grid(fake_data[:min(64,fake_data.size(0))], nrow=8)
                sample_store[iternum] = samples
                samples_processed = samples.detach().permute(1,2,0).cpu().numpy()
                samples_processed = (samples_processed+1)/2
                print("Showing samples at iter", iternum)
                plt.imshow(samples_processed)
                plt.show()
            
        if (iternum+1)%opt.print_every == 0:
            print("Epoch: {0}, batch_iter: {1}, disc_loss: {2}, gen_loss: {3}".format(epoch, iternum, discriminator_losses[-1], generator_losses[-1]))

        
        # step learning rates
        if iternum == stepthreshold:
            print("Stepping learning rate")
            model.netg_scheduler.step()
            model.netd_scheduler.step()
            
        
        if (iternum+1)%opt.checkpoint_every == 0:
            print("Checkpointing at iter", iternum)
            model.checkpoint(number=iternum+1, metrics=(sample_store, discriminator_losses, generator_losses))
        

            
    model.checkpoint(metrics=(sample_store, discriminator_losses, generator_losses)) 




def train_bigan(model, dataset, opt):
    model.netg.train()
    model.netd.train()
    model.nete.train()
    
    
    #if opt.pretraining:
    #    pass
        ## TODO: pretraining code to initialize the model
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    
    stepthreshold = opt.step_threshold
    
    discriminator_losses = []
    generator_losses = []
    encoder_losses = []
    sample_store = {}
    
    
    epoch = 0
    dataiter = iter(dataloader)
    for iternum in range(opt.num_iters):
        for i in range(opt.critic_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
                
            batch = batch[0] #batch[1] is the label    
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)
            
            encoded_real = model.nete(batch).detach()

            gen_fake = model.netg(z_codes).detach()
            
            disc_real = model.netd(encoded_real, batch).mean()
            disc_fake = model.netd(z_codes, gen_fake).mean()
            gradient_penalty = calc_bigan_gradient_penalty(model.netd, encoded_real, batch, z_codes, gen_fake, opt.gp_lambda)
            
            disc_cost = disc_fake - disc_real + gradient_penalty
            
            discriminator_losses.append(disc_cost.item())
            
            model.netd.zero_grad()
            disc_cost.backward()
            model.netd_optim.step()
            
        for i in range(opt.gen_iters):
            batch = next(dataiter, None)
            if batch is None:
                epoch += 1
                dataiter = iter(dataloader)
                batch = dataiter.next()
            
            batch = batch[0]
            z_codes = torch.randn(batch.size(0), opt.nz, device=opt.device)
            
            batch = batch.to(opt.device)         
            
            
            encoded_real = model.nete(batch)
            fake_data = model.netg(z_codes)
            
            gen_fake_cost = (-1)*model.netd(z_codes, fake_data).mean()
            enc_real_cost = model.netd(encoded_real, batch).mean()
            
            generator_losses.append(gen_fake_cost.item())
            encoder_losses.append(enc_real_cost.item())
            
            total_cost = gen_fake_cost + enc_real_cost
            
            
            
            model.netg.zero_grad()
            model.nete.zero_grad()
            total_cost.backward()
            model.netg_optim.step()
            model.nete_optim.step()
            
            
            if ((iternum+1)%opt.image_every == 0) and (i==0):
                samples = torchvision.utils.make_grid(fake_data[:min(64,fake_data.size(0))], nrow=8)
                sample_store[iternum] = samples
                samples_processed = samples.detach().permute(1,2,0).cpu().numpy()
                samples_processed = (samples_processed+1)/2
                print("Showing samples at iter", iternum)
                plt.imshow(samples_processed)
                plt.show()
            
        if (iternum+1)%opt.print_every == 0:
            print("Epoch: {0}, batch_iter: {1}, disc_loss: {2}, gen_loss: {3}, enc_loss: {4}".format(epoch, iternum, discriminator_losses[-1], generator_losses[-1], encoder_losses[-1]))

        
        # step learning rates
        if iternum == stepthreshold:
            print("Stepping learning rate")
            model.netg_scheduler.step()
            model.netd_scheduler.step()
            model.nete_scheduler.step()
            
            #stepthreshold += 1000 ###
        
        if (iternum+1)%opt.checkpoint_every == 0:
            print("Checkpointing at iter", iternum)
            model.checkpoint(number=iternum+1, metrics=(sample_store, discriminator_losses, generator_losses, encoder_losses)) 
        
            
    model.checkpoint(metrics=(sample_store, discriminator_losses, generator_losses, encoder_losses))



### Loading and testing gan models


def run_bigan_training(opt):
    model = models.BiGAN_Model(opt)
    dset = datasets.get_dataset(dataset=opt.dataset)
    train_bigan(model, dset, opt)



def run_wgan_training(opt):
    model = models.GAN_Model(opt)
    dset = datasets.get_dataset(dataset=opt.dataset)
    train_wgan(model, dset, opt)


def run_regular_training(opt):
    model = models.GAN_Model(opt)
    dset = datasets.get_dataset(dataset=opt.dataset)
    train_regular_gan(model, dset, opt)




if __name__=='__main__':
    opt=parser.parse_args()
    
    if opt.method == 'wgan':
        run_wgan_training(opt)
    elif opt.method == 'bigan':
        run_bigan_training(opt)
    elif opt.method == 'regular':
        run_regular_training(opt)









