#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regressor sampling/testing
"""


### Imports

import argparse
import os

import torch
import numpy as np

import models




### Command line arguments


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Sample, train, or test')
parser.add_argument('--loadfrom', type=str, default=None, help='Model to load from')
parser.add_argument('--saveto', type=str, default='model', help='Base name to save model to for checkpointing')
parser.add_argument('--modeldir', type=str, default='models', help='Directory where models are located')

# parameter arguments
parser.add_argument('--trainiters', type=int, default=1000, help='Training iterations')
parser.add_argument('--checkpoint_every', type=int, default=5, help='How often to checkpoint the model')








opt=parser.parse_args()







### Jacobian sampling and saving

def sample_jacobian(generator, inputs, true_outputs, saveto, batch_size=128, method='analytical'):
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=batch_size, shuffle=False, drop_last=False)
    
    if method == 'analytical':
        
        all_logprobs = []
        all_rmse = []
        for i, pts in enumerate(dataloader):
            z_codes, ims = pts
            pts.requires_grad_(True)
            output = generator(z_codes)
            num_points = z_codes.size(0)
            output_size = output.numel() // num_points
            input_size = z_codes.numel() // num_points
            
            output = output.view(output_size, -1) #flatten outputs
            true_outputs = true_outputs.view(output_size, -1)
            rmse_vals = torch.sqrt(((output-true_outputs)**2).mean(dim=1))
            
            gradients = [torch.autograd.grad(outputs=output[:,k], inputs=pts,
                                      grad_outputs=torch.ones(num_points),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0] for k in range(output.size(1))]
    
            
            gradients = torch.cat(gradients,dim=1)#.detach()
            gradients = gradients.reshape(num_points, output_size, input_size)
            
            gradients = gradients.detach().cpu().numpy()
    
    
            dim = input_size
            gauss_constant = (-dim / 2)*torch.log(4*torch.acos(torch.tensor(0.0))) # -dim/2 * log(2*pi)
            z_log_likelihood = gauss_constant + (-0.5)*(pts**2).sum(dim=1)
            
            z_log_likelihood = z_log_likelihood.detach().cpu().numpy()

            # Compute determinants with QR decomp:
            logprobs = np.zeros(num_points)
            for k in range(num_points):
                R = np.linalg.qr(gradients[k], mode='r')
                jacobian_score = -np.log(np.abs(R.diagonal())).sum() #negative because inverting the diagonal elements
                log_prob = z_log_likelihood[k] + jacobian_score
                logprobs[k] = log_prob
            
            logprobs = torch.from_numpy(logprobs)
            
            all_logprobs.append(logprobs)
            all_rmse.append(rmse_vals)
            
        logprobs = torch.cat(all_logprobs)
        rmse_vals = torch.cat(all_rmse)
        
        mean_rmse = rmse_vals.mean()
        lmbda = 1.0/mean_rmse
        log_penalties = torch.log(lmbda) - lmbda*rmse_vals
        
        combined = logprobs + log_penalties
        
        
        return logprobs, rmse_vals, log_penalties, combined
            
### Loading saved samples into dataset


def map_images_to_latent(generator, encoder, dataset, savefile, opt):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    
    generator.eval()
    all_codes = []
    all_metrics = []
    for i, batch in enumerate(dataloader):
        batch = batch[0].to(opt.device)
        batch_size = batch.size(0)
        #codes = torch.randn(batch_size, opt.nz, device=opt.device)
        codes = encoder(batch)
        codes.requires_grad_(True)
        optim = torch.optim.Adam([codes])
        losses = torch.ones(batch_size)
        iters = 0
        while losses.max() > 0.05 and iters<20: #This is the wrong approach; need to test z convergence directly
            fake_ims = generator(codes)
            fake_ims = fake_ims.view(batch_size, -1)
            batch = batch.view(batch_size, -1)
            losses = ((fake_ims-batch)**2).mean(dim=1)
            loss = losses.sum()
            #generator.zero_grad() #probably not necessary, but just in case of numerical issues
            codes.grad.fill_(0.0)
            loss.backward()
            optim.step()
            iters += 1
        all_codes.append(codes.detach().cpu())
        all_metrics.append(losses)
        print("Batch {0}, iters {1}, max loss {2}".format(i, iters, loss.max()))
        
    all_codes = torch.cat(all_codes,dim=0)
    all_metrics = torch.cat(all_metrics)
    torch.save( (all_codes,all_metrics) , savefile)


# Procedure: train encoder to invert GAN samples, using mixed reconstruction loss / discriminator objective
# For dataset points: Run through encoder to initialize z values, then use mixed objective to optimize individual z points
def train_encoder(generator, discriminator, savefile, opt):
    encoder = None #fill in
    encoder_optim = torch.optim.Adam(encoder.params(), lr=opt.encoder_lr, betas=opt.encoder_betas, weight_decay=opt.encoder_weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optim, step_size=opt.encoder_lr_step_size, gamma=opt.encoder_lr_gamma)
    
    all_losses = []
    for i in range(opt.encoder_iters):
        codes = torch.randn(opt.batch_size, opt.nz, device=opt.device)
        images = generator(codes).detach()
        
        predictions = encoder(images)
        reconstruction_loss = ((predictions-codes)**2).mean()
        discriminator_loss = (-1)*discriminator(generator(predictions)).mean()
        loss = reconstruction_loss + opt.encoder_lambda*discriminator_loss
        
        encoder.zero_grad()
        loss.backward()
        encoder_optim.step()
        
        all_losses.append(loss.item())
        
        if opt.encoder_step_lr == i:
            encoder_scheduler.step()
        
        if i%opt.print_every == 0:
            print("Encoder loss = {0}".format(loss.item()))
            
        if (i+1)%opt.checkpoint_every == 0:
            torch.save((encoder.state_dict(), encoder_optim.state_dict(), encoder_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'_'+str(i)+'.pth')
        
        
        
    torch.save((encoder.state_dict(), encoder_optim.state_dict(), encoder_scheduler.state_dict(), all_losses, opt), os.path.join(opt.modeldir, opt.basename)+'.pth')
    





def get_regressor_dataset(loadfrom):
    pass
    








### Training regressor model, checkpointing, extracting metrics


def train_regressor(model, dataset, opts):
    pass









### Testing regressor model

def test_regressor():
    pass













