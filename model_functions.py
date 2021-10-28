"""
Model update functions
"""

import utils
import torch
import numpy as np
import torchvision






def map_subcomponent_names_to_objects(cdict):
    cmap = {}
    if 'name' in cdict:
        if cdict.name is not None:
            cmap[cdict.name] = cdict.object_
    elif 'network' in cdict:
        for key in cdict:
            subcomponent = cdict[key]
            if subcomponent.name is not None:
                cmap[subcomponent.name] = subcomponent.object_
    else:
        for key in cdict:
            component = cdict[key]
            for subkey in component:
                subcomponent = component[subkey]
                if subcomponent.name is not None:
                    cmap[subcomponent.name] = subcomponent.object_
    
    return utils.EasyDict(**cmap)



# Copy-paste from other files
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d): 
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.normal_(m.weight,mean=0.0, std=0.4)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)




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


def discriminator_wgan_loss(fake_output, real_output, discriminator, fake_data, real_data, gp_lambda):
    gradient_penalty = calc_gradient_penalty(discriminator, real_data, fake_data, gp_lambda)
    return fake_output.mean() - real_output.mean() + gradient_penalty

def generator_wgan_loss(fake_discriminator_output):
    return (-1)*fake_discriminator_output.mean()   


def discriminator_regular_loss(fake_output, real_output):
    fake_targets = torch.zeros(len(fake_output), device=fake_output.device)
    real_targets = torch.ones(len(real_output), device=real_output.device)
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_output, fake_targets)
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_output, real_targets)
    return fake_loss + real_loss


def generator_regular_loss(fake_output):
    fake_targets = torch.ones(len(fake_output), device=fake_output.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_output, fake_targets)
    return loss

def cross_entropy_loss(predictions, targets):
    return torch.nn.functional.cross_entropy(predictions, targets)

def l2_loss(predictions, targets):
    return torch.nn.functional.mse_loss(predictions, targets)

# logsoftmax + nll
def bce_logits_loss(predictions, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)


loss_functions = {
    'wasserstein_gan':(discriminator_wgan_loss, generator_wgan_loss),
    'regular_gan':(discriminator_regular_loss, generator_regular_loss),
    'cross_entropy':cross_entropy_loss,
    'l2':l2_loss,
    'bce_logits':bce_logits_loss,
}


def get_loss_function(lossname):
    return loss_functions[lossname]


def stop(model):
    return False


def set_mode(model, mode):
    for key in model.components:
        component = model.components[key]
        if mode == 'train':
            component.network.object_.train()
        else:
            component.network.object_.eval()

def snapshot(model):
    gan_sample, _ = model.functions.sample(model, inputs=model.state.reference_codes)
    gan_sample = gan_sample.detach().cpu()
    generator_snapshot = torchvision.utils.make_grid(gan_sample, nrow=8, normalize=True) #value_range=(-1,1)) #value_range doesn't work for some pytorch versions
    return utils.EasyDict(**{'generator_snapshot': {'item':generator_snapshot, 'dtype':'image'}})



# base class
class Model_Function_Base:
    def __init__(self):
        pass
    def __call__(self):
        pass


class Set_Mode(Model_Function_Base):
    def __init__(self, omit=[]):
        self.omit = omit

    def __call__(self, model, mode):
        for key in model.components:
            if key not in self.omit:
                component_net = model.components[key].network.object_
                if mode == 'train':
                    component_net.train()
                else:
                    component_net.eval()
        model.state.mode = mode
                    


class GAN_Sample(Model_Function_Base):
    def __init__(self, model):
        self.latent_dimension = model.components.generator.network.opts.latent_dimension
        self.device = model.info.opts.device
#        self.generator = model.components.generator.network.object_

    def __call__(self, model, n_samples=64, inputs=None):
        if inputs is None:
            inputs = torch.randn(n_samples, self.latent_dimension, device=self.device)
        outputs = model.components.generator.network.object_(inputs)
    
        return outputs, inputs

class GAN_Update_Base(Model_Function_Base):
    def __init__(self, model, sample_func=None):
        name_obj_mapping = map_subcomponent_names_to_objects(model.components)
        for key in name_obj_mapping:
            setattr(self, key, name_obj_mapping[key])

        self.sample = sample_func if sample_func is not None else model.functions.sample

        self.critic_iters = model.info.opts.critic_iters
        self.gen_iters = model.info.opts.gen_iters

        self.state = utils.EasyDict()

        # State of learning rate scheduler
        self.state.generator_schedule_iter = iter(model.info.opts.generator_lr_schedule)
        self.state.discriminator_schedule_iter = iter(model.info.opts.discriminator_lr_schedule)
        self.state.generator_schedule_next = next(self.state.generator_schedule_iter, None)
        self.state.discriminator_schedule_next = next(self.state.discriminator_schedule_iter, None)

        # State counter for network updates (decides when to update discriminator vs generator)
        self.state.itercounter = 0

        discriminator_loss_func, generator_loss_func = get_loss_function(model.info.opts.loss_function) #confusing names; fix later
        self.discriminator_loss = discriminator_loss_func
        self.generator_loss = generator_loss_func
        
        self.model = model
        self.opts = model.info.opts
        self.device = model.info.opts.device


class Wasserstein_GAN_Update(GAN_Update_Base):
    def __init__(self, model, sample_func=None):
        super().__init__(model, sample_func)
       
    def __call__(self, model, batch, epoch, iternum):
        # Step learning rate if indicated
        if epoch == self.state.generator_schedule_next:
            print("Stepping generator learning rate")
            self.state.generator_schedule_next = next(self.state.generator_schedule_iter, None)
            self.generator_scheduler.step()
        if epoch == self.state.discriminator_schedule_next:
            print("Stepping discriminator learning rate")
            self.state.discriminator_schedule_next = next(self.state.discriminator_schedule_iter, None)
            self.discriminator_scheduler.step()

        # Update discriminator on batch
        real_batch = batch.image.to(self.device)
        batch_size = len(real_batch)

        discriminator_on_real = self.discriminator(real_batch)
        fake_ims, _ = self.sample(model, n_samples=batch_size)
        fake_ims = fake_ims.detach()
        discriminator_on_fake = self.discriminator(fake_ims)
        discriminator_loss = self.discriminator_loss(discriminator_on_fake, discriminator_on_real, self.discriminator, fake_ims, real_batch, self.opts.gp_lambda)
        
        self.discriminator.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optim.step()

        discriminator_loss_value = discriminator_loss.item()

        generator_loss_value = None
                
        # Update generator if indicated
        if (self.state.itercounter+1) == self.critic_iters:
            generator_loss_value = 0.0
            for i in range(self.gen_iters):
                fake_ims, _ = self.sample(model, n_samples=batch_size)
                discriminator_on_fake = self.discriminator(fake_ims)
                generator_loss = self.generator_loss(discriminator_on_fake)
                
                self.generator.zero_grad()
                generator_loss.backward()
                self.generator_optim.step()
                
                generator_loss_value += generator_loss.item()

            generator_loss_value /= self.gen_iters
            
        # Increment state counter
        self.state.itercounter = (self.state.itercounter+1) % self.critic_iters

        # Collect metrics
        metric_dict = {'train/discriminator_loss': discriminator_loss_value}
        
        if generator_loss_value is not None: 
            metric_dict['train/generator_loss'] = generator_loss_value

        return utils.dict_from_paths(metric_dict)


class Regular_GAN_Update(GAN_Update_Base):
    def __init__(self, model, sample_func=None):
        super().__init__(model, sample_func)
       
    def __call__(self, model, batch, epoch, iternum):
        # Step learning rate if indicated
        if epoch == self.state.generator_schedule_next:
            print("Stepping generator learning rate")
            self.state.generator_schedule_next = next(self.state.generator_schedule_iter, None)
            self.generator_scheduler.step()
        if epoch == self.state.discriminator_schedule_next:
            print("Stepping discriminator learning rate")
            self.state.discriminator_schedule_next = next(self.state.discriminator_schedule_iter, None)
            self.discriminator_scheduler.step()

        # Update discriminator on batch
        real_batch = batch.image.to(self.device)
        batch_size = len(real_batch)

        discriminator_on_real = self.discriminator(real_batch)
        fake_ims, _ = self.sample(model, n_samples=batch_size)
        fake_ims = fake_ims.detach()
        discriminator_on_fake = self.discriminator(fake_ims)
        discriminator_loss = self.discriminator_loss(discriminator_on_fake, discriminator_on_real)
        
        self.discriminator.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optim.step()

        discriminator_loss_value = discriminator_loss.item()
        
        # Update generator if indicated
        if (self.state.itercounter+1) == self.critic_iters:
            generator_loss_value = 0.0
            for i in range(self.gen_iters):
                fake_ims, _ = self.sample(model, n_samples=batch_size)
                discriminator_on_fake = self.discriminator(fake_ims)
                generator_loss = self.generator_loss(discriminator_on_fake)
                
                self.generator.zero_grad()
                generator_loss.backward()
                self.generator_optim.step()
                
                generator_loss_value += generator_loss.item()

            generator_loss_value /= self.gen_iters
            
        # Increment state counter
        self.state.itercounter = (self.state.itercounter+1) % self.critic_iters

        # Collect metrics
        metric_dict = {'train/discriminator_loss': discriminator_loss_value, 'train/generator_loss' : generator_loss_value}

        return utils.dict_from_paths(metric_dict)
       
   
"""
===================================================================================
Predictor updates (encoder, regressor, classifier, etc.)
===================================================================================
"""


class Predictor_Update_Base(Model_Function_Base):
    def __init__(self, model):
        name_obj_mapping = map_subcomponent_names_to_objects(model.components)
        for key in name_obj_mapping:
            setattr(self, key, name_obj_mapping[key])

        self.state = utils.EasyDict()

        # State of learning rate scheduler
        self.state.schedule_iter = iter(model.info.opts.lr_schedule)
        self.state.schedule_next = next(self.state.schedule_iter, None)

        # State counter for network updates (decides when to update discriminator vs generator)
        self.state.itercounter = 0

        self.loss_function = get_loss_function(model.info.opts.loss_function) #confusing names; fix later
        
        self.model = model
        self.opts = model.info.opts
        self.device = model.info.opts.device

        self.scheduler = None #Set this in derived class

    def _step_lr(self, epoch, iternum):
        if epoch == self.state.schedule_next:
            print("Stepping learning rate")
            self.state.schedule_next = next(self.state.schedule_iter, None)
            self.scheduler.step()

    def __call__(self):
        pass

    def run(self):
        pass


# Implement _step_lr submethod
# implement basic calls and runs (fast)
# for gan_encoder, if batch is None, invoke generator

class Encoder_Update(Predictor_Update_Base):
    def __init__(self, model): #, sample_func=None):
        super().__init__(model)
        self.scheduler = self.encoder_scheduler
#        self.sample = sample_func

    def __call__(self, model, batch, epoch, iternum):
        self._step_lr(epoch, iternum)

        outputs, metrics = self.run(self.model, batch)

        self.encoder.zero_grad()
        outputs.encoder_loss.backward()
        self.encoder_optim.step()
            
        return metrics
    
    def run(self, model, batch):
        image_batch = batch.fake_outputs.to(self.device)
        encodings = self.encoder(image_batch)
        outputs = utils.EasyDict(encodings=encodings)
        metrics = {}
        if 'latent_codes' in batch:
            labels = batch.latent_codes.to(self.device)
            encoder_loss = self.loss_function(encodings, labels)
            encoder_loss_value = encoder_loss.item()
            outputs.encoder_loss = encoder_loss
            metrics['{0}/encoder_loss'.format(model.state.mode)] = encoder_loss_value

        return outputs, utils.dict_from_paths(metrics)
        

class Regressor_Update(Predictor_Update_Base):
    def __init__(self, model):
        super().__init__(model)
        if model.info.opts.training:
            self.scheduler = self.regressor_scheduler

    def __call__(self, model, batch, epoch, iternum):
        self._step_lr(epoch, iternum)

        outputs, metrics = self.run(self.model, batch)

        self.regressor.zero_grad()
        outputs.regressor_loss.backward()
        self.regressor_optim.step()
            
        return metrics
       
    def run(self, model, batch):
        image_batch = batch.image.to(self.device)
        encodings = self.regressor(image_batch)
        outputs = utils.EasyDict(encodings=encodings)
        metrics = {}

        if 'logprobs' in batch:
            labels = torch.cat([batch.logprobs.unsqueeze(1), batch.reconstruction_losses.unsqueeze(1)],dim=1)
            labels = labels.to(self.device)
            regressor_loss = self.loss_function(encodings, labels)
            regressor_loss_value = regressor_loss.item()
            outputs.regressor_loss = regressor_loss
            metrics['{0}/regressor_loss'.format(model.state.mode)] = regressor_loss_value

        return outputs, utils.dict_from_paths(metrics)


class Classifier_Update(Predictor_Update_Base):
    def __init__(self, model):
        super().__init__(model)
        self.scheduler = self.classifier_scheduler

    def __call__(self, model, batch, epoch, iternum):
        self._step_lr(epoch, iternum)

        outputs, metrics = self.run(self.model, batch)

        self.classifier.zero_grad()
        outputs.classifier_loss.backward()
        self.classifier_optim.step()
            
        return metrics
    
    def run(self, model, batch):
        image_batch = batch.image.to(self.device)
        encodings = self.classifier(image_batch)
        outputs = utils.edict(encodings=encodings)
        metrics = {}
        if 'labels' in batch:
            labels = batch.labels.to(self.device)
            classifier_loss = self.loss_function(encodings, labels)
            classifier_loss_value = discriminator_loss.item()
            outputs.classifier_loss = classifier_loss
            metrics['{0}/classifier_loss'.format(model.state.mode)] = classifier_loss_value

        return outputs, utils.dict_from_paths(metrics)




class Invert_Generator(Model_Function_Base):
    def __init__(self, model):
        self.generator = model.components.generator.network.object_
        self.device = model.info.opts.device

        self.encoder = model.components.encoder.network.object_

        self.encoder_guidance_lambda = model.info.opts.encoder_guidance_lambda

        self.lossfunc = get_loss_function('l2')        

        self.inversion_opts = model.info.opts.inversion_opts


    def __call__(self, model, batch):
        images = batch.image.to(self.device)
        batch_size = images.size(0)

        initial_guesses = self.encoder(images) #initial guesses

        guesses = initial_guesses.clone().detach()
        guesses.requires_grad_(True)
        
        optim = torch.optim.Adam([guesses], lr=self.inversion_opts.latent_lr, betas=self.inversion_opts.latent_betas)
        grad_norms = torch.ones(batch_size)
        iters = 0
        while grad_norms.max() > 0.001 and iters<100: #This is the wrong approach; need to test z convergence directly
            fake_ims = self.generator(guesses)
            reconstruction_losses = ((fake_ims.view(batch_size,-1) - images.view(batch_size, -1))**2).mean(dim=1)
#            reconstruction_loss = self.lossfunc(fake_ims, images)
            reconstruction_loss = reconstruction_losses.mean()
            encoder_guidance_loss = self.lossfunc(guesses,initial_guesses)
                                    # self.lossfunc(self.encoder(guesses), guesses)
            
            loss = reconstruction_loss + self.encoder_guidance_lambda*encoder_guidance_loss
            
            if guesses.grad is not None:
                guesses.grad.fill_(0.0)
            loss.backward(retain_graph=True)
            grad_norms = torch.norm(guesses.grad, dim=1)
            optim.step()
            iters += 1

        return utils.DataDict(latent_codes = guesses.detach().cpu(), reconstruction_losses=reconstruction_losses.detach().cpu(), _size=batch_size)


class Log_Jacobian_Determinant(Model_Function_Base):
    def __init__(self, model):
        self.generator = model.components.generator.network.object_
        self.device = model.info.opts.device

    def __call__(self, model, batch):
        z_codes = batch.latent_codes
        z_codes = z_codes.to(self.device)
        z_codes.requires_grad_(True)
        output = self.generator(z_codes)
        
        num_points = z_codes.size(0)
        output_size = output.numel() // num_points
        input_size = z_codes.numel() // num_points
        
        #print("Starting gradients")
        gradients = [torch.autograd.grad(outputs=output[:,k], inputs=z_codes,
                              grad_outputs=torch.ones(num_points, device=self.device),
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
        
        return utils.DataDict(logprobs=logprobs, _size=num_points)







