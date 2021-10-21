"""
Model update functions
"""

import utils
import torch




"""
To implement: 
    (load_torch_class)
    (loading model weights / initializing model in build_model) 
    get_loss_function
    generator_wgan_loss
    discriminator_wgan_loss
    generator_regular_loss
    discriminator_regular_loss

All the stuff for regressor/classifier/etc.    
    
"""


def map_subcomponent_names_to_objects(cdict):
    cmap = {}
    if 'name' in cdict:
        cmap[cdict.name] = cdict.object_
    elif 'network' in cdict:
        for key in cdict:
            subcomponent = cdict[key]
            cmap[subcomponent.name] = subcomponent.object_
    else:
        for key in cdict:
            component = cdict[key]
            for subkey in component:
                subcomponent = component[subkey]
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
            #torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.weight,mean=0.0, std=0.4)
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


loss_functions = {
    'wasserstein_gan':(discriminator_wgan_loss, generator_wgan_loss),
    'regular_gan':(discriminator_regular_loss, generator_regular_loss),
}


def get_loss_function(lossname):
    return loss_functions[lossname]




def set_mode(model, mode):
    for key in model.components:
        component = model.components[key]
        if mode == 'train':
            component.network.object_.train()
        else:
            component.network.object_.eval()

def snapshot(model):
    gan_sample = model.functions.sample(model, codes=model.state.reference_codes).detach().cpu()
    generator_snapshot = torchvision.utils.make_grid(gan_sample, nrow=8, normalize=True, value_range=(-1,1))
    return utils.EasyDict(**{'generator_snapshot': {'item':generator_snapshot, 'dtype':'image'}})



# base class
class Model_Function_Base:
    def __init__(self):
        pass
    def __call__(self):
        pass

class GAN_Sample(Model_Function_Base):
    def __init__(self, model):
        self.latent_dimension = model.info.opts.generator_opts.opts.latent_dimension
        self.device = model.info.opts.device
        self.generator = model.components.generator.network.object_

    def __call__(self, model, n_samples=64, inputs=None):
        if inputs is None:
            inputs = torch.randn(n_samples, self.sample_dim, device=self.device)
        outputs = self.generator_net(inputs)
    
        return outputs, inputs

class GAN_Update_Base(Model_Function_Base):
    def __init__(self, model, sample_func=None):
#        self.generator = model.components.generator.network.object_
#        self.generator_optim = model.components.generator.optimizer.object_
#        self.generator_scheduler = model.components.generator.scheduler.object_
#        
#        self.discriminator = model.components.discriminator.network.object_
#        self.discriminator_optim = model.components.discriminator.optimizer.object_
#        self.discriminator_scheduler = model.components.discriminator.scheduler.object_

        name_obj_mapping = map_subcomponent_names_to_objects(model.components)
        for key in name_obj_mapping:
            setattr(self, key, name_obj_mapping[key])

        self.sample = sample_func if sample_func is not None else model.functions.sample

        self.critic_iters = model.info.opts.critic_iters
        self.gen_iters = model.info.opts.gen_iters

        self.state = utils.EasyDict()

        # State of learning rate scheduler
        self.state.generator_schedule_iter = iter(model.info.opt.generator_lr_schedule)
        self.state.discriminator_schedule_iter = iter(model.info.opt.discriminator_lr_schedule)
        self.state.generator_schedule_next = next(self.generator_schedule_iter, None)
        self.state.discriminator_schedule_next = next(self.discriminator_schedule_iter, None)

        # State counter for network updates (decides when to update discriminator vs generator)
        self.state.itercounter = 0

        discriminator_loss_func, generator_loss_func = get_loss_function(model.info.opt.loss_function) #confusing names; fix later
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
            self.state.generator_schedule_next = next(self.generator_schedule_iter, None)
            self.generator_scheduler.step()
        if epoch == self.state.discriminator_schedule_next:
            self.state.discriminator_schedule_next = next(self.discriminator_schedule_iter, None)
            self.discriminator_scheduler.step()

        # Update discriminator on batch
        real_batch = model.x.to(self.device)
        batch_size = len(image_batch)

        discriminator_on_fake = self.discriminator(real_batch)
        fake_ims, _ = self.sample(model, n_samples=batch_size)
        fake_ims = fake_ims.detach()
        discriminator_on_real = self.discriminator(fake_ims)
        discriminator_loss = self.discriminator_loss(discriminator_on_fake, discriminator_on_real, self.discriminator, fake_ims, real_batch, self.opts.gp_lambda)
        
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


class Regular_GAN_Update(GAN_Update_Base):
    def __init__(self, model, sample_func=None):
        super().__init__(model, sample_func)
       
    def __call__(self, model, batch, epoch, iternum):
        # Step learning rate if indicated
        if epoch == self.state.generator_schedule_next:
            self.state.generator_schedule_next = next(self.generator_schedule_iter, None)
            self.generator_scheduler.step()
        if epoch == self.state.discriminator_schedule_next:
            self.state.discriminator_schedule_next = next(self.discriminator_schedule_iter, None)
            self.discriminator_scheduler.step()

        # Update discriminator on batch
        real_batch = model.x.to(self.device)
        batch_size = len(image_batch)

        discriminator_on_fake = self.discriminator(real_batch)
        fake_ims, _ = self.sample(model, n_samples=batch_size)
        fake_ims = fake_ims.detach()
        discriminator_on_real = self.discriminator(fake_ims)
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
       
   

















