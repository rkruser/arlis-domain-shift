"""
Build full runnable models given config/command line args
"""

import os

import models
import utils
import model_functions

import copy

import torch
import torchvision



"""
Next steps:
1.         Write all individual model functions [1 hr optimistically?]  <---- Still here (>2 hours later I think)
1.1        Write GAN loss functions and other misc functions and integrate them with code
1.2        Move on to 1.5, 2, 3, before looping back to write other functions for other models

1.5        Write dataset wrapper for EasyDicts [30 mins optimistically]
2.         Fully specify opts in a config file or something [1 hr?]
3.         Look over whole thing and debug whole training procedure with specific arguments [???]
3.5        Maybe add extensive comments [30 mins]
4.         Write regressor / encoder / classifier build functions [30 mins since infrastructure in place]
5.         Write full experiment functions [???]
6.         Test full experiments [???]
7.         Make runnable on Vulcan [??? optional for today] 
7.5        Parallelize models more [??? optional for today]
8.         Nice matplotlib visualizer functions
9.         Run actual experiments
"""

"""
Misc notes:
Your work is somewhat related to:
- Knowledge distillation
- That one paper Pedro presented the other day
"""



"""
This file to do:
    - Check over all names of things for consistency
    - Check over logic
"""



"""
Functions for outlining model structure

empty_subcomponent: Returns a standard empty structured subcomponent
model_outline takes lists of component names and attaches empty subcomponents to each

"""
def empty_subcomponent():
    base_object_attributes = ['name', 'classname', 'opts', 'object_']
    subcomponent_objects = ['network', 'optimizer', 'scheduler']
    base_object_dict = {attr:None for attr in base_object_attributes}
    subcomponent_object_dict = {attr:copy.deepcopy(base_object_dict) for attr in subcomponent_objects}
    return subcomponent_object_dict
   

def model_outline(component_names):
    component_dict = {name:empty_subcomponent() for name in component_names}
    model = utils.EasyDict(**{'components':component_dict, 'functions':{}, 'info':{}, 'state':{}})




"""
Functions for saving model components


model.components is an EasyDict that contains functional components like "generator", "discriminator", "regressor", "encoder"

Each component contains a standard set of "subcomponents": "network", "optimizer", "scheduler"
    network: The PyTorch neural network
    optimizer: The optimizer that points to the parameters of the network
    scheduler: A learning rate scheduler that points to the optimizer

Each subcomponent contains a standard set of fields that specify its full state: "name", "classname", "opts", "object_"
    name: The name of the component
    classname: A string name that represents the PyTorch class of the component
    opts: The options dict to pass to the PyTorch class upon construction
    object_: Points to the actual Pytorch object that does the work

map_object_to_state_dict:
    Takes a subcomponent dict and returns a dict with the object_ field replaced by the PyTorch state_dict() for that object

map_state_dict_to_object:
    Takes a subcomponent dict and returns a dict with the state_dict field replaced by the given object

save_standard_component:
    Takes a model component and saves it to a Pytorch file

save_model:
    For use in model.functions.save. Takes a model and saves all its components.

load_standard_component:
    Takes a component save file and loads the full component dict with constructed objects
"""

def map_object_to_state_dict(subcomponent):
    save_subcomponent_dict = utils.EasyDict
    for field in subcomponent:
        if field == 'object_':
            save_subcomponent_dict['state_dict'] = subcomponent[field].state_dict()
        else:
            save_subcomponent_dict[field] = component[field]
            
    return save_subcomponent_dict

def save_standard_component(component, save_directory, append):
    save_dict = utils.EasyDict()
    for key in component:
        subcomponent_save_dict = map_object_to_state_dict(component[key])
        save_dict[key] = subcomponent_save_dict
    torch.save(save_dict, os.path.join(save_directory, component.network.name+append+'.pth'))

def save_model(model, epoch, iteration):
    if model.info.opts.append_iteration:
        append = '_'+str(epoch)+'_'+str(iteration)
    else:
        append = ''
    
    save_directory = model.info.opts.model_folder
    if not os.path.is_dir(save_directory): #handle this elsewhere?
        os.makedirs(save_directory)
    
    for key in model.components:
        save_standard_component(model.components[key], save_directory, append)


def map_state_dict_to_object(cdict, object_):
    return_dict = utils.EasyDict()
    for key in cdict:
        if key == 'state_dict':
            return_dict['object_'] = object_
        else:
            return_dict[key] = cdict[key]
    return return_dict


def load_standard_component(filename, network_only=False):
    component_dict = torch.load(filename)
    loaded_dict = {}
    
    network_object = models.load_torch_class(component_dict.network.classname, **component_dict.network.opts)
    network_object.load_state_dict(component_dict.network.state_dict)
    loaded_dict['network'] = map_state_dict_to_object(component_dict.network, network_object)

    if not network_only:
        optim_object = models.load_torch_class(component_dict.optimizer.classname, network_object.parameters(), **component_dict.optimizer.opts)
        optim_object.load_state_dict(component_dict.optimizer.state_dict)
        loaded_dict['optimizer'] = map_state_dict_to_object(component_dict.optimizer, optim_object)

        scheduler_object = models.load_torch_class(component_dict.scheduler.classname, optim_object, **component_dict.scheduler.opts)
        scheduler_object.load_state_dict(component_dict.scheduler.state_dict)
        loaded_dict['scheduler'] = map_state_dict_to_object(component_dict.scheduler, scheduler_object)
    else:
        loaded_dict['optimizer'] = empty_subcomponent()
        loaded_dict['scheduler'] = empty_subcomponent()

    return utils.EasyDict(**loaded_dict)



"""
Functions for building models directly

    build_gan_model: builds a full standard generative adversarial network model

"""



def build_gan_model(opts):
    component_names = ['generator', 'discriminator']
    model = model_outline(component_names)

    if opts.load_generator_from is not None:
        # Load generator and associated paramters from file if file is specified
        model.components.generator = load_standard_component(opts.load_generator_from, network_only=(not opt.training))

    else:
        # Construct and initialize generator
        generator = models.load_torch_class(opts.generator_opts.classname, **opts.generator_opts.opts)
        generator.apply(model_functions.weights_init)
    
        # Construct generator optimizer and scheduler if training is indicated
        if opts.training:
            generator_optim = models.load_torch_class(opt.generator_optim_opts.classname, generator.params(), **opts.generator_optim_opts.opts)
            generator_scheduler = models.load_torch_class(opt.generator_scheduler_class, generator_optim, **opts.generator_scheduler_opts)
        else:
            generator_optim = None
            generator_scheduler = None
    
        model.components.generator.network.update({'name':'generator_network', 
                                                 'classname':opts.generator_opts.classname, 
                                                 'opts':opts.generator_opts.opts,
                                                 'object_':generator})
        model.components.generator.optimizer.update({ 
                                                'name': 'generator_optim',
                                                'classname': opts.generator_optim_opts.classname,
                                                'opts':opts.generator_optim_opts.opts, 
                                                'object_':generator_optim})
        model.components.generator.scheduler.update({
                                                'name': 'generator_scheduler',
                                                'classname': opts.generator_scheduler_opts.classname,
                                                'object_':generator_scheduler, 
                                                'opts':opts.generator_scheduler_opts.opts})
    


    if opts.load_discriminator_from is not None:
        # Load discriminator and associated paramters from file if file is specified
        model.components.discriminator = load_standard_component(opts.load_discriminator_from, network_only=(not opt.training))

    else:
        # Construct and initialize discriminator
        discriminator = models.load_torch_class(opts.discriminator_opts.classname, **opts.discriminator_opts.opts)
        discriminator.apply(model_functions.weights_init)
    
        # Construct discriminator optimizer and scheduler if training is indicated
        if opts.training:
            discriminator_optim = models.load_torch_class(opt.discriminator_optim_opts.classname, discriminator.params(), **opts.discriminator_optim_opts.opts)
            discriminator_scheduler = models.load_torch_class(opt.discriminator_scheduler_opts.classname, discriminator_optim, **opts.discriminator_scheduler_opts.opts)
        else:
            discriminator_optim = None
            discriminator_scheduler = None
    
        model.components.discriminator.network.update({'name':'discriminator', 
                                                     'classname':opts.discriminator_opts.classname,
                                                     'opts':opts.discriminator_opts.opts, 
                                                     'object_':discriminator})
        model.components.discriminator.optimizer.update({
                                            'name': 'discriminator_optim',
                                            'classname': opts.discriminator_optim_opts.classname,
                                            'opts':opts.discriminator_optim_opts.opts,
                                            'object_':discriminator_optim,
        })
        model.components.discriminator.scheduler.update({
                                                   'name': 'discriminator_scheduler',
                                                   'classname': opts.discriminator_scheduler_opts.classname, 
                                                   'opts':opts.discriminator_scheduler_opts.opts, 
                                                   'object_':discriminator_scheduler})



    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Specify extra model state information
    model.state.reference_codes = torch.randn(64, opts.generator_opts.opts.latent_dimension, device=model.generator.network.object_.device)

    # Set model functions
    sample_func = model_functions.GAN_Sample(model)
    model.functions.set_mode = model_functions.set_mode
    model.functions.sample = sample_func
    model.functions.snapshot = model_functions.snapshot
    model.functions.run = None
    model.functions.save = save_model
    model.functions.stop = None
    model.functions.update = model_functions.GAN_Update(model, sample_func = sample_func)

    return model





def build_encoder_model(opts):
    pass


def build_regressor_model(opts):
    pass


def build_classifier_model(opts):
    pass








