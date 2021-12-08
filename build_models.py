"""
Build full runnable models given config/command line args
"""

import os

import models
import utils
#from utils import StateDictWrapper #Necessary for loading reasons
import model_functions

import copy

import torch
import torchvision


def base_object_dict():
    base_object_attributes = ['name', 'classname', 'opts', 'object_']
    base_object_dict = {attr:None for attr in base_object_attributes}
    return base_object_dict


def empty_subcomponent():
    subcomponent_objects = ['network', 'optimizer', 'scheduler']
    subcomponent_object_dict = {attr:copy.deepcopy(base_object_dict()) for attr in subcomponent_objects}
    return utils.EasyDict(**subcomponent_object_dict)
   

def model_outline(component_names):
    component_dict = {name:empty_subcomponent() for name in component_names}
    model = utils.EasyDict(**{'components':component_dict, 'functions':{}, 'info':{}, 'state':{}})
    return model


def map_object_to_state_dict(subcomponent):
    save_subcomponent_dict = utils.EasyDict()
    for field in subcomponent:
        if field == 'object_' and subcomponent[field] is not None:
            object_ = subcomponent[field]
            if isinstance(object_, torch.nn.DataParallel):
                object_ = object_._modules['module']
            save_subcomponent_dict['state_dict'] = utils.StateDictWrapper(object_.state_dict()) #Problem: easydict converts dicts to easydicts, which does not work for state_dict
        else:
            save_subcomponent_dict[field] = subcomponent[field]
            
    return save_subcomponent_dict

def save_standard_component(component, save_directory, append):
    save_dict = utils.EasyDict()
    for key in component:
        subcomponent_save_dict = map_object_to_state_dict(component[key])
        save_dict[key] = subcomponent_save_dict
    torch.save(save_dict, os.path.join(save_directory, component.network.name+append+'.pth'))


class Save_Model(model_functions.Model_Function_Base):
    def __init__(self, omit=[]):
        self.omit = omit

    def __call__(self, model, epoch, iteration):
        if model.info.opts.append_iteration:
            append = '_'+str(epoch)+'_'+str(iteration)
        else:
            append = ''
        
        save_directory = model.info.opts.model_folder
        
        for key in model.components:
            if key not in self.omit:
                save_standard_component(model.components[key], save_directory, append)


def map_state_dict_to_object(cdict, object_):
    return_dict = utils.EasyDict()
    for key in cdict:
        if key == 'state_dict':
            return_dict['object_'] = object_
        else:
            return_dict[key] = cdict[key]
    return return_dict


def load_standard_component(filename, network_only=False, device='cpu', external_opts = None):
    if external_opts is None:
        component_dict = torch.load(filename, map_location=device) #map_location here
        loaded_dict = {}
        network_object = models.load_torch_class(component_dict.network.classname, **component_dict.network.opts)
        network_object.load_state_dict(component_dict.network.state_dict.state_dict)
        network_object = network_object.to(device)
        loaded_dict['network'] = map_state_dict_to_object(component_dict.network, network_object)
    
        if not network_only:
            optim_object = models.load_torch_class(component_dict.optimizer.classname, network_object.parameters(), **component_dict.optimizer.opts)
            optim_object.load_state_dict(component_dict.optimizer.state_dict.state_dict)
            loaded_dict['optimizer'] = map_state_dict_to_object(component_dict.optimizer, optim_object)
    
            scheduler_object = models.load_torch_class(component_dict.scheduler.classname, optim_object, **component_dict.scheduler.opts)
            scheduler_object.load_state_dict(component_dict.scheduler.state_dict.state_dict)
            loaded_dict['scheduler'] = map_state_dict_to_object(component_dict.scheduler, scheduler_object)
        else:
            loaded_dict['optimizer'] = base_object_dict()
            loaded_dict['scheduler'] = base_object_dict()
    else:
        # network saved from some other program
        network_object = models.load_torch_class(external_opts.classname, filename=filename, **external_opts.opts)

        if not external_opts.pickled:
            component_dict = torch.load(filename, map_location=device) #map_location here
            network_object.load_state_dict(component_dict)

        network_object = network_object.to(device)


        loaded_dict = empty_subcomponent()
        loaded_dict.network.object_ = network_object
        loaded_dict.network.classname = external_opts.classname
        loaded_dict.network.name = external_opts.name
        loaded_dict.network.opts = external_opts.opts
       



        loaded_dict['optimizer'] = base_object_dict()
        loaded_dict['scheduler'] = base_object_dict()


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
        print("Loading generator from", opts.load_generator_from)
        model.components.generator = load_standard_component(opts.load_generator_from, network_only=(not opts.training), device=opts.device)

    else:
        # Construct and initialize generator
        print("Constructing and initializing generator")
        generator = models.load_torch_class(opts.generator_opts.classname, **opts.generator_opts.opts)
        generator.apply(model_functions.weights_init)
        generator = generator.to(opts.device)
    
        # Construct generator optimizer and scheduler if training is indicated
        if opts.training:
            generator_optim = models.load_torch_class(opts.generator_optim_opts.classname, generator.parameters(), **opts.generator_optim_opts.opts)
            generator_scheduler = models.load_torch_class(opts.generator_scheduler_opts.classname, generator_optim, **opts.generator_scheduler_opts.opts)
        else:
            generator_optim = None
            generator_scheduler = None
    
        model.components.generator.network.update({'name':'generator', 
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
        print("Loading discriminator from", opts.load_discriminator_from)
        model.components.discriminator = load_standard_component(opts.load_discriminator_from, network_only=(not opts.training), device=opts.device)

    else:
        # Construct and initialize discriminator
        print("Constructing and initializing discriminator")
        discriminator = models.load_torch_class(opts.discriminator_opts.classname, **opts.discriminator_opts.opts)
        discriminator.apply(model_functions.weights_init)
        discriminator = discriminator.to(opts.device)
    
        # Construct discriminator optimizer and scheduler if training is indicated
        if opts.training:
            discriminator_optim = models.load_torch_class(opts.discriminator_optim_opts.classname, discriminator.parameters(), **opts.discriminator_optim_opts.opts)
            discriminator_scheduler = models.load_torch_class(opts.discriminator_scheduler_opts.classname, discriminator_optim, **opts.discriminator_scheduler_opts.opts)
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


    

    if opts.max_devices > 1:
        if not opts.stylegan_generator:
            model.components.generator.network.object_ = torch.nn.DataParallel(model.components.generator.network.object_)
        model.components.discriminator.network.object_ = torch.nn.DataParallel(model.components.discriminator.network.object_)


    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Specify extra model state information
    model.state.reference_codes = torch.randn(64, opts.generator_opts.opts.latent_dimension, device=opts.device)

    # Set model functions
    sample_func = model_functions.GAN_Sample(model)
    snapshot_func = model_functions.snapshot if opts.generator_opts.classname != 'testg' else None

    model.functions.set_mode = model_functions.Set_Mode()
    model.functions.sample = sample_func
    model.functions.snapshot = snapshot_func
    model.functions.run = None
    model.functions.save = Save_Model()
    model.functions.stop = model_functions.stop

    if opts.loss_function == 'wasserstein_gan':
        model.functions.update = model_functions.Wasserstein_GAN_Update(model, sample_func = sample_func)
    else:
        model.functions.update = model_functions.Regular_GAN_Update(model, sample_func = sample_func)

    return model




def build_encoder_model(opts):
    component_names = ['generator', 'encoder']
    model = model_outline(component_names)

    # Load generator and associated paramters from file if file is specified
    print("Loading generator from", opts.load_generator_from)
    model.components.generator = load_standard_component(opts.load_generator_from, network_only=True, device=opts.device, external_opts=opts.load_generator_external_opts)
    model.components.generator.network.object_.eval() #Always stays eval


    if opts.load_encoder_from is not None:
        # Load encoder and associated paramters from file if file is specified
        print("Loading encoder from", opts.load_encoder_from)
        model.components.encoder = load_standard_component(opts.load_encoder_from, network_only=(not opts.training), device=opts.device)

    else:
        # Construct and initialize encoder
        print("Constructing and initializing encoder")
        encoder = models.load_torch_class(opts.encoder_opts.classname, **opts.encoder_opts.opts)
        encoder.apply(model_functions.weights_init)
        encoder = encoder.to(opts.device)
    
        # Construct encoder optimizer and scheduler if training is indicated
        if opts.training:
            encoder_optim = models.load_torch_class(opts.encoder_optim_opts.classname, encoder.parameters(), **opts.encoder_optim_opts.opts)
            encoder_scheduler = models.load_torch_class(opts.encoder_scheduler_opts.classname, encoder_optim, **opts.encoder_scheduler_opts.opts)
        else:
            encoder_optim = None
            encoder_scheduler = None
    
        model.components.encoder.network.update({'name':'encoder', 
                                                     'classname':opts.encoder_opts.classname,
                                                     'opts':opts.encoder_opts.opts, 
                                                     'object_':encoder})
        model.components.encoder.optimizer.update({
                                            'name': 'encoder_optim',
                                            'classname': opts.encoder_optim_opts.classname,
                                            'opts':opts.encoder_optim_opts.opts,
                                            'object_':encoder_optim,
        })
        model.components.encoder.scheduler.update({
                                                   'name': 'encoder_scheduler',
                                                   'classname': opts.encoder_scheduler_opts.classname, 
                                                   'opts':opts.encoder_scheduler_opts.opts, 
                                                   'object_':encoder_scheduler})


    if opts.max_devices > 1:
        model.components.encoder.network.object_ = torch.nn.DataParallel(model.components.encoder.network.object_)
        if not opts.stylegan_generator:
            model.components.generator.network.object_ = torch.nn.DataParallel(model.components.generator.network.object_)


    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Set model functions
    if opts.stylegan_generator:
        sample_func = model_functions.StyleGAN_Sample(model)
    else:
        sample_func = model_functions.GAN_Sample(model) #can probably use this unchanged, but check

    model.functions.set_mode = model_functions.Set_Mode(omit=['generator']) 
    model.functions.sample = sample_func
    model.functions.snapshot = None
    model.functions.save = Save_Model(omit=['generator']) 
    model.functions.stop = model_functions.stop

    if ('invert' in opts) and opts.invert:
        if opts.stylegan_generator:
            model.functions.invert = model_functions.Invert_StyleGAN_Generator(model)
        else:
            model.functions.invert = model_functions.Invert_Generator(model)
    else:
        if opts.stylegan_generator:
             update_func = model_functions.Encoder_Stylegan_Update(model)
        else:
            update_func = model_functions.Encoder_Update(model)

        model.functions.run = update_func.run
        model.functions.update = update_func

    return model


def build_generator_model(opts):
    component_names = ['generator']
    model = model_outline(component_names)

    # Load generator and associated paramters from file if file is specified
    print("Loading generator from", opts.load_generator_from)
    model.components.generator = load_standard_component(opts.load_generator_from, network_only=True, device=opts.device, external_opts = opts.load_generator_external_opts)
    model.components.generator.network.object_.eval() #Always stays eval


    if opts.max_devices > 1 and not opts.stylegan_generator:
        model.components.generator.network.object_ = torch.nn.DataParallel(model.components.generator.network.object_)

    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Set model functions
    if opts.stylegan_generator:
        sample_func = model_functions.StyleGAN_Sample(model)
    else:
        sample_func = model_functions.GAN_Sample(model) #can probably use this unchanged, but check
#    update_func = model_functions.Encoder_Update(model)
    model.functions.set_mode = model_functions.Set_Mode(omit=['generator']) 
    model.functions.sample = sample_func
    model.functions.snapshot = None
#    model.functions.run = update_func.run
#    model.functions.update = update_func
    model.functions.save = Save_Model(omit=['generator']) 
    model.functions.stop = model_functions.stop

    if ('invert' in opts) and opts.invert:
        if opts.stylegan_generator:
            model.functions.invert = model_functions.Invert_StyleGAN_Generator(model)
        else:
            model.functions.invert = model_functions.Invert_Generator(model)
    elif ('jacobian' in opts) and opts.jacobian:
        if opts.stylegan_generator:
            model.functions.jacobian = model_functions.StyleGAN_Logprobs(model)
        else:
            model.functions.jacobian = model_functions.Log_Jacobian_Determinant(model)


    return model



def build_regressor_model(opts):
    component_names = ['regressor']
    if opts.load_w_regressor:
        component_names.append('w_regressor')
#    elif opts.train_w_regressor:
#        component_names = ['w_regressor']

    model = model_outline(component_names)


    if opts.load_w_regressor:
        print("Loading w-regressor from", opts.w_regressor_path)
        model.components.w_regressor = load_standard_component(opts.w_regressor_path, network_only=True, device=opts.device)


    if opts.load_regressor_from is not None:
        # Load regressor and associated paramters from file if file is specified
        print("Loading regressor from", opts.load_regressor_from)
        model.components.regressor = load_standard_component(opts.load_regressor_from, network_only=(not opts.training), device=opts.device)

    else:
        # Construct and initialize regressor
        print("Constructing and initializing regressor")
        regressor = models.load_torch_class(opts.regressor_opts.classname, **opts.regressor_opts.opts)
        regressor.apply(model_functions.weights_init)
        regressor = regressor.to(opts.device)
    
        # Construct regressor optimizer and scheduler if training is indicated
        if opts.training:
            regressor_optim = models.load_torch_class(opts.regressor_optim_opts.classname, regressor.parameters(), **opts.regressor_optim_opts.opts)
            regressor_scheduler = models.load_torch_class(opts.regressor_scheduler_opts.classname, regressor_optim, **opts.regressor_scheduler_opts.opts)
        else:
            regressor_optim = None
            regressor_scheduler = None
    
        if opts.train_w_regressor:
            regressor_name = 'w_regressor'
        else:
            regressor_name = 'regressor'

        model.components.regressor.network.update({'name':regressor_name,
                                                     'classname':opts.regressor_opts.classname,
                                                     'opts':opts.regressor_opts.opts, 
                                                     'object_':regressor})
        model.components.regressor.optimizer.update({
                                            'name': 'regressor_optim',
                                            'classname': opts.regressor_optim_opts.classname,
                                            'opts':opts.regressor_optim_opts.opts,
                                            'object_':regressor_optim,
        })
        model.components.regressor.scheduler.update({
                                                   'name': 'regressor_scheduler',
                                                   'classname': opts.regressor_scheduler_opts.classname, 
                                                   'opts':opts.regressor_scheduler_opts.opts, 
                                                   'object_':regressor_scheduler})


        



    if opts.max_devices > 1:
        model.components.regressor.network.object_ = torch.nn.DataParallel(model.components.regressor.network.object_)

    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Set model functions
    sample_func = None #model_functions.GAN_Sample(model) #can probably use this unchanged, but check


    if opts.load_w_regressor and not opts.train_w_regressor:
        update_func = model_functions.Paired_Regressor_Update(model)
    else:
        update_func = model_functions.Regressor_Update(model)


    model.functions.set_mode = model_functions.Set_Mode()
    model.functions.sample = sample_func
    model.functions.snapshot = None
    model.functions.run = update_func.run
    model.functions.save = Save_Model() 
    model.functions.stop = model_functions.stop

    model.functions.update = update_func

    return model


# Implement for final stylegan pipeline; need to load w-regressor
def build_stylegan_regressor_model(opts):
    pass



def build_classifier_model(opts):
    component_names = ['classifier']
    model = model_outline(component_names)

    if opts.load_classifier_from is not None:
        # Load classifier and associated paramters from file if file is specified
        print("Loading classifier from", opts.load_classifier_from)
        model.components.classifier = load_standard_component(opts.load_classifier_from, network_only=(not opts.training), device=opts.device)

    else:
        # Construct and initialize classifier
        print("Constructing and initializing classifier")
        classifier = models.load_torch_class(opts.classifier_opts.classname, **opts.classifier_opts.opts)
        classifier.apply(model_functions.weights_init)
        classifier = classifier.to(opts.device)
    
        # Construct classifier optimizer and scheduler if training is indicated
        if opts.training:
            classifier_optim = models.load_torch_class(opts.classifier_optim_opts.classname, classifier.parameters(), **opts.classifier_optim_opts.opts)
            classifier_scheduler = models.load_torch_class(opts.classifier_scheduler_opts.classname, classifier_optim, **opts.classifier_scheduler_opts.opts)
        else:
            classifier_optim = None
            classifier_scheduler = None
    
        model.components.classifier.network.update({'name':'classifier', 
                                                     'classname':opts.classifier_opts.classname,
                                                     'opts':opts.classifier_opts.opts, 
                                                     'object_':classifier})
        model.components.classifier.optimizer.update({
                                            'name': 'classifier_optim',
                                            'classname': opts.classifier_optim_opts.classname,
                                            'opts':opts.classifier_optim_opts.opts,
                                            'object_':classifier_optim,
        })
        model.components.classifier.scheduler.update({
                                                   'name': 'classifier_scheduler',
                                                   'classname': opts.classifier_scheduler_opts.classname, 
                                                   'opts':opts.classifier_scheduler_opts.opts, 
                                                   'object_':classifier_scheduler})


    if opts.max_devices > 1:
        model.components.classifier.network.object_ = torch.nn.DataParallel(model.components.classifier.network.object_)

    
    # Specify model info
    model.info.loss_is_averaged = True
    model.info.opts = opts

    # Set model functions
    sample_func = None
    update_func = model_functions.Classifier_Update(model)
    model.functions.set_mode = model_functions.Set_Mode()  
    model.functions.sample = sample_func
    model.functions.snapshot = None
    model.functions.run = update_func.run 
    model.functions.save = Save_Model() 
    model.functions.stop = model_functions.stop

    model.functions.update = update_func 

    return model




