"""
Config file


In this file:
- Set all default arguments and stuff
- Set default folders / generate text file with default folders (give error in main function if folders not configured)
- Give main.py command line args something to update


"""

import os
import sys
import datetime
import ast
import argparse
import json


import torch
from utils import EasyDict as edict

#from datasets import collate_functions



local = edict( 
                use_cuda = torch.cuda.is_available(),
                device_count = torch.cuda.device_count(),
                parallelize_across_all = True,
             )


'''
folders = edict( model_base_folder = '/scratch0/model_runs/arlis_gan_project/models',
                 log_base_folder = '/scratch0/model_runs/arlis_gan_project/logs',
                 tensorboard_base_folder = '/scratch0/model_runs/arlis_gan_project/tensorboard' )
'''

if os.path.isfile('folder_config.json'):
    with open('folder_config.json','r') as folder_config_file:
        folders = json.load(folder_config_file)
        folders = edict(**folders)
else:
    folders = edict( model_base_folder = './models',
                     log_base_folder = './logs',
                     tensorboard_base_folder = './tensorboard',
                     dataset_folder = './datasets',
                     generated_dataset_folder = './datasets/generated_datasets',
)


global_run_details = edict( **local, 
                             max_devices = local.device_count if local.parallelize_across_all else min(1,local.device_count),
                             device = 'cuda:0' if local.use_cuda else 'cpu',
#                             training=True,
                          )
                             

#global_name_details = edict()
global_name_details = edict(
                             basename = 'model_folder', #Gets overridden by --basename in main.py
                           )


training_opts = edict (
                        batch_size = 128,
                        n_epochs = 100,
                        n_iters = 1e20, #Set very high because epochs matter more
                        checkpoint_every = 10,
                        print_every = 100,
                        print_batch_eta=True,
                        always_average_printouts = False,
                        use_tensorboard_per_iteration = False,
                        use_logfile = False, #?
                        shuffle=True,
                        drop_last=True,
                        tracking_opts = edict(
                            load_state_dict=None,
                            use_tensorboard = False,
                            use_logfile = True, #?
                        ), 
                        pin_memory = local.use_cuda,
                        collate_fn = 'tensor_stack',
                      )

running_opts = edict (
                        batch_size = 128,
                        print_every = 1000,
                        always_average_printouts = True,
                        use_tensorboard_per_iteration = False,
                        use_logfile = False,
                        shuffle=False,
                        drop_last=False,
                        tracking_opts = edict(
                            load_state_dict=None,
                            use_tensorboard = False,
                            use_logfile = True,
                        ), 
                        pin_memory = local.use_cuda,
                        collate_fn = 'tensor_stack',
                      )



dataset_opts = edict(
#    dataset_folder = '/scratch0/datasets',
#    dataset_folder = folders.dataset_folder,
    dataset = 'mnist',    
)


global_model_opts = edict(
    append_iteration = False,
    stylegan_generator = False,
    class_conditioned_generator = False,
    num_conditioned_classes = -1,
)

global_network_opts = edict(
    classname='',
    opts=edict(),
)
global_optimizer_opts = edict(
    classname = 'adam',
    opts = edict(), # Will be filled in later
)
global_scheduler_opts = edict(
    classname='steplr',
    opts = edict(),
)



"""
Model-specific options
"""














"""
Class-specific options
(Use if some options only apply to the given model classes)
"""

class_default_opts_dict = {
    'adam':edict(lr=0.0002, betas=(0.5,0.999), weight_decay=0), # 0.5 as b1 is preferred
    'steplr':edict(step_size=1, gamma=0.1),
    'netg32':edict(latent_dimension = 100, output_dimension=(3,32,32), hidden_dimension_base=64),
    'netd32':edict(input_dimension=(3,32,32), hidden_dimension_base=64),
    'nete32':edict(input_dimension=(3,32,32), hidden_dimension_base=64, output_dimension=100),
    'netr32':edict(input_dimension=(3,32,32), hidden_dimension_base=64, output_dimension=2),
    'w_regressor':edict(input_dimension=512, hidden_dimension=1024, output_dimension=1, num_inner_layers=4), #w_regressor network
    'netc32':edict(input_dimension=(3,32,32), hidden_dimension_base=64, feature_dimension=128, output_dimension=1),
    'testg':edict(latent_dimension=8),
    'stylegan2-ada-cifar10':edict(name='pretrained_stylegan', opts={'latent_dimension':512}, pickled=True, class_condition=torch.arange(10)),
#    'nete32_stylegan_cifar10':edict(input_dimension=(3,32,32), hidden_dimension_base=256, output_dimension=512),

}

def class_opts(classname):
    return class_default_opts_dict.get(classname, edict())


"""
Task-specific options
(Use for tasks like loading, sampling, and evaluating models)
"""


def train_gan_opts(all_opts, parameters, basename):
    generator_opts = edict(
        dictionary=global_network_opts,
        classname = 'netg32', #supercedes classname in global_network_opts
    )
    generator_optim_opts = edict(**global_optimizer_opts) 
    generator_scheduler_opts = edict(**global_scheduler_opts)
    
    discriminator_opts = edict(
        dictionary=global_network_opts,
        classname = 'netd32',
    )
    discriminator_optim_opts = edict(**global_optimizer_opts)
    discriminator_scheduler_opts = edict(**global_scheduler_opts)

    train_gan_task_opts = edict(
        load_generator_from = None,
        load_discriminator_from = None,
#        load_tracker_from = None,
        loss_function = 'wasserstein_gan',
        gp_lambda = 10.0, #for wasserstein loss
        critic_iters = 5,
        gen_iters = 1,
        generator_lr_schedule = [20,60,100],
        discriminator_lr_schedule = [20,60,100],
    )


    all_opts.training = True

    all_opts.update(train_gan_task_opts)
    all_opts.generator_opts = generator_opts
    all_opts.generator_optim_opts = generator_optim_opts
    all_opts.generator_scheduler_opts = generator_scheduler_opts
    
    all_opts.discriminator_opts = discriminator_opts
    all_opts.discriminator_optim_opts = discriminator_optim_opts
    all_opts.discriminator_scheduler_opts = discriminator_scheduler_opts            

    all_opts.training_opts = training_opts


    # Handle parameter dependencies
    all_opts.training_opts.tracking_opts.logdir = os.path.join(all_opts.log_base_folder, basename)
    all_opts.training_opts.tracking_opts.tensorboard_logdir = os.path.join(all_opts.tensorboard_base_folder, basename)
    all_opts.training_opts.tracking_opts.savedir=all_opts.model_folder
    all_opts.training_opts.print_keys = ['train/discriminator_loss', 'train/generator_loss']
    all_opts.training_opts.register_meters = ['train/discriminator_loss', 'train/generator_loss']
    all_opts.training_opts.tracking_opts.savefile='gan_train_metrics.pth'
    all_opts.training_opts.tracking_opts.logfile='gan_train_log.out'

    return all_opts


def train_encoder_opts(all_opts, parameters, basename):
    encoder_opts = edict(
        dictionary=global_network_opts,
        classname='nete32',
    )
    encoder_optim_opts = edict(**global_optimizer_opts)
    encoder_scheduler_opts = edict(**global_scheduler_opts)
    
    train_encoder_task_opts = edict(
        load_encoder_from = None,
        load_generator_from = os.path.join(all_opts.model_folder,'generator.pth'),
        load_generator_external_opts = None, #if stylegan, need classname, name, opts, pickled=T/F # class default opts dict?
        lr_schedule = [],
        loss_function = 'l2',
        invert=False,
    )

    all_opts.training = True

    all_opts.update(train_encoder_task_opts)
    all_opts.encoder_opts = encoder_opts
    all_opts.encoder_optim_opts = encoder_optim_opts
    all_opts.encoder_scheduler_opts = encoder_scheduler_opts
    all_opts.training_opts = training_opts # tailor to encoder



    # Handle parameter dependencies
    all_opts.training_opts.tracking_opts.logdir = os.path.join(all_opts.log_base_folder, basename)
    all_opts.training_opts.tracking_opts.tensorboard_logdir = os.path.join(all_opts.tensorboard_base_folder, basename)
    all_opts.training_opts.tracking_opts.savedir=all_opts.model_folder
    all_opts.training_opts.print_keys = ['train/encoder_loss']
    all_opts.training_opts.register_meters = ['train/encoder_loss']
    all_opts.training_opts.tracking_opts.savefile='encoder_train_metrics.pth'
    all_opts.training_opts.tracking_opts.logfile='encoder_train_log.out'

    return all_opts

def invert_dataset_opts(all_opts, parameters, basename):
    invert_dataset_task_opts = edict(
        load_generator_from = os.path.join(all_opts.model_folder,'generator.pth'),
        load_encoder_from = os.path.join(all_opts.model_folder,'encoder.pth'),
        invert=True,
        encoder_guidance_lambda = 0.5,
    )
    inversion_opts = edict(
        latent_lr = 0.001,
        latent_betas = (0.9,0.999),
        inversion_iters = 100,
    ) 

    all_opts.update(invert_dataset_task_opts)
    all_opts.inversion_opts = inversion_opts
    all_opts.training_opts = training_opts

    all_opts.training = False
    all_opts.training_opts.batch_size=128
    all_opts.training_opts.shuffle=False
    all_opts.training_opts.drop_last = False

    return all_opts

def calculate_jacobians_opts(all_opts, parameters, basename):
    calculate_jacobians_task_opts = edict(
        load_generator_from = os.path.join(all_opts.model_folder,'generator.pth'),
        invert=False,
        jacobian=True,
        stylegan_w_jacobians = False,
#        stylegan_w_classlabels = True
    )

    all_opts.update(calculate_jacobians_task_opts)
    all_opts.training_opts = training_opts

    all_opts.training = False
    all_opts.training_opts.batch_size=20
    all_opts.training_opts.shuffle=False
    all_opts.training_opts.drop_last = False

    return all_opts


def train_regressor_opts(all_opts, parameters, basename):
    regressor_opts = edict(
        dictionary=global_network_opts,
        classname='netr32',
    )

    regressor_optim_opts = edict(**global_optimizer_opts)
    regressor_scheduler_opts = edict(**global_scheduler_opts)
    
    train_regressor_task_opts = edict(
        load_regressor_from = None,
        lr_schedule = [],
        loss_function = 'l2',
        train_w_regressor = False,
        load_w_regressor = False,
        w_regressor_path = os.path.join(all_opts.model_folder, 'w_regressor.pth'),
    )
    

    all_opts.training = True

    all_opts.update(train_regressor_task_opts)
    all_opts.regressor_opts = regressor_opts
    all_opts.regressor_optim_opts = regressor_optim_opts
    all_opts.regressor_scheduler_opts = regressor_scheduler_opts
    all_opts.training_opts = training_opts # tailor to regressor

    # Handle parameter dependencies
    all_opts.training_opts.tracking_opts.logdir = os.path.join(all_opts.log_base_folder, basename)
    all_opts.training_opts.tracking_opts.tensorboard_logdir = os.path.join(all_opts.tensorboard_base_folder, basename)
    all_opts.training_opts.tracking_opts.savedir=all_opts.model_folder
    all_opts.training_opts.print_keys = ['train/regressor_loss']
    all_opts.training_opts.register_meters = ['train/regressor_loss'] #, 'test/regressor_loss']
    all_opts.training_opts.tracking_opts.savefile='regressor_train_metrics.pth'
    all_opts.training_opts.tracking_opts.logfile='regressor_train_log.out'

    return all_opts




def sample_gan_opts(all_opts, parameters, basename):
    sample_gan_task_opts = edict(
        load_generator_from = None,
    )


    all_opts.training = False

    return all_opts


def apply_regressor_opts(all_opts, parameters, basename):
    train_regressor_task_opts = edict(
        load_regressor_from = os.path.join(all_opts.model_folder,'regressor.pth'),
        lr_schedule = [],
        loss_function = 'l2',
        regressor_output_name = 'regressor_output.pth',
    )
    
    all_opts.update(train_regressor_task_opts)
    all_opts.training = False

    all_opts.update(train_regressor_task_opts)
    all_opts.training_opts = training_opts # tailor to regressor

    # Handle parameter dependencies
    all_opts.training_opts.tracking_opts.logdir = os.path.join(all_opts.log_base_folder, basename)
    all_opts.training_opts.tracking_opts.tensorboard_logdir = os.path.join(all_opts.tensorboard_base_folder, basename)
    all_opts.training_opts.tracking_opts.savedir=all_opts.model_folder
    all_opts.training_opts.print_keys = ['train/regressor_loss']
    all_opts.training_opts.register_meters = ['train/regressor_loss'] #, 'test/regressor_loss']
    all_opts.training_opts.tracking_opts.savefile='regressor_train_metrics.pth'
    all_opts.training_opts.tracking_opts.logfile='regressor_train_log.out'

    return all_opts


def train_classifier_opts(all_opts, parameters, basename):
    pass






def save_parameters(key, parameter_string):
    command_line_parameters = ast.literal_eval(parameter_string)
    
    if os.path.isfile('saved_parameters.json'):
        with open('saved_parameters.json','r') as saved_parameter_file_read:
            all_saved_parameters = json.load(saved_parameter_file_read)
    else:
        all_saved_parameters = {}

    all_saved_parameters[key] = command_line_parameters
    with open('saved_parameters.json','w') as saved_parameter_file_write:
        json.dump(all_saved_parameters, saved_parameter_file_write)


def view_parameters(print_each):
    if not os.path.isfile('saved_parameters.json'):
        print("No saved parameters to show")
    else:
        with open('saved_parameters.json','r') as saved_parameter_file_read:
            all_saved_parameters = json.load(saved_parameter_file_read)
        for key in all_saved_parameters:
            if print_each:
                print("{0} : {1}".format(key, str(all_saved_parameters[key])))
            else:
                print(key)

"""
Collect all run options together
"""

task_opt_functions = {
    'train_gan':train_gan_opts,
    'train_encoder':train_encoder_opts,
    'train_regressor':train_regressor_opts,
    'invert_dataset':invert_dataset_opts,
    'calculate_jacobians':calculate_jacobians_opts,
    'sample_gan':sample_gan_opts,
    'apply_regressor':apply_regressor_opts,
}

def collect_options(mode, basename, parameters, parameter_key=None):
    command_line_parameters = edict(**ast.literal_eval(parameters))

    if parameter_key is not None:
        with open('saved_parameters.json','r') as saved_parameter_file:
            all_saved_parameters = json.load(saved_parameter_file)
            parameters = edict(**all_saved_parameters[parameter_key])
            parameters.recursive_update(command_line_parameters)
    else:
        parameters = command_line_parameters 

    all_opts = edict() 
    all_opts.update(folders)
    all_opts.update(global_name_details)
    all_opts.update(global_run_details)
    all_opts.update(global_model_opts)
    all_opts.dataset_opts = dataset_opts
    all_opts.dataset_opts.dataset_folder = folders.dataset_folder

    all_opts.load_tracker_from = None


    all_opts.basename = basename
    all_opts.model_folder = os.path.join(all_opts.model_base_folder, basename)

    all_opts = task_opt_functions[mode](all_opts, parameters, basename)

    # TODO: fix order of operations here (does not affect current things, but still)

    # Update defaults with current parameters
    parameter_walk = parameters.walk(return_dict=True)
    for key in parameter_walk:
        all_opts.set_path(key, parameter_walk[key])
    
    # Add remaining class specific options if any
    for key in all_opts:
        item = all_opts[key]
        if isinstance(item, edict) and 'classname' in item:
            if 'opts' in item:
                item.opts.update(class_opts(item.classname), disjoint_only=True)
            else:
                item.update(class_opts(item.classname), disjoint_only=True)


    return all_opts



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base_folder', type=str, default='./models', help='The root folder for saving models')
    parser.add_argument('--log_base_folder', type=str, default='./logs', help='The root folder for saving logs')
    parser.add_argument('--tensorboard_base_folder', type=str, default='./tensorboard', help='The root folder for saving tensorboard output')
    parser.add_argument('--dataset_folder', type=str, default='./datasets', help='Where to look for datasets')
    parser.add_argument('--generated_dataset_folder', type=str, default='./datasets/generated_datasets', help='Where to look for generated datasets')

    opt = parser.parse_args()

    with open('folder_config.json', 'w') as config_file:
        json.dump(vars(opt), config_file)







