"""
Config file


In this file:
- Set all default arguments and stuff
- Set default folders / generate text file with default folders (give error in main function if folders not configured)
- Give main.py command line args something to update


TODO:
- add ability to print default options to command line on request
- add ability to parse command line options and update dicts in the right order
- add full options for each task and organize them
- make sure everything makes sense


"""

import os
import sys
import datetime
import ast


import torch
from utils import EasyDict as edict



local = edict( 
                use_cuda = torch.cuda.is_available(),
                device_count = torch.cuda.device_count(),
                parallelize_across_all = False, #not yet implemented
             )


folders = edict( model_base_folder = './models',
                 log_base_folder = './logs',
                 tensorboard_base_folder = './tensorboard' )


global_run_details = edict( **local, 
                             max_devices = device_count if local.parallelize_across_all else 1,
                             device = 'cuda:0' if local.use_cuda else 'cpu',
                          )
                             

#global_name_details = edict()
global_name_details = edict(
                             basename = 'model_folder', #Gets overridden by --basename in main.py
                           )


training_opts = edict (
                        batch_size = 128,
                        n_epochs = 100,
                        checkpoint_every = 10,
                        print_every = 100,
                        always_average_printouts = False,
                        use_tensorboard_per_iteration = False,
                        use_logfile = False,
                        shuffle=True,
                        drop_last=True,
                        tracking_opts = edict(
                            load_state_dict=None,
                            use_tensorboard = False,
                            use_logfile = True,
                        ), 
                      )

dataset_opts = edict(
    dataset = 'mnist',    
)


global_model_opts = edict(
    append_iteration = False,
    

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


# GAN model
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




# Encoder
encoder_opts = edict()
encoder_optim_opts = edict(**global_optimizer_opts)
encoder_scheduler_opts = edict(**global_scheduler_opts)




# Regressor
regressor_opts = edict()
regressor_optim_opts = edict(**global_optimizer_opts)
regressor_scheduler_opts = edict(**global_scheduler_opts)




"""
Class-specific options
(Use if some options only apply to the given model classes)
"""
def class_opts(classname):
    if classname == 'adam':
        return edict(lr=0.0001, betas=(0,9,0.999), weight_decay=0)
    elif classname == 'steplr':
        return edict(step_size=1, gamma=0.1)
    elif classname == 'netg32':
        return edict(latent_dimension = 100, output_dimension=(3,32,32), hidden_dimension_base=64)
    elif classname == 'netd32':
        return edict(input_dimension=(3,32,32), hidden_dimension_base=64)

    return edict()


"""
Task-specific options
(Use for tasks like loading, sampling, and evaluating models)
"""

train_gan_task_opts = edict(
    load_generator_from = None,
    load_discriminator_from = None,
    loss_function = 'wasserstein_gan',
    gp_lambda = 10.0, #for wasserstein loss
    critic_iters = 5,
    gen_iters = 1,
    generator_lr_schedule = [20,60,100],
    discriminator_lr_schedule = [20,60,100],
)




"""
Collect all run options together
"""
def collect_options(mode, basename, parameters):
    parameters = edict(ast.literal_eval(parameters))
    all_options = edict() 
    all_options.update(folders)
    all_options.update(global_name_details)
    all_options.update(global_run_details)
    all_options.update(global_model_opts)
    all_options.update(train_gan_task_opts)
    if mode == 'train_gan':
        all_options.generator_opts = generator_opts
        all_options.generator_optim_opts = generator_optim_opts
        all_options.generator_scheduler_opts = generator_scheduler_opts
        
        all_options.discriminator_opts = discriminator_opts
        all_options.discriminator_optim_opts = discriminator_optim_opts
        all_options.discriminator_scheduler_opts = discriminator_scheduler_opts            

        all_options.training_opts = training_opts
        all_options.dataset_opts = dataset_opts

        # Update defaults with current parameters
        parameter_walk = parameters.walk(return_dict=True)
        for key in parameter_walk:
            all_options.set_path(key, parameter_walk[key])

        # Add remaining class specific options if any
        for key in all_options:
            item = all_options[key]
            if isinstance(item, edict) and 'classname' in item:
                item.opts.update(class_opts(item.classname), disjoint_only=True)


    # Handle parameter dependencies
    all_options.basename = basename
    all_options.model_folder = os.path.join(all_options.model_base_folder, basename)
    all_options.training_opts.tracking_opts.logdir = os.path.join(all_options.log_base_folder, basename)
    all_options.training_opts.tracking_opts.tensorboard_logdir = os.path.join(all_options.tensorboard_base_folder, basename)
    all_options.training_opts.tracking_opts.savedir=all_options.model_folder
    all_options.training_opts.tracking_opts.savefile='gan_train_metrics.pth'
    all_options.training_opts.tracking_opts.logfile='gan_train_log.out'


    # *** Do something about making the folders




    return all_options




















