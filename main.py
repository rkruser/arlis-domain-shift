"""
Run main pipeline for program


"""


import os
import sys
import argparse
import ast
import datetime
import pickle


import utils
import config
import datasets
import build_models
import train

import torch

from datasets import collate_functions


modes = ['train_gan', 'sample_gan', 'train_encoder', 'invert_dataset', 'calculate_jacobians', 'train_regressor', 'apply_regressor', 'save_parameters', 'view_saved_parameters']
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str, default='{}', help='Global configuration parameters. Argument should be formatted as a python-correct string in quotations, specifying (nested) dictionaries of parameters as desired. Use --show_options to view all possible options for the current specified configuration.')
parser.add_argument('--basename', type=str, default='model_'+str(datetime.datetime.now()), help='Base name for the current run')
parser.add_argument('--show_options', action='store_true', help='Show the available options for the current mode and configuration of parameters, and then exit')
parser.add_argument('--mode', choices=modes, required=True, help='Possible operations')

parser.add_argument('--print_each', action='store_true', help="Only used with view_saved_parameters. If provided, print each saved set of parameters next to its key.")
parser.add_argument('--parameter_key', type=str, default=None, help='If used in save_parameters mode, saves the provided value of --parameters to saved_parameters.json under the given key. If used in any other mode, loads the parameters from the given key (and then updates them with any additional --parameters arguments provided)')




"""
python --basename cifar_stylegan --mode train_encoder --parameters "{'stylegan_generator':True, 'load_generator_from':'', 'load_generator_external_opts':{'classname':'stylegan2-ada-cifar10', 'name':'cifar_pretrained_stylegan', 'opts':None, 'pickled'=True}}"
python --basename cifar_stylegan --mode invert_dataset --parameters "{}"
python --basename cifar_stylegan --mode calculate_jacobians --parameters "{}"
python --basename cifar_stylegan --mode train_regressor --parameters "{}" #w_regressor training
python --basename cifar_stylegan --mode train_regressor --parameters "{}" #full regressor training

"""



"""
Handle GAN training
"""
def train_gan(opts):
    model = build_models.build_gan_model(opts)
    dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opts.training_opts.batch_size, shuffle=opts.training_opts.shuffle, drop_last=opts.training_opts.drop_last, collate_fn = collate_functions[opts.training_opts.collate_fn], pin_memory=opts.training_opts.pin_memory)

    if opts.load_tracker_from is not None:
        tracker_state_dict = pickle.load(open(opts.load_tracker_from, 'rb'))
        opts.training_opts.tracking_opts.load_state_dict = tracker_state_dict # a StateDictWrapper object

    train.train(model, dataloader, opts.training_opts)


"""
Handle GAN sampling
"""
def sample_gan(opt):
    pass



"""
Handle encoder training
"""
def train_encoder(opts):
    model = build_models.build_encoder_model(opts)
#    dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder)

    latent_dimension = model.components.generator.network.opts.latent_dimension
    generator_net = model.components.generator.network.object_

#    if opts.stylegan_generator:
    dataloader = datasets.GeneratorOutputLoader(generator_net, 1000, opts.training_opts.batch_size, latent_dimension, opts.device,
                                                    class_conditioned=opts.class_conditioned_generator, 
                                                    num_classes=opts.num_conditioned_classes, stylegan=opts.stylegan_generator)
#    else:
#        dataloader = datasets.GeneratorOutputLoader(generator_net, 1000, opts.training_opts.batch_size, latent_dimension, opts.device)

    if opts.load_tracker_from is not None:
        tracker_state_dict = pickle.load(open(opts.load_tracker_from, 'rb'))
        opts.training_opts.tracking_opts.load_state_dict = tracker_state_dict # a StateDictWrapper object

    train.train(model, dataloader, opts.training_opts)


"""
Handle GAN inverting
"""
def invert_dataset(opts):
    model = build_models.build_encoder_model(opts)
    dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opts.training_opts.batch_size, shuffle=opts.training_opts.shuffle, drop_last=opts.training_opts.drop_last, collate_fn = collate_functions[opts.training_opts.collate_fn], pin_memory=opts.training_opts.pin_memory)

    latent_dataset = train.invert_generator(model, dataloader, opts)  # includes codes and reconstruction losses
    latent_dataset = datasets.normalize_dataset(latent_dataset) #normalize scalar components

    cur_generated_dataset_folder = os.path.join(opts.generated_dataset_folder, opts.basename)
    utils.make_directories([cur_generated_dataset_folder])
    torch.save(latent_dataset, os.path.join(cur_generated_dataset_folder, 'latent_dataset.pth'))

"""
Handle Jacobian calculations
"""
def calculate_jacobians(opts):
    cur_generated_dataset_folder = os.path.join(opts.generated_dataset_folder, opts.basename)
    if opts.stylegan_w_jacobians:
        dataset = datasets.RandomDataset(length=50000, point_size=512, nlabels=10)
        output_name = 'w_jacobian_dataset.pth'
    else:
        dataset = datasets.get_generated_dataset('latent_dataset.pth', cur_generated_dataset_folder)
        output_name = 'jacobian_dataset.pth'

    model = build_models.build_generator_model(opts) #Perhaps generator only model
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opts.training_opts.batch_size, shuffle=opts.training_opts.shuffle, drop_last=opts.training_opts.drop_last, collate_fn = collate_functions[opts.training_opts.collate_fn], pin_memory=opts.training_opts.pin_memory)
    jacobian_dataset = train.calculate_jacobians(model, dataloader, opt)
    jacobian_dataset = datasets.normalize_dataset(jacobian_dataset) #normalize scalar components
    torch.save(jacobian_dataset, os.path.join(cur_generated_dataset_folder, output_name)) #includes only log jacobian


"""
Handle regressor training
"""
def train_regressor(opts):
    cur_generated_dataset_folder = os.path.join(opts.generated_dataset_folder, opts.basename)

    model = build_models.build_regressor_model(opts)

    if opts.train_w_regressor:
        dataset = datasets.get_generated_dataset('w_jacobian_dataset.pth', cur_generated_dataset_folder)
    elif opts.stylegan_generator:
        regular_dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder)
        latent_dataset = datasets.get_generated_dataset('latent_dataset.pth', cur_generated_dataset_folder)
        jacobian_dataset = datasets.get_generated_dataset('w_jacobian_dataset.pth', cur_generated_dataset_folder)
        dataset = datasets.ConcatDatasets(regular_dataset, latent_dataset, jacobian_dataset) 
    else:
        regular_dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder)
        # also load latent_dataset.pth and jacobian_dataset.pth and stitch them together
        latent_dataset = datasets.get_generated_dataset('latent_dataset.pth', cur_generated_dataset_folder)
        jacobian_dataset = datasets.get_generated_dataset('jacobian_dataset.pth', cur_generated_dataset_folder)
        dataset = datasets.ConcatDatasets(regular_dataset, latent_dataset, jacobian_dataset) 


    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opts.training_opts.batch_size, shuffle=opts.training_opts.shuffle, drop_last=opts.training_opts.drop_last, collate_fn = collate_functions[opts.training_opts.collate_fn], pin_memory=opts.training_opts.pin_memory)

    if opts.load_tracker_from is not None:
        tracker_state_dict = pickle.load(open(opts.load_tracker_from, 'rb'))
        opts.training_opts.tracking_opts.load_state_dict = tracker_state_dict # a StateDictWrapper object

    train.train(model, dataloader, opts.training_opts)


# don't think these are necessary
def train_stylegan_w_regressor(opts):
    pass

def train_stylegan_full_regressor(opts):
    pass


"""
Handle regressor application
"""

def apply_regressor(opts):
    model = build_models.build_regressor_model(opts)
    dataset = datasets.get_dataset(dataset=opts.dataset_opts.dataset, dataset_folder=opts.dataset_opts.dataset_folder, train=opts.dataset_opts.train) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opts.training_opts.batch_size, shuffle=False, drop_last=False, collate_fn = collate_functions[opts.training_opts.collate_fn], pin_memory=opts.training_opts.pin_memory)
    if opts.load_tracker_from is not None:
        tracker_state_dict = pickle.load(open(opts.load_tracker_from, 'rb'))
        opts.training_opts.tracking_opts.load_state_dict = tracker_state_dict # a StateDictWrapper object
    regressor_outputs = train.simple_regressor_run(model, dataloader, opts)

    cur_generated_dataset_folder = os.path.join(opts.generated_dataset_folder, opts.basename)
    torch.save(regressor_outputs, os.path.join(cur_generated_dataset_folder, opts.regressor_output_name))





function_map = {
        'train_gan':train_gan,
        'sample_gan':sample_gan,
        'train_encoder':train_encoder,
        'invert_dataset':invert_dataset,
        'calculate_jacobians':calculate_jacobians,
        'train_regressor':train_regressor,
        'apply_regressor':apply_regressor,
        }            


def main(opt):
    if opt.mode == 'save_parameters':
        config.save_parameters(opt.parameter_key, opt.parameters)
        sys.exit()
    elif opt.mode == 'view_saved_parameters':
        config.view_parameters(opt.print_each)
        sys.exit()

    all_options = config.collect_options(opt.mode, opt.basename, opt.parameters, opt.parameter_key)
    
    if opt.show_options:
        print(all_options)
        sys.exit()

    run_func = function_map.get(opt.mode)
    if run_func is None:
        print("{0} mode not implemented".format(opt.mode))
        sys.exit()

    utils.make_directories([all_options.model_base_folder, all_options.log_base_folder, all_options.tensorboard_base_folder], all_options.basename)
    utils.make_directories([all_options.dataset_folder, all_options.generated_dataset_folder])

    run_func(all_options)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)





"""
Next:
- Write out all commands for stylegan inversion (partially done above; need to edit config so stylegan external opts can be easily loaded)
- define w regressor network (done in models.py)
- 

"""



