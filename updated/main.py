"""
Run main pipeline for program


"""


import os
import sys
import argparse
import ast
import datetime


import utils
import config


modes = ['train_gan', 'sample_gan', 'train_encoder', 'invert_dataset', 'calculate_jacobians', 'train_regressor', 'apply_regressor']
parser = argparse.ArgumentParser()
parser.add_argument('--parameters', type=str, default='{}', help='Global configuration parameters. Argument should be formatted as a python-correct string in quotations, specifying (nested) dictionaries of parameters as desired. Use --show_options to view all possible options for the current specified configuration.')
parser.add_argument('--basename', type=str, default='model_'+str(datetime.datetime.now()), help='Base name for the current run')
parser.add_argument('--show_options', action='store_true', help='Show the available options for the current mode and configuration of parameters, and then exit')
parser.add_argument('--mode', choices=modes, required=True, help='Possible operations')

"""
Next steps:
- Collect all GAN training options
- Comb through names for consistency
- Write train_gan function
- Test train_gan step by step

"""


"""
Handle GAN training
"""
def train_gan(opt):
    print("In train gan")
    print(opt)


"""
Handle GAN sampling
"""
def sample_gan(opt):
    pass



"""
Handle encoder training
"""
def train_encoder(opt):
    pass



"""
Handle GAN inverting
"""
def invert_dataset(opt):
    pass



"""
Handle Jacobian calculations
"""
def calculate_jacobians(opt):
    pass


"""
Handle regressor training
"""
def train_regressor(opt):
    pass


"""
Handle regressor application
"""
def apply_regressor(opt):
    pass




function_map = {
        'train_gan':train_gan,
        'sample_gan':sample_gan,
        'train_encoder':train_encoder,
        'invert_dataset':invert_dataset,
        'calculate_jacobians':calculate_jacobians,
        'train_regressor':train_regressor,
        'apply_regressor':apply_regressor
        }            


def main(opt):
    all_options = config.collect_options(opt.mode, opt.basename, opt.parameters)
    
    if opt.show_options:
        print(all_options)
        sys.exit()

    run_func = function_map.get(opt.mode)
    if run_func is None:
        print("{0} mode not implemented".format(opt.mode))
        sys.exit()
    run_func(all_options)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)



