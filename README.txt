# Leveraging Pretrained GAN Models to detect Domain Shift

This codebase allows you to train models through various stages and datasets that depend on one another.

When the user executes a basic command through main.py (see examples below), the execution model is the following:
1. config.py loads default parameters
2. main.py calls a function that updates the default parameters with any parameters specified via the --parameters command line option. The argument to --parameters should be a string in quotes that specifies a (nested) python dictionary with the desired options in the right places. --show_options (see below) can help you figure out what to put in this dictionary.
3. All the parameters are accumulated into opts, which is fed to the function which handles the specific training/testing mode
4. The mode handler function first builds the model by calling the specified function from build_models.py.
5. Then it obtains the dataset by querying get_dataset from datasets.py. It creates a dataloader from this dataset.
6. Then it calls the specified function from train.py, either train() or run() or invert_dataset() or jacobians()
7. The function runs the model on the dataset, printing and saving output as needed
8. At the end, the mode handler function saves anything that needs saving, and exits


The project uses a folder structure as follows:
./models
./logs
./tensorboard
./datasets
./datasets/generated_datasets

These folders will be automatically created if they do not exist previously. On each run, main.py creates a subfolder of models, logs, tensorboard, and generated_datasets with the given --basename, and saves any generated files into those basename subfolders.

The locations of these folders can be permanently reconfigured on your machine by typing
python config.py --model_base_folder [new model folder] --log_base_folder [new log folder] --tensorboard_base_folder [new tensorboard folder] --dataset_folder [new dataset folder] --generated_dataset_folder [new_generated_dataset_folder]

Sets of parameters can be saved via
python main.py --mode save_parameters --parameters "{...dict of params to save...}" --parameter_key [name_of_parameter_set]

They can then be invoked via
python main.py --basename [current basename] --mode [mode to run] --parameter_key [name of parameter set to load]



The project can be extended as follows:
- You can add a new type of neural network model by adding an entry to load_torch_class in models.py that constructs the class (the network itself can be in any file as long as you import it into models)
- You can add a new dataset by adding an entry to get_dataset in datasets.py
- You can create a new type of model by adding a build function to build_models.py and associated helper functions to model_functions.py
- You can create a new type of execution mode by adding a mode handler to main.py and a task_opt_function for that mode to config.py

The primary data structure for tracking options and model structure is the EasyDict, defined in utils.py, which is simply a dictionary which also allows you to access the elements via the dot operator. i.e. d = EasyDict(); d['entry1'] = 4; d.entry2 = 6; print(d.entry1, d['entry2'])



Command examples:

python main.py --help  :  Show command line options
python main.py --mode train\_gan --show_options  : Print a nested dictionary of all available options that could be fed into --parameters



Full training flow:

Train a GAN of class (netg32, netd32) on 'mnist':
Command: python main.py --basename test1 --mode train_gan --parameters "{'generator_opts':{'classname':'netg32'}, 'discriminator_opts':{'classname':'netd32'}, 'training_opts':{'batch_size':128, 'n_epochs':50,'print_every':200}, 'dataset_opts':{'dataset':'mnist'}}"
Produces: models/test1/{generator.pth, discriminator.pth, gan_train_metrics.pth}


Train an encoder of class nete32 based on the GAN:
Command: python main.py --basename test1 --mode train_encoder --parameters "{'encoder_opts':{'classname':'nete32'}, 'training_opts':{'batch_size':128, 'n_epochs':50,'print_every':200}}"
Uses: generator.pth
Produces: models/test1/{encoder.pth, encoder_train_metrics.pth}


Go back from the data to the latent code space with the help of the encoder and save the results:
Command: python main.py --basename test1 --mode invert_dataset --parameters "{'dataset_opts':{'dataset':'mnist'}}"
Uses: generator.pth, encoder.pth
Produces: datasets/generated_datasets/test1/latent_dataset.pth


Calculate the log probabilities from the Jacobian:
Command: python main.py --basename test1 --mode calculate_jacobians
Uses: generator.pth, latent_dataset.pth
Produces: datasets/generated_datasets/test1/jacobian_dataset.pth

Train a regressor to predict reconstruction losses and log probabilities for the given dataset
Command: python main.py --basename test1 --mode train_regressor --parameters "{'regressor_opts':{'classname':'netr32'}, 'training_opts':{'batch_size':128, 'n_epochs':50,'print_every':200}, 'dataset_opts':{'dataset':'mnist'}}"
Uses: latent_dataset.pth, jacobian_dataset.pth
Produces: models/test1/{regressor.pth, regressor_train_metrics.pth}

Apply the regressor to a new dataset and save the results
Command: python main.py --basename test1 --mode apply_regressor --parameters "{'dataset_opts':{'dataset':'mnist'}}"
Uses: regressor.pth
Produces: datasets/generated_datasets/test1/regressor_output.pth








Files in this project:

main.py : The main function that reads command line arguments and initiates all procedures
models.py : The actual neural network models
model\_functions.py : Functions and classes used in build\_models.py for training and testing.
build\_models.py : Construct model objects for running
performance\_tracking.py : Functions for tracking and logging train/test performance
config.py : Default parameters 
utils.py : Basic objects and functions used throughout
train.py : actually perform training
datasets.py : Load datasets
