## Requirements
- Numpy
- PyTorch
- PyTorch Lightning
- yaml


#### Reproducing the experiments

In general every experiment consists of 2 parts.
1) Training the classifier this is done with `train_classifier.py`
2) Training the explanation model `train_cvae_lightning.py`

Both scripts have one argument that is used namely the `--config` argument. 
In the directory `./config` you can find the config files for all the experiments

There are some notable exceptions. For Fashionmnist 034 we used the pretrained classifier of the authors.
This means that the classifier does not have to be trained in that case. 

### Creating a classifier
To create a classifier, run the file train_classifier.py and specify a config file.
The config file contains all necessary information to run an experiment.

### Creating a generative model
To create a generative model, run the file train_cvae_lightning.py and specify a config file.
This config file should be the same as the config file used to create the classifier.

### Config
The config file specifies all necessary information to run an experiment.
For example, it specifies which models are used and how long the training process is.
All hyperparameters are explained below:

#### Hyperparameters
The following hyperparameters general parameters:
- save_dir:             Directory where to save and load the models
- device:               Either 'cpu' or 'gpu'
- seed:                 Seed for reproducibility
- num_workers:          Number of workers for the dataloader
- log_dir:              Folder where to store the Tensorboard logs
- progress_bar:         Indicate whether to show the progress when training with PyTorch Lightning
- n_samples_each_class: Indicate how many samples are created for each class when displaying the sweeping of the latent variables
- max_images:		Either an integer to limit the number of generated image, or 'null' to remove limit
- callback_every:	Generate images on every x epoch
- callback_digits:	Either True of False. Whether to generate images that display latent changes for each digit
- sweeping_stepsize:    Increase or decrease to either make the generated images more coarse grained or fine grained
- show_probs:		Whether to show the classifier probabilities in the image border


The following hyperparameters specify the dataset:
- dataset:              Specifies dataset used. Choices are 'mnist', 'fmnist', or 'cifar10'
- include_classes:      Specifies which classes to include in the training process. Either '1,3,5...' or 'null' for all classes

The following hyperparameters are used when training the classifier:
- classifier:           Specifies model to train. Choices are 'mnist_cnn', 'fmnist_cnn', 'vgg_11' or 'dummy'
- model_name:           Name of the saved classifier. Necessary for the generative model to find the classifier
- epochs:               Number of epochs
- batch_size:           Batch size
- lr:                   Learning rate
- momentum:             Momentum
- optimizer:            Either 'SGD' or 'Adam'
- no_print:             If True, only print when validating

The following hyperparameters are used when training the generative model (vae):
- vae_model:            Specifies model to train. Choices are 'mnist_cvae', 'fmnist_cvae' or 'cifar10_cvae'
- model_name:           Name of the classifier model
- use_causal:           If True, use the causal term to train the generative model. False otherwise
- n_alpha:              Number of causal factors
- n_beta:               Number of non-causal factors
- alpha_samples:        Number of alpha samples used to approximate causal term
- beta_samples:         Number of beta samples used to approximate causal term 
- lam_ml:               Lambda term. Increase to decrease relative importance of the causal term
- batch_size:           Batch size
- lr:                   Learning rate
- b1:                   Beta 1 (for Adam optimizer)
- b2:                   Beta 2 (for Adam optimizer)
- weight_decay:         Weight decay
- optimizer:            Either 'SGD' or 'Adam'
- image_size:           Size of the input images. 28 means 28x28
- epochs:               Number of epochs