import dataset
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import time
import models_classifiers
import models_vae

from torchvision.utils import make_grid, save_image

def select_dataloader(config):
    '''
    Selects a dataloader given the hyperparameters
    :param config: Set of hyperparameters
    :return: DataLoader
    '''

    # Parse the list of digits to include
    if config["mnist_digits"] == None:
        mnist_digits = None
    else:
        mnist_digits = config["mnist_digits"]

    if config["dataset"] == "mnist":
        return dataset.get_mnist_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "mnist_overfit":
        return dataset.get_mnist_overfit_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "fmnist":
        return dataset.get_fmnist_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "cifar10":
        return dataset.get_cifar10_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    else:
        raise Exception("No valid dataset selected!")


def select_classifier(config):
    '''
    Selects a model given the hyperparameters
    :param config: Set of hyperparameters
    :return: nn.module
    '''

    if config["classifier"] in ['mnist_cnn', 'fmnist_cnn']:
        if config["mnist_digits"] == None:
            output_dim = 10
        else:
            output_dim = len(config["mnist_digits"])
        return models_classifiers.MNIST_CNN(output_dim)

    elif config["classifier"] == 'mnist_cnn_overfit':
        if config["mnist_digits"] == "None":
            output_dim = 10
        else:
            output_dim = len(config["mnist_digits"])
        return models_classifiers.MNIST_CNN_Overfit(output_dim)

    elif config["classifier"] == 'cifar10_cnn':
        if config["mnist_digits"] == "None":
            output_dim = 10
        else:
            output_dim = len(config["mnist_digits"])
        return models_classifiers.CIFAR10_CNN(output_dim)

    else:
        raise Exception("No valid model selected!")


def select_optimizer(config, model, model_2=None):
    '''
    Selects an optimizer given the hyperparameters
    :param config: Set of hyperparameters
    :param model: model that we want to optimize
    :return: nn.optim.optimizer
    '''
    if model_2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model_2.parameters())

    if config["optimizer"] == "SGD":
        return torch.optim.SGD(params, lr=config["lr"], momentum=config["momentum"])
    elif config["optimizer"] == "Adam":
        return torch.optim.Adam(params, lr=config["lr"], betas=(config["b1"], config["b2"]))
    else:
        raise Exception("No valid optimizer selected!")


def select_vae_model(config, z_dim):
    if config["vae_model"] in ["mnist_cvae", "fmnist_cvae"]:
        encoder = models_vae.Encoder(z_dim, 1, config["image_size"] ** 2)
        decoder = models_vae.Decoder(z_dim, 1, config["image_size"] ** 2)
    elif config["vae_model"] == "cifar10_cvae":
        encoder = models_vae.Encoder_cifar10(z_dim, 3, config["image_size"]**2)
        decoder = models_vae.Decoder_cifar10(z_dim, 3, config["image_size"]**2)
    else:
        raise Exception("No valid encoder/decoder selected!")

    return encoder, decoder

# Very basic weight initialization method
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_classifier_config(config):
    return  {**config['classifier'], **config['general'],**config['dataset'], }


def to_vae_config(config):
    return {**config['vae'], **config['general'],**config['dataset'], "classifier": config["classifier"]["classifier"]}

def to_visualize_config(config):
    return {**config['vae'], **config['general'], **config['dataset'], "classifier": config["classifier"]["classifier"]}