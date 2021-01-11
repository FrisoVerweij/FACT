import MNIST
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import time
import MNIST_CNN_model
import CIFAR10_CNN_model
import MNIST_cvae_model
import CIFAR_cvae_model

from torchvision.utils import make_grid, save_image

@torch.no_grad()
def sample(encoder, decoder, batch_size, device):
    """
    Function for sampling a new batch of random images.
    """

    mean = torch.zeros((batch_size, encoder.z_dim)).to(device)
    log_std = torch.log(torch.ones((batch_size, encoder.z_dim)).to(device))
    z = encoder.reparameterize(mean, log_std)

    images = decoder(z)

    # We make a grid of these images
    grid = make_grid(images, nrow=int(batch_size / 8), normalize=True, range=(0, 1))
    #trainer.logger.experiment.add_image("images %i" % epoch, grid, global_step=epoch)

    # if save_to_disk is true we also save the image on the disk
    #if self.save_to_disk:
    #path = trainer.logger.log_dir + "/image_%i.png" % epoch
    save_image(grid, './image0.png')

    print("Logged images")

    return images

def select_dataloader(flags):
    '''
    Selects a dataloader given the hyperparameters
    :param flags: Set of hyperparameters
    :return: DataLoader
    '''

    # Parse the list of digits to include
    if flags.mnist_digits == "None":
        mnist_digits = None
    else:
        mnist_digits = flags.mnist_digits.split(',')
        mnist_digits = [int(digit) for digit in mnist_digits]

    if flags.dataset == "mnist":
        return MNIST.get_mnist_dataloaders(batch_size=flags.batch_size, digits_to_include=mnist_digits)

    elif flags.dataset == "fmnist":
        return MNIST.get_fmnist_dataloaders(batch_size=flags.batch_size, digits_to_include=mnist_digits)

    elif flags.dataset == "cifar10":
        return MNIST.get_cifar10_dataloaders(batch_size=flags.batch_size, digits_to_include=mnist_digits)

    else:
        raise Exception("No valid dataset selected!")


def select_model(flags):
    '''
    Selects a model given the hyperparameters
    :param flags: Set of hyperparameters
    :return: nn.module
    '''

    if flags.model in ['mnist_cnn', 'fmnist_cnn']:
        if flags.mnist_digits == "None":
            output_dim = 10
        else:
            output_dim = len(flags.mnist_digits.split(','))
        return MNIST_CNN_model.MNIST_CNN(output_dim)

    elif flags.model == 'cifar10_cnn':
        if flags.mnist_digits == "None":
            output_dim = 10
        else:
            output_dim = len(flags.mnist_digits.split(','))
        return CIFAR10_CNN_model.CIFAR10_CNN(output_dim)

    else:
        raise Exception("No valid model selected!")


def select_optimizer(flags, model, model_2=None, b1=0.5, b2=0.999):
    '''
    Selects an optimizer given the hyperparameters
    :param flags: Set of hyperparameters
    :param model: model that we want to optimize
    :return: nn.optim.optimizer
    '''
    if model_2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model_2.parameters())

    if flags.optimizer == "SGD":
        return torch.optim.SGD(params, lr=flags.lr, momentum=flags.momentum)
    elif flags.optimizer == "Adam":
        return torch.optim.Adam(params, lr=flags.lr, betas=(b1, b2))
    else:
        raise Exception("No valid optimizer selected!")


def encoder_decoder(flags, z_dim):
    if flags.encoder_decoder == "cvae" and flags.dataset in ["mnist", "fmnist"]:
        encoder = MNIST_cvae_model.Encoder(z_dim, 1, flags.x_dim)
        decoder = MNIST_cvae_model.Decoder(z_dim, 1, flags.x_dim)
    elif flags.encoder_decoder == "cvae" and flags.dataset == "cifar10":
        encoder = CIFAR_cvae_model.Encoder(z_dim, 3, flags.x_dim)
        decoder = CIFAR_cvae_model.Decoder(z_dim, 3, flags.x_dim)
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

# Not currently used
def load_pretrained_mnist(model, PATH, *args, **kwargs):
    modelB = model(*args, **kwargs)
    modelB.load_state_dict(torch.load(PATH), strict=False)
    return modelB