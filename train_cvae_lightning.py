import argparse
import os

import yaml
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.generators import GenerateCallbackDigit, GenerateCallbackLatent
from dataset import get_mnist_dataloaders
from models.models_pl import CVAE

from utils import *

import numpy as np


def get_x_vals(val_loader, n_classes=2, n_for_each_class=4):
    ###
    data, targets = next(iter(val_loader))
    y_val = targets.numpy()

    indices = [

    ]
    for i in range(n_classes):
        indices += list(np.where(y_val == i)[0])[:n_for_each_class]

    indices = torch.tensor(indices)

    x_val = torch.index_select(data, 0, indices)
    return x_val


def train_cvae_pl(config):
    """
    Function for training and testing a VAE model.
    Inputs:
       config:

    """
    config = prepare_variables_pl(config)

    pl.seed_everything(config["seed"])  # To be reproducible

    os.makedirs(config['log_dir'], exist_ok=True)
    train_loader, val_loader = get_mnist_dataloaders(digits_to_include=config['mnist_digits'])

    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier']))
    classifier.to(config['device'])

    x_val = get_x_vals(val_loader)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback_digit = GenerateCallbackDigit(x_val, every_n_epochs=1, save_to_disk=True)
    gen_callback_latent = GenerateCallbackLatent(x_val, every_n_epochs=1, save_to_disk=True)

    trainer = pl.Trainer(default_root_dir=config["log_dir"],
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=config['epochs'],
                         log_every_n_steps=1,
                         callbacks=[gen_callback_digit, gen_callback_latent],
                         progress_bar_refresh_rate=1 if config["progress_bar"] else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model

    model = CVAE(config["z_dim"], config['channel_dimension'], config["x_dim"], classifier, config,
                 device=config["device"])

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    #### Calculate and return metric


def prepare_variables_pl(config):
    # The device to run the model on
    device = config['device']
    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size'] ** 2
    image_size = config['image_size']

    if config['mnist_digits'] == None:
        n_classes = 10
    else:
        n_classes = len(config['mnist_digits'])

    params = {
        "number_of_classes": n_classes,
        "alpha_samples": config['alpha_samples'],
        "beta_samples": config['beta_samples'],
        "z_dim": z_dim,
        "n_alpha": config['n_alpha'],
        "n_beta": config['n_beta'],
        "channel_dimension": 1,
        "x_dim": x_dim
    }

    config = {**config, **params}
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)

    train_cvae_pl(config)
