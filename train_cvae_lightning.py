import argparse
import os

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.generators import GenerateCallbackDigit, GenerateCallbackLatent

import numpy as np
import torch

from models.models_pl import Generic_model
from utils import *


def get_x_vals(val_loader, n_classes=2, n_for_each_class=4):
    ###

    # Here we get a sufficient number of samples to get the images from
    data, targets = next(iter(val_loader))
    for x, y in val_loader:
        data = torch.cat([data, x], dim=0)
        targets = torch.cat([targets, y], dim=0)

        # Here we make sure that we have all the classes and that we have enough samples
        if len(torch.unique(targets)) >= n_classes:
            y_val = targets.numpy()
            indices = []
            for i in range(n_classes):
                indices += list(np.where(y_val == i)[0])[:n_for_each_class]

            if indices == n_classes*n_for_each_class:
                break

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

    # Select the data
    train_loader, val_loader = select_dataloader(config)

    # Select and load the classifier
    classifier = select_classifier(config)
    if config["classifier"] != "mnist_dummy":

        if config["model_name"][-3:] == ".pt":
            classifier.load_state_dict(
                torch.load(config['save_dir']  + config['model_name'])['model_state_dict_classifier'])
        else:
            classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier'] + "_" + config['model_name']))

    # Select the encoder, decoder and optimizer
    encoder, decoder = select_vae_model(config)
    optimizer = select_optimizer(config, encoder, decoder)

    # What does this do? Is it just for the callbacks (also the n_samples_total below?)
    x_val = get_x_vals(val_loader, n_classes=config['number_of_classes'],
                       n_for_each_class=config['n_samples_each_class'])

    n_samples_total = config['number_of_classes'] * config['n_samples_each_class']

    if config['max_images'] is not None:
        number_of_latents = config['max_images']
    else:
        number_of_latents = config['n_alpha'] + config['n_beta']

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback_digit = GenerateCallbackDigit(x_val, dataset=config['dataset'], every_n_epochs=1, n_samples=n_samples_total, save_to_disk=True)
    gen_callback_latent = GenerateCallbackLatent(x_val,  dataset=config['dataset'], every_n_epochs=1, latent_dimensions=number_of_latents, n_samples=n_samples_total, save_to_disk=True)

    trainer = pl.Trainer(default_root_dir=config["log_dir"],
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=config['epochs'],
                         log_every_n_steps=1,
                         callbacks=[gen_callback_digit, gen_callback_latent],
                         progress_bar_refresh_rate=1 if config["progress_bar"] else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    model = Generic_model(config, encoder, decoder, classifier, optimizer).to(config['device'])

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)
    print(config)

    train_cvae_pl(config)