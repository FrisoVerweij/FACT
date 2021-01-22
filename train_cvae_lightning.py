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

    classifier = load_classifier(config)

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
    gen_callback_digit = GenerateCallbackDigit(x_val, dataset=config['dataset'],
                                               every_n_epochs=config['callback_every'],
                                               n_samples=n_samples_total, save_to_disk=True,
                                               show_prob=config['show_probs'])
    gen_callback_latent = GenerateCallbackLatent(x_val, dataset=config['dataset'],
                                                 every_n_epochs=config['callback_every'],
                                                 latent_dimensions=number_of_latents, n_samples=n_samples_total,
                                                 save_to_disk=True, show_prob=config['show_probs'])

    callbacks = [gen_callback_digit, gen_callback_latent] if config['callback_digits'] else [gen_callback_latent]

    trainer = pl.Trainer(default_root_dir=config["log_dir"],
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=config['epochs'],
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=1 if config["progress_bar"] else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    model = Generic_model(config, encoder, decoder, classifier, optimizer).to(config['device'])

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    ### At the end we save the encoder and decoder in it's enterity
    torch.save(encoder.state_dict(), config['encoder_path'])
    torch.save(decoder.state_dict(), config['decoder_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8_final.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)
    print(config)

    train_cvae_pl(config)
