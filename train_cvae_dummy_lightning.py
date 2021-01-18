import argparse
import os
import random

import yaml
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.generators import GenerateCallbackDigit, GenerateCallbackLatent
from dataset import get_mnist_dataloaders
from models.models_pl import CVAE

from utils import *


# def modify_data(data):
#     data[:, :, 0, 0] = random.randint(0, 1)
#     return data


def train_cvae_pl(config):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """
    pl.seed_everything(config["seed"])  # To be reproducible

    os.makedirs(config['log_dir'], exist_ok=True)
    train_loader, val_loader = get_mnist_dataloaders(digits_to_include=config['mnist_digits'], modify=True)

    ### Get the pictures
    data, targets = next(iter(val_loader))
    y_val = targets.numpy()
    x_val = data

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
    model = CVAE(z_dim, config['channel_dimension'], x_dim, classifier, config, device=device)

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.encoder.state_dict(),
               str(config['save_dir']) + str(config['vae_model']) + "_encoder" + "_" + config['model_name'])
    torch.save(model.decoder.state_dict(),
               str(config['save_dir']) + str(config['vae_model']) + "_decoder" + "_" + config['model_name'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_local/mnist_dummy.yml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)
    # The device to run the model on
    device = config['device']
    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size'] ** 2

    from models.models_classifiers import BiggestDummy
    classifier = BiggestDummy(config['image_size']**2)
    # classifier = select_classifier(config)
    # classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier'] + "_" + config['model_name']))
    classifier.to(device)

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
    }

    config = {**config, **params}

    train_cvae_pl(config)
