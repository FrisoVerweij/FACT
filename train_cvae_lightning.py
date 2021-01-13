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




def train_cvae(args, z_dim, channel_dimension, x_dim, classifier, device, params, causal_effect, lam_ML, lr, betas,
               image_size):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader = get_mnist_dataloaders(digits_to_include=[3, 8])

    ### Get the pictures
    data, targets = next(iter(val_loader))
    y_val = targets.numpy()
    x_val = data

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback_digit = GenerateCallbackDigit(x_val, every_n_epochs=1, save_to_disk=True)
    gen_callback_latent = GenerateCallbackLatent(x_val, every_n_epochs=1, save_to_disk=True)

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         log_every_n_steps=1,
                         callbacks=[gen_callback_digit, gen_callback_latent],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = CVAE(z_dim, channel_dimension, x_dim, classifier, device, params, causal_effect, lam_ML, lr, betas, image_size)

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')
    # Other hyperparameters
    parser.add_argument('--epochs', default=20, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='CVAE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true', default=True,
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)
    # The device to run the model on
    device = config['device']
    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size'] ** 2
    image_size = config['image_size']
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier']))
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
        "n_beta": config['n_beta']
    }

    channel_dimension = 1
    train_cvae(args, z_dim, channel_dimension, x_dim, classifier, device, params, config['use_causal'],
               config['lam_ml'], config['lr'], (config['b1'], config['b2']), image_size)
