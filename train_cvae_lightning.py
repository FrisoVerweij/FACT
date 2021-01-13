import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import get_mnist_dataloaders
from models_vae import Encoder, Decoder
from train_cvae import VAE_LL_loss, joint_uncond
from utils import *
from visualize import visualize


class CVAE(pl.LightningModule):

    def __init__(self, z_dim, channel_dimension, x_dim, classifier, device, params, causal_effect, lam_ML, lr, betas,
                 yval, xval, image_size):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            model_name - String denoting what encoder/decoder class to use.  Either 'MLP' or 'CNN'
            hidden_dims - List of hidden dimensionalities to use in the MLP layers of the encoder (decoder reversed)
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
            lr - Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, channel_dimension, x_dim).to(device)
        self.decoder = Decoder(z_dim, channel_dimension, x_dim).to(device)

        self.classifier = classifier

        self.params = params
        self.use_causal_effect = causal_effect
        self.lam_ML = lam_ML

        #### We make sure we have standard data to generate the prictures from
        self.y_val = yval
        self.x_val = xval
        self.image_size = image_size

    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """
        latent_out, mu, logvar = self.encoder(imgs)

        x_generated = self.decoder(latent_out)

        nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

        causalEffect, ceDebug = joint_uncond(self.params, self.decoder, self.classifier, self.device)

        loss = self.use_causal_effect * causalEffect + self.lam_ML * nll

        return loss, causalEffect, nll

    @torch.no_grad()
    def sample(self):
        latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
        samples = []
        x_val = self.x_val
        x_val = x_val.to(self.device)

        z, mu, logvar = self.encoder(x_val)
        z = z.detach().cpu().numpy()
        z = z[0]
        for latent_dim in range(z_dim):
            for latent_val in latentsweep_vals:
                z_new = z.copy()
                z_new[latent_dim] += latent_val
                x_generated = self.decoder(torch.unsqueeze(torch.from_numpy(z_new), 0).to(self.device))

                samples.append(x_generated.squeeze(0))
        print(len(samples))
        return samples



    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        loss, causalEffect, nll = self.forward(batch[0])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_causalEffect", causalEffect, on_step=True, on_epoch=True)
        self.log("train_NLL", nll, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements

        loss, causalEffect, nll = self.forward(batch[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_causalEffect", causalEffect, on_step=False, on_epoch=True)
        self.log("val_NLL", nll, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("test_bpd", bpd)


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch + 1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        # Hints:
        # - You can access the logging directory path via trainer.logger.log_dir, and
        # - You can access the tensorboard logger via trainer.logger.experiment
        # - Use the torchvision function "make_grid" to create a grid of multiple images
        # - Use the torchvision function "save_image" to save an image grid to disk

        samples = pl_module.sample()
        grid = make_grid(samples, nrow=7 )
        name = 'samples_{}'.format(epoch)
        logger = trainer.logger.experiment
        logger.add_image('sample', grid, epoch)

        if self.save_to_disk:
            save_image(grid, trainer.logger.log_dir + '\\' + name + "_sample.png")


def train_cvae(args, z_dim, channel_dimension, x_dim, classifier, device, params, causal_effect, lam_ML, lr, betas,
               image_size):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader = get_mnist_dataloaders(digits_to_include=[3, 8], )

    ### Get the pictures
    data, targets = next(iter(val_loader))
    y_val = targets.numpy()
    x_val = data

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(every_n_epochs=1, save_to_disk=True)
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         log_every_n_steps=1,
                         callbacks=[gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = CVAE(z_dim, channel_dimension, x_dim, classifier, device, params, causal_effect, lam_ML, lr, betas, y_val,
                 x_val, image_size)

    # Training
    # gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    # # Testing
    # model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    #
    # # Manifold generation
    # if args.z_dim == 2:
    #     img_grid = visualize_manifold(model.decoder)
    #     save_image(img_grid,
    #                os.path.join(trainer.logger.log_dir, 'vae_manifold.png'),
    #                normalize=False)
    #
    # return test_result


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
