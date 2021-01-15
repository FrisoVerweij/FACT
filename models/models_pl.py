import pytorch_lightning as pl

from models.models_vae import Encoder, Decoder
#from train_cvae import VAE_LL_loss, joint_uncond
from utils import *


class CVAE(pl.LightningModule):

    def __init__(self, z_dim, channel_dimension, x_dim, classifier, params, device="cpu"):
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
        self.lr = params['lr']
        self.betas = (params['b1'], params['b2'])
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, channel_dimension, x_dim).to(device)
        self.decoder = Decoder(z_dim, channel_dimension, x_dim).to(device)

        self.classifier = classifier

        self.params = params
        self.use_causal = params["use_causal"]
        self.lam_ML = params["lam_ml"]

        self.image_size = params["image_size"]

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

        loss = self.use_causal * causalEffect + self.lam_ML * nll

        return loss, causalEffect, nll

    @torch.no_grad()
    def sample(self, x_val):
        latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
        samples = []
        labels = []
        x_val = x_val.to(self.device)

        z, mu, logvar = self.encoder(x_val)
        z = z.detach().cpu().numpy()
        z = z[0]
        for latent_dim in range(self.z_dim):
            for latent_val in latentsweep_vals:
                z_new = z.copy()
                z_new[latent_dim] += latent_val
                x_generated = self.decoder(torch.unsqueeze(torch.from_numpy(z_new), 0).to(self.device))
                y, y_probs = self.classifier(x_generated)

                y = torch.argmax(y_probs, dim=1)

                labels.append(y)
                samples.append(x_generated.squeeze(0))
        return samples, labels

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, betas=self.betas)
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




class Generic_model(pl.LightningModule):

    def __init__(self, config, encoder, decoder, classifier, optimizer):
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
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.optimizer = optimizer
        self.config = config

        self.save_hyperparameters()

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

        causalEffect, ceDebug = joint_uncond(self.config, self.decoder, self.classifier, self.device)

        loss = self.config['use_causal'] * causalEffect + self.config['lam_ml'] * nll

        return loss, causalEffect, nll

    @torch.no_grad()
    def sample(self, x_val):
        latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
        samples = []
        labels = []
        x_val = x_val.to(self.device)

        z, mu, logvar = self.encoder(x_val)
        z = z.detach().cpu().numpy()
        z = z[0]
        for latent_dim in range(self.config['z_dim']):
            for latent_val in latentsweep_vals:
                z_new = z.copy()
                z_new[latent_dim] += latent_val
                x_generated = self.decoder(torch.unsqueeze(torch.from_numpy(z_new), 0).to(self.device))
                y, y_probs = self.classifier(x_generated)

                y = torch.argmax(y_probs, dim=1)

                labels.append(y)
                samples.append(x_generated.squeeze(0))
        return samples, labels

    def configure_optimizers(self):
        # Create optimizer
        optimizer = self.optimizer
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
