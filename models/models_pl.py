import pytorch_lightning as pl
from utils import *


class Generic_model(pl.LightningModule):

    def __init__(self, config, encoder, decoder, classifier, optimizer):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.optimizer = optimizer
        self.config = config

        self.sweep_length = int(6 / self.config['sweeping_stepsize']) + 1

        # self.save_hyperparameters()        <-- Causes cifar-10 vae to freeze?

    def forward(self, imgs):

        latent_out, mu, logvar = self.encoder(imgs)

        x_generated = self.decoder(latent_out)

        # This loss seemed to work out a lot better for cifar10 for both of the models that we implemented
        if self.config["vae_model"] == "cifar10_cvae":
            nll = kl_divergence_loss(mu, logvar) + reconstruction_loss(x_generated, imgs)
        else:
            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

        causalEffect = 0
        if self.config['use_causal']:
            causalEffect, ceDebug = joint_uncond(self.config, self.decoder, self.classifier, self.device)
            loss = self.config['lam_ml'] * nll + causalEffect
        else:
            loss = nll
        return loss, causalEffect, nll

    @torch.no_grad()
    def sample(self, x_val):
        self.encoder.eval() # if we do not do this it raises issues for batch normalization
        self.decoder.eval()

        latentsweep_vals = np.arange(-3, 3 + self.config['sweeping_stepsize'], self.config['sweeping_stepsize']).tolist()
        samples = []
        labels = []
        labels_probs = []
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

                y = torch.argmax(y_probs, dim=-1)
                y_prob = torch.max(y_probs, dim=-1)
                labels.append(y)
                labels_probs.append(y_prob)
                samples.append(x_generated.squeeze(0))

        self.encoder.train() # make sure to set it back to training
        self.decoder.train()
        return samples, labels, labels_probs

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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        self.encoder.eval() # if we do not do this it raises issues for batch normalization
        self.decoder.eval()

        loss, causalEffect, nll = self.forward(batch[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_causalEffect", causalEffect, on_step=False, on_epoch=True)
        self.log("val_NLL", nll, on_step=False, on_epoch=True)

        self.encoder.train() # make sure to set it back to training
        self.decoder.train()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.encoder.eval() # if we do not do this it raises issues for batch normalization
        self.decoder.eval()

        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("test_bpd", bpd)

        self.encoder.train() # make sure to set it back to training
        self.decoder.train()
