import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, z_dim, channel_dimension, x_dim,
                 filt_per_layer=64):  # x_dim : total number of pixels
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(int(channel_dimension), filt_per_layer, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=0),
            nn.ReLU(),
        )
        self.z_dim = z_dim
        self.fc_mu = nn.Linear(int(filt_per_layer * x_dim / 16), z_dim)
        self.fc_logvar = nn.Linear(int(filt_per_layer * x_dim / 16), z_dim)

    def encode(self, x):
        z = self.model(x)
        z = z.view(z.shape[0], -1)
        return self.fc_mu(z), self.fc_logvar(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dimenion, x_dim,  # x_dim : total number of pixels
                 filt_per_layer=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = channel_dimenion
        self.x_dim = x_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, int(filt_per_layer * self.x_dim / 16)),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, int(channel_dimenion), 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1,
                            int(np.sqrt(self.x_dim) / 4),
                            int(np.sqrt(self.x_dim) / 4))

        x = self.model(t)

        return x

class Encoder_cifar(nn.Module):

    def __init__(self, z_dim, channel_dimension, x_dim,
                 filt_per_layer=128):  # x_dim : total number of pixels
        super(Encoder_cifar, self).__init__()
        self.z_dim = z_dim
        #  conv output width = (W_in - W_conv + 2*pad) / stride   + 1
        self.model = nn.Sequential(
            nn.Conv2d(channel_dimension, filt_per_layer, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, filt_per_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, 2 * filt_per_layer, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            nn.ReLU(),
            nn.Conv2d(2 * filt_per_layer, 2 * filt_per_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filt_per_layer, 2 * filt_per_layer, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            nn.ReLU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(2 * 16 * filt_per_layer, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(2 * 16 * filt_per_layer, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
        )

    def encode(self, x):
        z = self.model(x)
        z = z.view(z.shape[0], -1)
        return self.fc_mu(z), self.fc_logvar(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder_cifar(nn.Module):

    def __init__(self, z_dim, channel_dimenion, x_dim,  # x_dim : total number of pixels
                 filt_per_layer=128):
        super(Decoder_cifar, self).__init__()
        self.z_dim = z_dim
        self.c_dim = channel_dimenion
        self.x_dim = x_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * filt_per_layer, 2 * filt_per_layer, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            nn.ReLU(),
            nn.Conv2d(2 * filt_per_layer, 2 * filt_per_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filt_per_layer, filt_per_layer, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, filt_per_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, channel_dimenion, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * 16 * filt_per_layer),
            nn.ReLU()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1, 4, 4)
        x = self.model(t)
        return x

