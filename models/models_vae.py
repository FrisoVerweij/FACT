import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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


class Encoder_cifar10(nn.Module):

    def __init__(self, z_dim, channel_dimension, x_dim,
                 filt_per_layer=128):  # x_dim : total number of pixels
        super(Encoder_cifar10, self).__init__()
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


class Decoder_cifar10(nn.Module):

    def __init__(self, z_dim, channel_dimenion, x_dim,  # x_dim : total number of pixels
                 filt_per_layer=128):
        super(Decoder_cifar10, self).__init__()
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


class Encoder_cifar10_sasha(nn.Module):

    def __init__(self, z_dim, channel_dimension, x_dim,
                 filt_per_layer=128):  # x_dim : total number of pixels
        super(Encoder_cifar10_sasha, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.Conv2d(channel_dimension, filt_per_layer//4 ,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer//4),
            nn.ReLU(),
            nn.Conv2d(filt_per_layer // 4, filt_per_layer // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer // 2),
            nn.ReLU(),
            nn.Conv2d(filt_per_layer // 2, filt_per_layer, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer),
            nn.ReLU(),
        )

        # encoded feature's size and volume
        self.feature_size = x_dim // 8
        self.feature_volume = filt_per_layer * (self.feature_size ** 2)

        self.fc_mu = nn.Linear(self.feature_volume, z_dim)
        self.fc_logvar = nn.Linear(self.feature_volume, z_dim)

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


class Decoder_cifar10_sasha(nn.Module):

    def __init__(self, z_dim, channel_dimenion, x_dim,  # x_dim : total number of pixels
                 filt_per_layer=128):
        super(Decoder_cifar10_sasha, self).__init__()
        self.x_dim = x_dim
        self.c_dim = channel_dimenion
        self.filt_per_layer = filt_per_layer
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer // 2, filt_per_layer // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer // 4, channel_dimenion, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channel_dimenion),
            nn.Sigmoid(),
        )

        # encoded feature's size and volume
        self.feature_size = x_dim // 8
        self.feature_volume = filt_per_layer * (self.feature_size ** 2)

        self.fc = nn.Linear(z_dim, self.feature_volume)

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1,
                            int(self.feature_size),
                            int(self.feature_size))

        x = self.model(t)
        return x


