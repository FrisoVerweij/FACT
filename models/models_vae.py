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
            nn.Conv2d(channel_dimension, filt_per_layer // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filt_per_layer // 4),
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


class Conv_Block(nn.Module):
    def __init__(self, input, output, kernel, stride, padding, bias):
        super(Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input, output, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )

    def forward(self, imgs):
        return self.block(imgs)


class Conv_Block_Transpose(nn.Module):
    def __init__(self, input, output, kernel, stride, padding, bias):
        super(Conv_Block_Transpose, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )

    def forward(self, imgs):
        return self.block(imgs)


class Encoder_own(nn.Module):
    def __init__(self, nz, nc, x_dim, ngf, ngpu=None):
        super(Encoder_own, self).__init__()
        # nz is the length of z
        # ngf is the size of the feature maps
        # nc is the number of channels

        #  conv output width = (W_in - W_conv + 2*pad) / stride   + 1
        #  pytorch convolustions have arguments nn.Conv2d(input_c, output_c, kernel, stride, padding)
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # input size = batch, 3, 32, 32
            Conv_Block(nc, ngf, 3, 1, 1, bias=False),
            Conv_Block(ngf, ngf * 2, 4, 1, 0, bias=False),
            # output size = batch, ngf*2, 29, 29

            Conv_Block(ngf * 2, ngf * 3, 3, 1, 1, bias=False),
            Conv_Block(ngf * 3, ngf * 4, 4, 2, 1, bias=False),
            # output size = batch, ngf*4, 14, 14

            Conv_Block(ngf * 4, ngf * 5, 3, 1, 1, bias=False),
            Conv_Block(ngf * 5, ngf * 6, 4, 2, 1, bias=False),
            # output size = batch, ngf*6, 7, 7

            Conv_Block(ngf * 6, ngf * 7, 3, 1, 1, bias=False),
            Conv_Block(ngf * 7, ngf * 8, 4, 2, 1, bias=False),
            # output size = batch, ngf*8, 3, 3

        )

        self.fc_mu = nn.Linear(ngf * 8 * 3 * 3, nz)
        self.fc_logvar = nn.Linear(ngf * 8 * 3 * 3, nz)

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


class Decoder_own(nn.Module):
    def __init__(self, nz, nc, x_dim, ngf, ngpu=None):
        super(Decoder_own, self).__init__()
        # nz is the length of z
        # ngf is the size of the feature maps
        # nc is the number of channels
        self.nc = nc
        self.nz = nz
        self.ngf = ngf

        #  convTranspose output width = (W_in - 1) * stride - 2*padding + kernel_size
        #  pytorch convolustions have arguments nn.ConvTranspose2d(input_c, output_c, kernel, stride, padding)
        self.ngpu = ngpu
        self.model = nn.Sequential(

            # input size = batch, ngf*8, 3, 3
            Conv_Block_Transpose(ngf * 8, ngf * 7, 3, 1, 1, bias=False),
            Conv_Block_Transpose(ngf * 7, ngf * 6, 4, 1, 1, bias=False),
            # output size = batch, ngf*6, 4, 4

            Conv_Block_Transpose(ngf * 6, ngf * 5, 3, 1, 1, bias=False),
            Conv_Block_Transpose(ngf * 5, ngf * 4, 4, 2, 1, bias=False),
            # output size = batch, ngf*4, 8, 8

            Conv_Block_Transpose(ngf * 4, ngf * 3, 3, 1, 1, bias=False),
            Conv_Block_Transpose(ngf * 3, ngf * 2, 4, 2, 1, bias=False),
            # output size = batch, ngf*2, 16, 16

            Conv_Block_Transpose(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output size = batch, nc, 32, 32
        )

        self.fc = nn.Linear(nz, self.ngf * 8 * 3 * 3)

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, self.ngf * 8, 3, 3)
        x = self.model(t)

        return x


class Encoder_captain(nn.Module):
    def __init__(self, latent_variable_size, nc, ngf, ndf, device='cuda'):
        super(Encoder_captain, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.device = device

        # Encoder
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(128, 512)  # 14 * 14 * 32 -> 128
        self.fc1 = nn.Linear(512, latent_variable_size)
        self.fc2 = nn.Linear(512, latent_variable_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        # print((x>0.000).sum())
        w_mean = self.fc1(x)
        w_std = self.fc2(x)

        return w_mean, w_std

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.Tensor(std.size()).normal_().to(self.device)

        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar


class Decoder_captain(nn.Module):
    def __init__(self, latent_variable_size, nc, ngf, ndf, device='cuda'):
        super(Decoder_captain, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.device = device

        # Decoder
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(latent_variable_size, 500)
        self.fc4 = nn.Linear(500, 128)  # 14 * 14 * 32 -> 128

        #  convTranspose output width = (W_in - 1) * stride - 2*padding + kernel_size
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def decode(self, z):
        x = self.fc3(z)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        # print((x>0.000).sum())
        x = x.view(-1, 32, 2, 2)  # 14, 14 -> 2, 2
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.sigmoid(x)

        return x

    def forward(self, z):
        res = self.decode(z)
        return res
