from __future__ import division

import numpy as np
import torch

import matplotlib.pyplot as plt

from author_code.util import *
from author_code.load_mnist import *
from MNIST_dataloader import select_dataloader
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config_visualize.yml')
args = parser.parse_args()

config = yaml.load(open(args.config, "r"))

# --- parameters ---
# z_dim     = 8
# alpha_dim = 1
# c_dim     = 1
# img_size  = 28
# class_use = np.array([3,8])
# latent_sweep_vals = np.linspace(-2,2,25)
# latent_sweep_plot = [0,4,8,12,16,20,24]
# classifier_file = 'pretrained_models/pretrained_models_local/mnist_cnn'
# vae_file = 'pretrained_models/mnist_cvae'

class_use = np.array(config['mnist_digits'])
latent_sweep_plot = config['latent_sweep_vals']
latent_sweep_vals = np.linspace(*config['latent_sweep_vals'])

# --- initialize ---
class_use_str = np.array2string(class_use)
# y_dim = class_use.shape[0]
y_dim = len(class_use)
newClass = range(0, y_dim)
nsweep = len(latent_sweep_vals)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- load test data ---
# X, Y, tridx = load_mnist_classSelect('train',class_use,newClass)
# vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
# ntrain = X.shape[0]
traindata_loader, testdata_loader = select_dataloader(config)
# x_val, y_val = testdata_loader

data, targets = next(iter(testdata_loader))
y_val = targets.numpy()
x_val = np.reshape(data, (data.shape[0], data.shape[2], data.shape[3], data.shape[1]))


# --- load VAE ---
from MNIST_CVAE_model import Decoder, Encoder

encoder = Encoder(config['z_dim'], config['c_dim'], config['img_size']**2).to(device)
decoder = Decoder(config['z_dim'], config['c_dim'], config['img_size']**2).to(device)
# checkpoint_vae = torch.load("author_code/pretrained_models/mnist_38_vae/model.pt", map_location=device)
# encoder.load_state_dict(checkpoint_vae['model_state_dict_encoder'], strict=False)
# decoder.load_state_dict(checkpoint_vae['model_state_dict_decoder'], strict=False)
encoder.load_state_dict(torch.load(config['encoder_file'], map_location=device),strict=False)
decoder.load_state_dict(torch.load(config['decoder_file'], map_location=device),strict=False)

# --- load classifier ---
from MNIST_CNN_model import MNIST_CNN as CNN

classifier = CNN(y_dim).to(device)
classifier.load_state_dict(torch.load(config['classifier_file'], map_location=device), strict=False)
# checkpoint_model = torch.load(classifier_file, map_location=device)
# classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

# %% generate latent factor sweep plot
sample_ind = np.concatenate((np.where(y_val == 0)[0][:4],
                             np.where(y_val == 1)[0][:4]))
cols = [[0.047, 0.482, 0.863], [1.000, 0.761, 0.039], [0.561, 0.788, 0.227]]
border_size = 3
nsamples = len(sample_ind)
latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
Xhats = np.zeros((config['z_dim'], nsamples, len(latentsweep_vals), config['img_size'], config['img_size'], 1))
yhats = np.zeros((config['z_dim'], nsamples, len(latentsweep_vals)))
for isamp in range(nsamples):
    x = torch.from_numpy(np.expand_dims(x_val[sample_ind[isamp]], 0))
    x_torch = x.permute(0, 3, 1, 2).float().to(device)
    z = encoder(x_torch)[0][0].detach().cpu().numpy()
    for latent_dim in range(config['z_dim']):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            ztilde = z.copy()
            ztilde[latent_dim] += latentsweep_val
            xhat = decoder(torch.unsqueeze(torch.from_numpy(ztilde), 0).to(device))
            yhat = np.argmax(classifier(xhat)[0].detach().cpu().numpy())
            img = 1. - xhat.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
            Xhats[latent_dim, isamp, ilatentsweep, :, :, 0] = img
            yhats[latent_dim, isamp, ilatentsweep] = yhat

for latent_dim in range(config['z_dim']):
    fig, axs = plt.subplots(nsamples, len(latentsweep_vals))
    for isamp in range(nsamples):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            img = Xhats[latent_dim, isamp, ilatentsweep, :, :, 0].squeeze()
            yhat = int(yhats[latent_dim, isamp, ilatentsweep])
            img_bordered = np.tile(np.expand_dims(np.array(cols[yhat]), (0, 1)),
                                   (config['img_size'] + 2 * border_size, config['img_size'] + 2 * border_size, 1))
            img_bordered[border_size:-border_size, border_size:-border_size, :] = \
                np.tile(np.expand_dims(img, 2), (1, 1, 3))
            axs[isamp, ilatentsweep].imshow(img_bordered, interpolation='nearest')
            axs[isamp, ilatentsweep].axis('off')
    axs[0, round(len(latentsweep_vals) / 2) - 1].set_title('Sweep latent dimension %d' % (latent_dim + 1))
    if True:
        print('Exporting latent dimension %d...' % (latent_dim + 1))
        plt.savefig('./figures/fig_mnist_qual_latentdim%d.svg' % (latent_dim + 1), bbox_inches=0)

print('Columns - latent values in sweep: ' + str(latentsweep_vals))
print('Rows - sample indices in vaX: ' + str(sample_ind))
