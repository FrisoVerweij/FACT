from datetime import time

import torch
import numpy as np
import yaml
import argparse

import MNIST_dataloader
from MNIST_CNN_model import MNIST_CNN
from MNIST_CVAE_model import Encoder, Decoder


def train_cvae(encoder, decoder, classifier, dataloader, n_epochs, optimizer, device, params, config, use_causal_effect=True,
               lam_ML=0.000001, ):
    # --- train ---
    for i in range(n_epochs):
        for x, y in dataloader:
            inputs = x.to(device)
            targets = y.to(device)
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(inputs)
            x_generated = decoder(latent_out)
            nll, nll_mse, nll_kld = VAE_LL_loss(inputs, x_generated, logvar, mu)

            causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)

            loss = use_causal_effect * causalEffect + lam_ML * nll

            loss.backward()
            optimizer.step()

            print(loss.item())
        print(i)

    torch.save(encoder.state_dict(), str(config['save_dir']) + str(config['model_name']) + "_encoder")
    torch.save(decoder.state_dict(), str(config['save_dir']) + str(config['model_name']) + "_decoder")

    return encoder, decoder


def VAE_LL_loss(Xbatch, Xest, logvar, mu):
    batch_size = Xbatch.shape[0]
    sse_loss = torch.nn.MSELoss(reduction='sum')  # sum of squared errors
    KLD = 1. / batch_size * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = 1. / batch_size * sse_loss(Xest, Xbatch)
    auto_loss = mse + KLD
    return auto_loss, mse, KLD


def joint_uncond(params, decoder, classifier, device):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['number_of_classes']).to(device)
    zs = np.zeros((params['alpha_samples'] + params['beta_samples'], params['z_dim']))  ### placeholder for the samples
    for i in range(0, params['alpha_samples']):
        alpha = np.random.randn(params['n_alpha'])
        zs = np.zeros((params['beta_samples'], params['z_dim']))
        for j in range(0, params['beta_samples']):
            beta = np.random.randn(params['n_beta'])
            zs[j, :params['n_alpha']] = alpha
            zs[j, params['n_alpha']:] = beta
        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[1]
        p = 1. / float(params['beta_samples']) * torch.sum(yhat, 0)  # estimate of p(y|alpha)
        I = I + 1. / float(params['alpha_samples']) * torch.sum(torch.mul(p, torch.log(p + eps)))
        q = q + 1. / float(params['alpha_samples']) * p  # accumulate estimate of p(y)

    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    negCausalEffect = -I
    info = {"xhat": xhat, "yhat": yhat}
    return negCausalEffect, info


def load_pretrained_mnist(model, PATH, *args, **kwargs):
    modelB = model(*args, **kwargs)
    modelB.load_state_dict(torch.load(PATH), strict=False)
    return modelB


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))

    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size']**2

    z_dim = config['n_alpha'] + config['n_beta']

    encoder = Encoder(z_dim, 1, x_dim).to(device)
    decoder = Decoder(z_dim, 1, x_dim).to(device)

    PATH = './pretrained_models_local/mnist_cnn_new'

    classifier = load_pretrained_mnist(MNIST_CNN, PATH, 2).to(device)

    train_dataset, test_dataset = MNIST_dataloader.get_mnist_dataloaders(batch_size=config['batch_size'], digits_to_include=config['mnist_digits'])

    # todo: all these params should also be in config file, but maybe there should be a separate one for classifier/cvae
    params_use = list(decoder.parameters()) + list(encoder.parameters())
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999
    optimizer = torch.optim.Adam(params_use, lr=lr, betas=(b1, b2))
    params = {
        "number_of_classes": 2,
        "alpha_samples": 10,
        "beta_samples": 10,
        "z_dim": z_dim,
        "n_alpha": config['n_alpha'],
        "n_beta": config['n_beta']
    }

    encoder, decoder = train_cvae(encoder, decoder, classifier, train_dataset, config['epochs'], optimizer, device, params, config)

