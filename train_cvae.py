from datetime import time

import torch
import numpy as np

import MNIST
from MNIST_CNN_model import MNIST_CNN
from MNIST_cvae_model import Encoder, Decoder


def train_cvae(encoder, decoder, classifier, dataloader, n_epochs, optimizer, device, params, use_causal_effect=False,
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
            #print(nll)

            #causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)
            #print(causalEffect)
            #loss = use_causal_effect * causalEffect + lam_ML * nll
            loss = nll
            loss.backward()
            optimizer.step()

            print(loss.item())
        print(i)



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
    zs = np.zeros((params['alpha_samples'] * params['beta_samples'], params['z_dim']))
    for i in range(0, params['alpha_samples']):
        alpha = np.random.randn(params['n_alpha'])
        zs = np.zeros((params['beta_samples'], params['z_dim']))
        for j in range(0, params['beta_samples']):
            beta = np.random.randn(params['n_beta'])
            zs[j, :params['n_alpha']] = alpha
            zs[j, params['n_alpha']:] = beta
        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[0]
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
    n_beta = 2
    n_alpha = 1
    z_dim = n_alpha + n_beta
    x_dim = 28 * 28

    z_dim = n_alpha + n_beta

    encoder = Encoder(z_dim, 1, x_dim).to(device)
    decoder = Decoder(z_dim, 1, x_dim).to(device)

    PATH = './pretrained_models_local/mnist_cnn'

    classifier = load_pretrained_mnist(MNIST_CNN, PATH, 2).to(device)

    train_dataset, test_dataset = MNIST.get_mnist_dataloaders(batch_size=64, digits_to_include=[3, 8])
    n_epochs = 10

    params_use = list(decoder.parameters()) + list(encoder.parameters())
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999
    optimizer = torch.optim.Adam(params_use, lr=lr, betas=(b1, b2))

    params = {
        "number_of_classes": 2,
        "alpha_samples": 64,
        "beta_samples": 64,
        "z_dim": z_dim,
        "n_alpha": n_alpha,
        "n_beta": n_beta
    }

    train_cvae(encoder, decoder, classifier, train_dataset, n_epochs, optimizer, device, params)
