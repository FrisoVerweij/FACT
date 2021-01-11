from datetime import time
import argparse

import torch
import numpy as np

import MNIST
from MNIST_CNN_model import MNIST_CNN
import MNIST_cvae_model
from utils import *

def train(flags, seed=1):
    # Set seeds to make test reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = flags.device

    # Some necessary hyperparameters
    n_beta = flags.n_beta  # 2
    n_alpha = flags.n_alpha  # 1
    z_dim = n_alpha + n_beta
    x_dim = flags.x_dim  # 28x28

    if flags.mnist_digits == "None":
        n_classes = 10
    else:
        n_classes = len(flags.mnist_digits.split(','))

    params = {
        "number_of_classes": n_classes,
        "alpha_samples": flags.alpha_samples,
        "beta_samples": flags.beta_samples,
        "z_dim": z_dim,
        "n_alpha": n_alpha,
        "n_beta": n_beta
    }

    # Prepare the generative model
    encoder, decoder = encoder_decoder(flags, z_dim)
    encoder.apply(weights_init_normal) # the same weight initialization they authors use in their code but does not seem to have the desired effec
    decoder.apply(weights_init_normal)
    encoder, decoder = encoder.to(device), decoder.to(device)

    # Prepare the classifier
    classifier = select_model(flags).to(device)
    classifier.load_state_dict(torch.load(str(flags.save_dir) + str(flags.model)), strict=False)

    # Prepare the dataset
    train_dataset, test_dataset = select_dataloader(flags)

    n_epochs = flags.epochs

    b1 = 0.5  # lets not make these hyperparameters as they are the default values (I think)
    b2 = 0.999
    optimizer = select_optimizer(flags, encoder, decoder, b1, b2)

    train_cvae(encoder, decoder, classifier, train_dataset, n_epochs, optimizer, device, params, flags.use_causal, flags.lam_ml)

def train_cvae(encoder, decoder, classifier, dataloader, n_epochs, optimizer, device, params, use_causal_effect=True,
               lam_ML=0.001, ):

    # --- train ---
    for i in range(n_epochs):
        for x, y in dataloader:
            inputs = x.to(device)
            targets = y.to(device)
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(inputs)

            x_generated = decoder(latent_out)

            #with torch.no_grad():
            #    print(x_generated.shape)

            nll, nll_mse, nll_kld = VAE_LL_loss(inputs, x_generated, logvar, mu)

            causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)

            loss = use_causal_effect * causalEffect + lam_ML * nll
            #loss = nll

            loss.backward()
            optimizer.step()

        grid = make_grid(x_generated, nrow=int(8), normalize=True, range=(0, 1))
        save_image(grid, './image0.png')

        #image = sample(encoder, decoder, 64, device)
        #print(torch.unique(image))
        print(loss.item())
        print(i)

        torch.save(encoder.state_dict(), str(flags.save_dir) + str(flags.model) + "_encoder")
        torch.save(decoder.state_dict(), str(flags.save_dir) + str(flags.model) + "_decoder")

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

def gen_from_latent(path):
    a=2


if __name__ == "__main__":
    # Create parser to get hyperparameters from user
    parser = argparse.ArgumentParser()

    # Parse hyperparameters
    parser.add_argument('--encoder_decoder', type=str, default='cvae', choices=['cvae'],
                        help='the generative model')
    parser.add_argument('--model', type=str, default='mnist_cnn', choices=['mnist_cnn', 'fmnist_cnn', 'cifar10_cnn'],
                        help='model to use at the end')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum term')

    parser.add_argument('--save_dir', type=str, default="./pretrained_models_local/",
                        help="directory of the pretrained models")


    # New possible hyperparameters
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help="optimizer used to train the model")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help="device to run the algorithm on")

    parser.add_argument('--dataset', type=str, default="mnist", choices=['mnist', 'fmnist', 'cifar10'],
                        help="dataset to train on")
    parser.add_argument('--mnist_digits', type=str, default="3,8",
                        help="list of digits to include in the dataset. If nothing is given, all are included. "
                             "E.g. --mnist_digits=3,8 to include just 3 and 8")

    # Additional hyperparameters
    parser.add_argument('--n_beta', type=int, default=2,
                        help="The number of latent variables that we DO NOT want to correlate")
    parser.add_argument('--n_alpha', type=int, default=1,
                        help="The number of latent variables that we DO want to correlate")
    parser.add_argument('--x_dim', type=int, default=28*28,
                        help="The total number of pixels in the image")
    parser.add_argument('--alpha_samples', type=int, default=10,
                        help="The total number of samples for alpha")
    parser.add_argument('--beta_samples', type=int, default=10,
                        help="The total number of samples for beta")

    parser.add_argument('--use_causal', type=bool, default=True,
                        help="If we should use the causal effect or not")
    parser.add_argument('--lam_ml', type=float, default=0.001,
                        help="Factor that is multiplied with the nll")

    flags = parser.parse_args()

    train(flags, 1)
