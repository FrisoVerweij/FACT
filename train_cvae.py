import torch
import yaml
from utils import *


def train_cvae(encoder, decoder, classifier, dataloader, n_epochs, optimizer, device, params, use_causal_effect, lam_ML):
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

        grid = make_grid(x_generated, nrow=int(8), normalize=True, range=(0, 1))
        save_image(grid, './image0.png')

        print(loss.item())
        print(i)

    torch.save(encoder.state_dict(), str(config['save_dir']) + str(config['vae_model']) + "_encoder")
    torch.save(decoder.state_dict(), str(config['save_dir']) + str(config['vae_model']) + "_decoder")

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
        with torch.no_grad():
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
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))

    # The device to run the model on
    device = config['device']

    # The latent variable dimensions z_dim and the number of pixels x_dim
    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size']**2

    # The decoder and encoder to use
    encoder, decoder = select_vae_model(config, z_dim)
    encoder, decoder = encoder.to(device), decoder.to(device)

    # The classifier to use
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier']))
    classifier.to(device)

    # The dataset is loaded
    train_dataset, test_dataset = select_dataloader(config)

    # The optimizer is loaded
    optimizer = select_optimizer(config, encoder, decoder)

    # Additional parameters that are mainly used for the cuasal term of the loss
    params = {
        "number_of_classes": 2,
        "alpha_samples": config['alpha_samples'],
        "beta_samples": config['beta_samples'],
        "z_dim": z_dim,
        "n_alpha": config['n_alpha'],
        "n_beta": config['n_beta']
    }

    encoder, decoder = train_cvae(encoder, decoder, classifier, train_dataset, config['epochs'], optimizer,
                                  device, params, config['use_causal'], config['lam_ml'])


