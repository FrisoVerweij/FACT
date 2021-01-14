import torch
import yaml
from utils import *
import time
import numpy as np


def train_cvae(encoder, decoder, classifier, dataloader, n_epochs, optimizer, device, params, use_causal_effect,
               lam_ML):
    # --- train ---
    start_time = time.time()
    for epoch in range(n_epochs):
        batch_count = 0
        for x, y in dataloader:
            inputs = x.to(device)
            targets = y.to(device)
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(inputs)

            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(inputs, x_generated, logvar, mu)

            causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)

            loss = use_causal_effect * causalEffect + lam_ML * nll

            # this is for CIFAR10, just ignore it
            # loss = kl_divergence_loss(mu, logvar) + reconstruction_loss(x_generated, inputs)
            # loss = nll

            loss.backward()
            optimizer.step()

        print("[Train Epoch %d/%d] [Batch %d] time: %4.4f [loss: %f]" % (
            epoch, config['epochs'], batch_count, time.time() - start_time,
            loss.item()))

        grid = make_grid(x_generated, nrow=int(8), normalize=True, range=(0, 1))
        save_image(grid, './example_image.png')

    torch.save(encoder.state_dict(),
               str(config['save_dir']) + str(config['vae_model']) + "_encoder" + "_" + config['model_name'])
    torch.save(decoder.state_dict(),
               str(config['save_dir']) + str(config['vae_model']) + "_decoder" + "_" + config['model_name'])

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
    classifier.eval()
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


def algorithm1(classifier, train_data, test_data, epochs, device, params, use_causal, config):
    # step 1
    plateau = False
    K, L, lamb = 0, 0, 0.0  # K = alpha L = beta
    nll_best = 10000.0  # initialize the value to a very high value
    start_time = time.time()
    optimize_on = 'nll'
    while (not plateau):
        L += 1
        _, _, average_nll, _, _ = \
            train_and_test(optimize_on, K,L,lamb, classifier, train_data, test_data, epochs, device, params, use_causal, config)

        if average_nll < nll_best:
            nll_best = average_nll
        else:
            plateau = True

        print("step 1, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average nll: %f] time: %4.4f" %
              (K,L,lamb, average_nll, time.time() - start_time))

    print("step 1, finished, [best K: %d] [best L: %d] [best lamb: %f] [best loss: %f] time: %4.4f" %
          (K, L, lamb, nll_best, time.time() - start_time))

    # step 2 and 3
    plateau = False
    causal_best = 10000.0
    start_time = time.time()
    optimize_on = 'both'
    while(not plateau):
        K += 1
        L -= 1
        lamb = 0.0 # not sure if lambda resets after every iteration
        approaches = False

        # step 3
        while(not approaches):
            lamb += 0.01
            _, _, average_nll, average_causal, _ = \
                train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, epochs, device, params, use_causal, config)

            # note that here lambda is not incremented until the best one is found but until it reaches the desired
            # accuracy
            if average_nll < nll_best:
                approaches = True

            print("step 2, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average nll: %f] time: %4.4f" %
                  (K, L, lamb, average_nll, time.time() - start_time))

        if average_causal < causal_best:
            causal_best = average_causal
        else:
            plateau = True

        print("step 3, finished iteration, [best K: %d] [best L: %d] [best lamb: %f] [average causal: %f] time: %4.4f" %
              (K, L, lamb, average_causal, time.time() - start_time))

    print("finished algorithm, [best K: %d] [best L: %d] [best lamb: %f] [best nll: %f] [best causal: %f] time: %4.4f" %
          (K, L, lamb, nll_best, causal_best, time.time() - start_time))

def train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, epochs, device, params, use_causal, config):
    z_dim = K + L

    encoder, decoder = select_vae_model(config, z_dim)
    encoder, decoder = encoder.to(device), decoder.to(device)
    optimizer = select_optimizer(config, encoder, decoder)

    params['z_dim'], params['n_alpha'], params['n_beta'] = z_dim, K, L

    encoder_trained, decoder_trained = train(optimize_on, encoder, decoder,optimizer, classifier, train_data, epochs,
                                        device, params, use_causal, lamb)

    average_nll, average_causal_effect, average_total = test(encoder_trained, decoder_trained, classifier, test_data,
                                        device, params, use_causal, lamb)

    return encoder_trained, decoder_trained, average_nll, average_causal_effect, average_total

def train(optimize_on, encoder, decoder, optimizer, classifier, train_dataset, epochs, device, params, use_causal, lamb):

    for epoch in range(epochs):  # n_epochs determines how long we will train the model before we finish
        for imgs, targets in train_dataset:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

            if optimize_on == 'nll':
                loss = nll

            elif optimize_on == 'causal':
                causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)
                loss = causalEffect

            elif optimize_on == 'both':
                causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)
                loss = use_causal * causalEffect + lamb * nll

            loss.backward()
            optimizer.step()

    return encoder, decoder

def test(encoder, decoder,classifier, test_data, device, params, use_causal, lamb):
    # Test the model based on its average test loss
    encoder.eval(); decoder.eval()
    total_nll = 0; total_causal_effect = 0; total_loss = 0
    batch = 0
    with torch.no_grad():
        for imgs, targets in test_data:
            imgs, targets = imgs.to(device), targets.to(device)

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)
            causalEffect, ceDebug = joint_uncond(params, decoder, classifier, device)
            loss = use_causal * causalEffect + lamb * nll

            total_nll += nll
            total_causal_effect += causalEffect
            total_loss += loss

            batch += 1

        average_nll = total_nll / (batch + 1)
        average_causal_effect = total_causal_effect / (batch + 1)
        average_total = total_loss / (batch + 1)

    return average_nll, average_causal_effect, average_total


if __name__ == "__main__":
    # Set seeds to make test reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse the arguments
    parser = argparse.ArgumentParser()

    #parser.add_argument('--config', default='config/mnist_3_8.yml')
    #parser.add_argument('--config', default='config/cifar10_sasha.yml')
    parser.add_argument('--config', default='config/mnist_3_8_algorithm.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)

    # The device to run the model on
    device = config['device']

    # The latent variable dimensions z_dim
    z_dim = config['n_alpha'] + config['n_beta']

    # The decoder and encoder to use
    encoder, decoder = select_vae_model(config, z_dim)
    encoder, decoder = encoder.to(device), decoder.to(device)

    # The classifier to use
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier'] + "_" + config['model_name']))
    classifier.to(device)

    # The dataset is loaded
    train_dataset, test_dataset = select_dataloader(config)

    # The optimizer is loaded
    optimizer = select_optimizer(config, encoder, decoder)

    # Additional parameters that are mainly used for the cuasal term of the loss
    if config['mnist_digits'] == None:
        n_classes = 10
    else:
        n_classes = len(config['mnist_digits'])
    params = {
        "number_of_classes": n_classes,
        "alpha_samples": config['alpha_samples'],
        "beta_samples": config['beta_samples'],
        "z_dim": z_dim,
        "n_alpha": config['n_alpha'],
        "n_beta": config['n_beta']
    }

    if not config['find_params']:
        encoder, decoder = train_cvae(encoder, decoder, classifier, train_dataset, config['epochs'], optimizer,
                                     device, params, config['use_causal'], config['lam_ml'])
    else:
        algorithm1(classifier, train_dataset, test_dataset, config['epochs'], device, params, config['use_causal'], config)


