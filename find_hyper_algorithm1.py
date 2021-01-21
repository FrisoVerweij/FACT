import torch
import time

import yaml
import argparse
import os

from utils import *

def algorithm1(classifier, train_data, test_data, config):
    # step 1
    plateau = False
    K, L, lamb = 0, 0, 0.0  # K = alpha L = beta
    nll_best = 10000.0  # initialize the value to a very high value
    start_time = time.time()
    optimize_on = 'nll'
    while (not plateau):
        L += 1
        _, _, average_nll, _, _ = \
            train_and_test(optimize_on, K ,L ,lamb, classifier, train_data, test_data, config)

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
    k_prev, l_prev, lamb_prev = 0, 0, 0.0  # after we plateau we take the values just before
    while(not plateau):
        K_prev, L_prev, lamb_prev = K, L, lamb
        K += 1
        L -= 1
        lamb = 0.0 # not sure if lambda resets after every iteration
        approaches = False

        # step 3
        while(not approaches):
            lamb += 0.01
            _, _, average_nll, average_causal, _ = \
                train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, config)

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
          (K_prev, L_prev, lamb_prev, nll_best, causal_best, time.time() - start_time))

def train_and_test(optimize_on, K, L, lamb, classifier, train_data, test_data, config):
    z_dim = K + L
    config['z_dim'], config['n_alpha'], config['n_beta'] = z_dim, K, L

    encoder, decoder = select_vae_model(config)
    encoder, decoder = encoder.to(config['device']), decoder.to(config['device'])
    optimizer = select_optimizer(config, encoder, decoder)

    encoder_trained, decoder_trained = train(optimize_on, encoder, decoder,optimizer, classifier, train_data,lamb, config)

    average_nll, average_causal_effect, average_total = test(encoder_trained, decoder_trained, classifier, test_data, lamb, config)

    return encoder_trained, decoder_trained, average_nll, average_causal_effect, average_total

def train(optimize_on, encoder, decoder, optimizer, classifier, train_dataset, lamb, config):

    for epoch in range(config['epochs']):  # n_epochs determines how long we will train the model before we finish
        for imgs, targets in train_dataset:
            imgs, targets = imgs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

            if optimize_on == 'nll':
                loss = nll

            elif optimize_on == 'causal':
                causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
                loss = causalEffect

            elif optimize_on == 'both':
                causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
                loss = config['use_causal'] * causalEffect + lamb * nll

            loss.backward()
            optimizer.step()

    return encoder, decoder

def test(encoder, decoder, classifier, test_data, lamb, config):
    # Test the model based on its average test loss
    encoder.eval(); decoder.eval()
    total_nll = 0; total_causal_effect = 0; total_loss = 0
    batch = 0
    with torch.no_grad():
        for imgs, targets in test_data:
            imgs, targets = imgs.to(config['device']), targets.to(config['device'])

            latent_out, mu, logvar = encoder(imgs)
            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)
            causalEffect, ceDebug = joint_uncond(config, decoder, classifier, config['device'])
            loss = config['use_causal'] * causalEffect + lamb * nll

            total_nll += nll
            total_causal_effect += causalEffect
            total_loss += loss

            batch += 1

        average_nll = total_nll / (batch + 1)
        average_causal_effect = total_causal_effect / (batch + 1)
        average_total = total_loss / (batch + 1)

    return average_nll, average_causal_effect, average_total


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/mnist_3_8_final.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_vae_config(config)
    config = prepare_variables_pl(config)

    # Set seeds to make test reproducible
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # The device to run the model on
    device = config['device']

    # The decoder and encoder to use
    encoder, decoder = select_vae_model(config)
    encoder, decoder = encoder.to(device), decoder.to(device)

    # The classifier to use
    classifier = select_classifier(config)
    classifier.load_state_dict(torch.load(config['save_dir'] + config['classifier'] + "_" + config['model_name']))
    classifier.to(device)

    # The dataset is loaded
    train_dataset, test_dataset = select_dataloader(config)

    # The optimizer is loaded
    optimizer = select_optimizer(config, encoder, decoder)

    algorithm1(classifier, train_dataset, test_dataset, config)


