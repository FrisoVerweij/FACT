import torch
import yaml
from utils import *
import time
import numpy as np
from torchvision.utils import make_grid, save_image
import argparse

def train_cvae(config, dataloader, encoder, decoder, classifier, optimizer):

    # --- train ---
    start_time = time.time()
    for epoch in range(config['epochs']):
        batch_count = 0
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()

            latent_out, mu, logvar = encoder(imgs)

            x_generated = decoder(latent_out)

            nll, nll_mse, nll_kld = VAE_LL_loss(imgs, x_generated, logvar, mu)

            causalEffect, ceDebug = joint_uncond(config, decoder, classifier, device)

            loss = config['use_causal'] * causalEffect + config['lam_ml'] * nll

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

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')

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

    encoder, decoder = train_cvae(config, train_dataset, encoder, decoder, classifier, optimizer)



