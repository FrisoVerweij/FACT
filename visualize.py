from __future__ import division
import matplotlib.pyplot as plt

import yaml
from matplotlib import image

from utils import *


def visualize_from_config(config):
    # --- initialize ---
    z_dim = config['n_alpha'] + config['n_beta']
    device = config['device']
    image_size = config['image_size']
    # --- load test data ---
    traindata_loader, testdata_loader = select_dataloader(config)
    data, targets = next(iter(testdata_loader))
    y_val = targets.numpy()
    x_val = np.reshape(data, (data.shape[0], data.shape[2], data.shape[3], data.shape[1]))
    # --- load VAE ---
    encoder, decoder = select_vae_model(config, z_dim)
    encoder, decoder = encoder.to(device), decoder.to(device)
    encoder.load_state_dict(torch.load(config['save_dir'] + config['vae_model'] + "_encoder", map_location=device),
                            strict=False)
    decoder.load_state_dict(torch.load(config['save_dir'] + config['vae_model'] + "_decoder", map_location=device),
                            strict=False)

    # --- load classifier ---
    classifier = select_classifier(config).to(device)
    classifier.load_state_dict(torch.load(config['save_dir'] + config["classifier"], map_location=device), strict=False)

    # %% generate latent factor sweep plot
    latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
    visualize(y_val, z_dim, x_val, device, encoder, decoder, classifier, latentsweep_vals, image_size)
    print('Columns - latent values in sweep: ' + str(latentsweep_vals))


def visualize(y_val, z_dim, x_val, device, encoder, decoder, classifier, latentsweep_vals, image_size):
    ## Get the sample indexes.
    sample_ind = get_sample_indices(y_val, [0, 1], [4, 4])

    cols = [[0.047, 0.482, 0.863], [1.000, 0.761, 0.039], [0.561, 0.788, 0.227]]
    border_size = 3
    nsamples = len(sample_ind)

    Xhats, Yhats = generate_samples(z_dim, nsamples, latentsweep_vals, x_val, sample_ind, device, encoder, decoder,
                                    classifier, image_size)

    imgs = create_figures(z_dim, nsamples, latentsweep_vals, Xhats, Yhats, cols, border_size, image_size)
    return imgs


def generate_samples(z_dim, nsamples, latentsweep_vals, x_val, sample_ind, device, encoder, decoder, classifier,
                     image_size):
    Xhats = np.zeros((z_dim, nsamples, len(latentsweep_vals), image_size, image_size, 1))
    yhats = np.zeros((z_dim, nsamples, len(latentsweep_vals)))
    for isamp in range(nsamples):
        x = torch.from_numpy(np.expand_dims(x_val[sample_ind[isamp]], 0))
        x_torch = x.permute(0, 3, 1, 2).float().to(device)
        z = encoder(x_torch)[0][0].detach().cpu().numpy()
        for latent_dim in range(z_dim):
            for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
                ztilde = z.copy()
                ztilde[latent_dim] += latentsweep_val
                xhat = decoder(torch.unsqueeze(torch.from_numpy(ztilde), 0).to(device))
                yhat = np.argmax(classifier(xhat)[0].detach().cpu().numpy())
                img = 1. - xhat.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()

                Xhats[latent_dim, isamp, ilatentsweep, :, :, 0] = img
                yhats[latent_dim, isamp, ilatentsweep] = yhat
    return Xhats, yhats


def create_figures(z_dim, nsamples, latentsweep_vals, Xhats, Yhats, cols, border_size, image_size):
    imgs = []
    for latent_dim in range(z_dim):
        fig, axs = plt.subplots(nsamples, len(latentsweep_vals))
        for isamp in range(nsamples):
            for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
                img = Xhats[latent_dim, isamp, ilatentsweep, :, :, 0].squeeze()

                yhat = int(Yhats[latent_dim, isamp, ilatentsweep])
                img_bordered = np.tile(np.expand_dims(np.array(cols[yhat]), (0, 1)),
                                       (image_size + 2 * border_size, image_size + 2 * border_size,
                                        1))
                img_bordered[border_size:-border_size, border_size:-border_size, :] = \
                    np.tile(np.expand_dims(img, 2), (1, 1, 3))

                axs[isamp, ilatentsweep].imshow(img_bordered, interpolation='nearest')
                axs[isamp, ilatentsweep].axis('off')
        axs[0, round(len(latentsweep_vals) / 2) - 1].set_title('Sweep latent dimension %d' % (latent_dim + 1))

        print('Exporting latent dimension %d...' % (latent_dim + 1))
        name = './figures/fig_mnist_qual_latentdim%d.png' % (latent_dim + 1)
        plt.savefig(name, bbox_inches=0)
        img = image.imread(name)
        imgs.append(img)

    return imgs


def get_sample_indices(y_val, labels, n_labels):
    sample_indices_list = [
        np.where(y_val == label)[0][:n_label] for label, n_label in zip(labels, n_labels)
    ]
    sample_ind = np.concatenate(sample_indices_list)
    return sample_ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/mnist_3_8.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"))
    config = to_visualize_config(config)
    visualize_from_config(config)
