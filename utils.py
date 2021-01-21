import dataset
import torch.nn as nn
import torch
from models import models_classifiers, models_vae, CNN_classifier_author
import numpy as np


def select_dataloader(config):
    '''
    Selects a dataloader given the hyperparameters
    :param config: Set of hyperparameters
    :return: DataLoader
    '''

    # Parse the list of digits to include
    if config["mnist_digits"] is None:
        mnist_digits = None
    else:
        mnist_digits = config["mnist_digits"]

    if config["dataset"] == "mnist":
        if config["classifier"] == "mnist_dummy":
            return dataset.get_mnist_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits,
                                                 dummy=True)
        else:
            return dataset.get_mnist_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "mnist_overfit":
        return dataset.get_mnist_overfit_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "fmnist":
        return dataset.get_fmnist_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    elif config["dataset"] == "cifar10":
        return dataset.get_cifar10_dataloaders(batch_size=config["batch_size"], digits_to_include=mnist_digits)

    else:
        raise Exception("No valid dataset selected!")


def select_classifier(config):
    '''
    Selects a model given the hyperparameters
    :param config: Set of hyperparameters
    :return: nn.module
    '''

    if config["mnist_digits"] is None:
        output_dim = 10
    else:
        output_dim = len(config["mnist_digits"])

    if config["classifier"] in ['mnist_cnn', 'fmnist_cnn']:
        if config["model_name"][-3:] == ".pt":
            print("load .pt")

            return CNN_classifier_author.CNN(output_dim)
        else:
            return models_classifiers.MNIST_CNN(output_dim)

    elif config["classifier"] == 'mnist_cnn_overfit':
        return models_classifiers.MNIST_CNN_Overfit(output_dim)

    elif config["classifier"] == 'cifar10_cnn':
        return models_classifiers.CIFAR10_CNN(output_dim)
    elif config["classifier"] == 'mnist_dummy':
        return models_classifiers.BiggestDummy()
    else:
        raise Exception("No valid model selected!")


def select_optimizer(config, model, model_2=None):
    '''
    Selects an optimizer given the hyperparameters
    :param config: Set of hyperparameters
    :param model: model that we want to optimize
    :return: nn.optim.optimizer
    '''
    if model_2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model_2.parameters())

    if config["optimizer"] == "SGD":
        return torch.optim.SGD(params, lr=config["lr"], momentum=config["momentum"])
    elif config["optimizer"] == "Adam":
        return torch.optim.Adam(params, lr=config["lr"], betas=(config["b1"], config["b2"]),
                                weight_decay=config["weight_decay"])
    else:
        raise Exception("No valid optimizer selected!")


def select_vae_model(config):
    if config["vae_model"] in ["mnist_cvae", "fmnist_cvae"]:
        encoder = models_vae.Encoder(config['z_dim'], 1, config["image_size"] ** 2)
        decoder = models_vae.Decoder(config['z_dim'], 1, config["image_size"] ** 2)
    elif config["vae_model"] == "cifar10_cvae":
        encoder = models_vae.Encoder_cifar10(config['z_dim'], 3, config["image_size"] ** 2)
        decoder = models_vae.Decoder_cifar10(config['z_dim'], 3, config["image_size"] ** 2)
    elif config["vae_model"] == "cifar10_cvae_sasha":
        encoder = models_vae.Encoder_cifar10_sasha(config['z_dim'], 3, config[
            "image_size"])  # note that here we do not square the image size
        decoder = models_vae.Decoder_cifar10_sasha(config['z_dim'], 3, config["image_size"])
    elif config["vae_model"] == "cifar10_cvae_captain":
        encoder = models_vae.Encoder_captain(config['z_dim'], 3, config["image_size"], config["image_size"],
                                             config["device"])
        decoder = models_vae.Decoder_captain(config['z_dim'], 3, config["image_size"], config["image_size"],
                                             config["device"])
    else:
        raise Exception("No valid encoder/decoder selected!")

    return encoder, decoder


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def prepare_variables_pl(config):
    # The device to run the model on
    z_dim = config['n_alpha'] + config['n_beta']
    x_dim = config['image_size'] ** 2

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
        "n_beta": config['n_beta'],
        "channel_dimension": 1,
        "x_dim": x_dim
    }
    base_path = config['save_dir'] + config["model_name"]
    config['encoder_path'] = base_path + '_encoder'
    config['decoder_path'] = base_path + '_decoder'

    config = {**config, **params}
    return config


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


def reconstruction_loss(x_reconstructed, x):
    return nn.BCELoss(reduction='sum')(x_reconstructed, x) / x.size(0)


def kl_divergence_loss(mean, logvar):
    return ((mean ** 2 + logvar.exp() - 1 - logvar) / 2).mean()


def to_classifier_config(config):
    return {**config['classifier'], **config['general'], **config['dataset'], }


def to_vae_config(config):
    return {**config['vae'], **config['general'], **config['dataset'], "classifier": config["classifier"]["classifier"]}


def to_visualize_config(config):
    return {**config['vae'], **config['general'], **config['dataset'], "classifier": config["classifier"]["classifier"]}


def load_classifier(config):
    # Select and load the classifier
    classifier = select_classifier(config)
    if config["classifier"] != "mnist_dummy":

        if config["model_name"][-3:] == ".pt":
            classifier.load_state_dict(
                torch.load(config['save_dir'] + config['model_name'])['model_state_dict_classifier'])
        else:
            classifier.load_state_dict(
                torch.load(config['save_dir'] + config['classifier'] + "_" + config['model_name']))
    return classifier


def get_x_vals(val_loader, n_classes=2, n_for_each_class=4):
    ###

    # Here we get a sufficient number of samples to get the images from
    data, targets = next(iter(val_loader))
    for x, y in val_loader:
        data = torch.cat([data, x], dim=0)
        targets = torch.cat([targets, y], dim=0)

        # Here we make sure that we have all the classes and that we have enough samples
        if len(torch.unique(targets)) >= n_classes:
            y_val = targets.numpy()
            indices = []
            for i in range(n_classes):
                indices += list(np.where(y_val == i)[0])[:n_for_each_class]

            if indices == n_classes * n_for_each_class:
                break

    indices = torch.tensor(indices)
    x_val = torch.index_select(data, 0, indices)

    return x_val


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_encoder_decoder(config):
    # Select the encoder, decoder and optimizer
    encoder, decoder = select_vae_model(config)

    encoder.load_state_dict(torch.load(config['encoder_path']))
    decoder.load_state_dict(torch.load(config['decoder_path']))
    return encoder, decoder