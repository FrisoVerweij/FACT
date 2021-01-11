import numpy as np
import os
import torch
import util
from load_mnist import load_mnist_classSelect, load_fashion_mnist_classSelect, load_svhn_classSelect
import torchvision.transforms as transforms
import pickle
from torch.autograd import Variable
import time
import loss_functions
import causaleffect
import re
import datetime
import scipy.io as sio
from torchvision.transforms import transforms
#import gif

import models.VAE as VAE
import models.CVAE as CVAE
import models.CNN_classifier as CNN_model
from MNIST import get_mnist_dataloaders
# from imagenet_zebra_gorilla_dataloader import Imagenet_Gor_Zeb # no single mention about this in their paper

def train_CVAE(params):
    if params["debug_level"] > 0:
        print("parameters")
        print(params)

    debug = {}
    debug["loss"] = np.zeros((params['steps']))
    debug["loss_ce"] = np.zeros((params['steps']))
    debug["loss_nll"] = np.zeros((params['steps']))
    debug["loss_nll_logdet"] = np.zeros((params['steps']))
    debug["loss_nll_quadform"] = np.zeros((params['steps']))
    debug["loss_nll_mse"] = np.zeros((params['steps']))
    debug["loss_nll_kld"] = np.zeros((params['steps']))

    if params['save_plot']: frames = []
    else: frames=None

    if params['randseed'] is not None:
        if params["debug_level"] > 0:
            print('Setting random seed to ' + str(params['randseed']))
        np.random.seed(params['randseed'])
        torch.manual_seed(params['randseed'])

    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #class_use, class_use_str, newClass, test_size, X, Y, tridx, vaX, vaY, vaidx, sample_inputs_torch, ntrain = data_setup(params['data_type'], device)

    train_dataloader, test_dataloader = data_setup(params['data_type'], device)

    encoder, decoder, optimizer = decoder_encoder_setup(params, device)

    classifier = classifier_setup(params, device)

    train(params, optimizer, device, iter(train_dataloader), encoder, decoder, classifier, debug, save_dir, frames)




def train(params, optimizer, device, train_dataloader, encoder, decoder, classifier, debug, save_dir, frames):
    # --- train ---
    start_time = time.time()
    for k in range(0, params['steps']):
        # --- reset gradients to zero ---
        # (you always need to do this in pytorch or the gradients are
        # accumulated from one batch to the next)
        optimizer.zero_grad()

        # --- compute negative log likelihood ---
        # randomly subsample batch_size samples of x
        #randIdx = np.random.randint(0, params['ntrain'], params['batch_size'])

        # We obtain the batch. For the linear gaussian and imagenet this lookd quite different
        #Xbatch = torch.from_numpy(X[randIdx]).float()
        #Xbatch = Xbatch.permute(0, 3, 1, 2)
        Xbatch, target = train_dataloader.next()

        # this is the batch that we will be working with
        Xbatch = Xbatch.to(device)

        # The general nll loss. Also looks different for different encoders
        latent_out, mu, logvar = encoder(Xbatch)
        Xest = decoder(latent_out)
        nll, nll_mse, nll_kld = loss_functions.VAE_LL_loss(Xbatch, Xest, logvar, mu)

        # mutual information loss. A bunch og methods are missing from causaleffect.py but we do not need them
        causalEffect, ceDebug = causaleffect.joint_uncond(params, decoder, classifier, device)

        # The loss is obtained by applying the formula in the paper
        loss = params['use_ce'] * causalEffect + params['lam_ML'] * nll

        # Backward step to compute the gradients
        loss.backward()

        optimizer.step()

        # The debug is normally also a jungle of if else statements
        debug["loss"][k] = loss.item()
        debug["loss_ce"][k] = causalEffect.item()
        debug["loss_nll"][k] = (params['lam_ML'] * nll).item()
        debug["loss_nll_mse"][k] = (params['lam_ML'] * nll_mse).item()
        debug["loss_nll_kld"][k] = (params['lam_ML'] * nll_kld).item()

        # Not exactly sure what they print here
        if params['debug_level'] > 0:
            print("[Step %d/%d] time: %4.2f  [CE: %g] [ML: %g] [loss: %g]" % \
                  (k, params['steps'], time.time() - start_time, debug["loss_ce"][k],
                   debug["loss_nll"][k], debug["loss"][k]))

        # The plotting function does not yet work here
        if params['debug_plot'] and k % 1000 == 0:
            # their plotting functions is hella weird
            #plot()
            a = 2

    # Not sure why they only save the last batch here, that is super weird
    debug["X"] = Xbatch.detach().cpu().numpy()
    debug["Xest"] = Xest.detach().cpu().numpy()

    # Now we only have to save the debug data
    if params['save_output']:
        datestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:10]))
        timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[11:19]))
        results_folder = save_dir
        matfilename = 'results_' + datestamp + '_' + timestamp + '.mat'
        sio.savemat(results_folder + matfilename, {'params': params, 'data': debug})
        if params['debug_level'] > 0:
            print('Finished saving data to ' + matfilename)

    # Now we also want to save the plot but that is still super weird
    if params['save_plot']:
        print('Saving plot...')
        # gif is not metioned in the dependencies nor is it a basic python library
        # the framses variable also only occurs here w.r.t. the training
        #gif.save(frames, "results.gif", duration=100)
        print('Done!')

# This plot function is horrible
"""def plot(params):
    # --- debug plot ---
    print('Generating plot frame...')
    if params['data_type'] == '2dpts':
        # generate samples of p(x | alpha_i = alphahat_i)
        decoded_points = {}
        decoded_points["ai_vals"] = lfplot_aihat_vals # this variable is untracable, don't know wtf they did
        decoded_points["samples"] = np.zeros((2, lfplot_nsamp, len(lfplot_aihat_vals), params["z_dim"]))
        for l in range(params["z_dim"]):  # loop over latent dimensions
            for i, aihat in enumerate(lfplot_aihat_vals):  # loop over fixed aihat values
                for m in range(lfplot_nsamp):  # loop over samples to generate
                    z = np.random.randn(params["z_dim"])
                    z[l] = aihat
                    x = decoder(torch.from_numpy(z).float(), What, gamma)
                    decoded_points["samples"][:, m, i, l] = x.detach().numpy()
        frame = plotting.debugPlot_frame(X, ceDebug["Xhat"], W, What, k,
                                         steps, debug, params, classifier,
                                         decoded_points)
        if save_plot:
            frames.append(frame)
    elif data_type == 'mnist' or data_type == 'imagenet' or data_type == 'fmnist':
        torch.save({
            'step': k,
            'model_state_dict_classifier': classifier.state_dict(),
            'model_state_dict_encoder': encoder.state_dict(),
            'model_state_dict_decoder': decoder.state_dict(),
            'optimizer_state_dict': optimizer_NN.state_dict(),
            'loss': loss,
        }, save_dir + 'network_batch' + str(batch_size) + '.pt')
        sample_latent, mu, var = encoder(sample_inputs_torch)
        sample_inputs_torch_new = sample_inputs_torch.permute(0, 2, 3, 1)
        sample_inputs_np = sample_inputs_torch_new.detach().cpu().numpy()
        sample_img = decoder(sample_latent)
        sample_latent_small = sample_latent[0:10, :]
        imgOut_real, probOut_real, latentOut_real = sweepLatentFactors(sample_latent_small, decoder, classifier,
                                                                       device, img_size, c_dim, y_dim, False)
        rand_latent = torch.from_numpy(np.random.randn(10, z_dim)).float().to(device)
        imgOut_rand, probOut_rand, latentOut_rand = sweepLatentFactors(rand_latent, decoder, classifier, device,
                                                                       img_size, c_dim, y_dim, False)
        samples = sample_img
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.detach().cpu().numpy()
        save_images(samples, [8, 8],
                    '{}train_{:04d}.png'.format(save_dir, k))
        # sio.savemat(save_dir + 'sweepLatentFactors.mat',{'imgOut_real':imgOut_real,'probOut_real':probOut_real,'latentOut_real':latentOut_real,'loss_total':debug["loss"][:k],'loss_ce':debug["loss_ce"][:k],'loss_nll':debug['loss_nll'][:k],'samples_out':samples,'sample_inputs':sample_inputs_np})
        sio.savemat(save_dir + 'sweepLatentFactors.mat',
                    {'imgOut_real': imgOut_real, 'probOut_real': probOut_real, 'latentOut_real': latentOut_real,
                     'imgOut_rand': imgOut_rand, 'probOut_rand': probOut_rand, 'latentOut_rand': latentOut_rand,
                     'loss_total': debug["loss"][:k], 'loss_ce': debug["loss_ce"][:k],
                     'loss_nll': debug['loss_nll'][:k], 'samples_out': samples, 'sample_inputs': sample_inputs_np})"""

# This method returns the classifier
def classifier_setup(params, device):

    classifier = CNN_model.CNN(params['y_dim']).to(device)
    batch_orig = 64

    # Here we left out the 'oneHyperplane' and 'twoHyperplane' classifiers
    if params['classifier_net'] == 'cnn':
        checkpoint = torch.load('./author_code/pretrained_models/mnist_38_classifier/model.pt')

    elif params['classifier_net'] == 'cnn_fmnist':
        checkpoint = torch.load('./author_code/fmnist_batch64_lr0.1_class034/network_batch' + str(batch_orig) + '.pt')

    classifier.load_state_dict(checkpoint['model_state_dict_classifier'])

    return classifier

# Very basic weight initialization method
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# This method returns the encoder, decoder and optimizer
def decoder_encoder_setup(params, device):
    # Choose the correct encoder and decoder
    # We ignore the linear, nonlinear gaussian and the imagenet CNN for now
    if params['decoder_net'] == 'VAE':
        encoder = VAE.Encoder(params['x_dim'], params['z_dim']).to(device)
        decoder = VAE.Decoder(params['x_dim'], params['z_dim']).to(device)

    elif params['decoder_net'] in ['VAE_CNN','VAE_fMNIST']:
        encoder = CVAE.Encoder(params['z_dim'], params['c_dim'], params['img_size']).to(device)
        decoder = CVAE.Decoder(params['z_dim'], params['c_dim'], params['img_size']).to(device)

    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)

    # Specify the optimizer
    # NOTE: we only include the decoder parameters in the optimizer because we don't want to update the classifier parameters
    params_use = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params_use, lr=params['lr'], betas=(params['b1'], params['b2']))

    return encoder, decoder, optimizer

# This method returns all the things that we need for the dataset
def data_setup(data_set, device):
    class_use = np.array([3, 8])
    class_use_str = np.array2string(class_use)
    y_dim = class_use.shape[0]
    newClass = range(0, y_dim)
    test_size = 64

    # 3 datasets that interesting right now, the 'imagenet' and '2dpts' ones are better to implement separetely
    if data_set == 'mnist':
        #X, Y, tridx = load_mnist_classSelect('train', class_use, newClass)
        #vaX, vaY, vaidx = load_mnist_classSelect('val', class_use, newClass) # they do not seem to use this
        train_dataloader, test_dataloader = get_mnist_dataloaders(batch_size=params['batch_size'])

    elif data_set == 'fmnist':
        X, Y, tridx = load_fashion_mnist_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', class_use, newClass)

    elif data_set == 'svhn':

        X, Y, tridx = load_svhn_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_svhn_classSelect('val', class_use, newClass)

    """sample_inputs = vaX[0:test_size]
    sample_inputs_torch = torch.from_numpy(sample_inputs)
    sample_inputs_torch = sample_inputs_torch.permute(0, 3, 1, 2).float().to(device)
    ntrain = X.shape[0]"""

    return train_dataloader, test_dataloader

    #return class_use, class_use_str, newClass, test_size, X, Y, tridx, vaX, vaY, vaidx, sample_inputs_torch, ntrain


if __name__ == "__main__":
    print("started training")

    params = {
        "steps": 8000,     # number of training step
        "batch_size": 32,   # batch size
        "z_dim": 2,     # latent variable dimension
        "z_dim_true": 2,
        "x_dim": 2,     # number of pixels
        "c_dim": 1,     # convolution dimension
        "y_dim": 2,     # y dimension
        "alpha_dim": 4,
        "lam_ML":0.001,     # determines ratio between causality and likelihood
        "gamma": 0.001,     #
        "lr": 0.0001,       # Adam learning rate
        "b1": 0.5,   # Adam bias 1
        "b2": 0.999,   # Adam bias 2
        "ntrain": 5000,
        "No": 15,
        "Ni": 15,
        "use_ce": True,   # use causal effect
        "objective": "IND_UNCOND",
        "decoder_net": "VAE_CNN",   # options are ['linGauss','nonLinGauss','VAE','VAE_CNN','VAE_Imagenet','VAE_fMNIST']
        "classifier_net": "cnn",    # options are ['oneHyperplane','twoHyperplane','cnn','cnn_imagenet','cnn_fmnist']
        "data_type" : "mnist",      # options are ["2dpts","mnist","imagenet","fmnist"]
        "break_up_ce": True,
        "randseed": 43,
        "save_output": False,
        "debug_level": 2,
        "img_size": 28,
        "data_std": 2,
        "save_plot": False
    }

    trail_results = train_CVAE(params)

    # don't know if we actually need this loop and params
    """lambda_change = [0.001]
    obj_change = ["JOINT_UNCOND"]
    alpha_change = [0]
    z_dim_change = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for obj_use in obj_change:
        for z_use in z_dim_change:
            for lam_use in lambda_change:
                for alpha_use in alpha_change:
                    # obk_use
                    # z_use
                    # lam_use
                    # alpha_use

                    trail_results = CVAE(params)"""