import MNIST_dataloader
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import time
import MNIST_CNN_model
import yaml # pip install pyyaml


def train(config, seed):
    '''
    Given a set of config parameters, train a neural network
    :config: dicionary hyperparameters specified in the config file
    :param seed: seed for reproducibility
    :return: nn.module: model
    '''
    # Set seeds to make test reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_add(seed)

    device = torch.device(config['device'])

    # Select the dataloaders for the right dataset
    traindata_loader, testdata_loader = select_dataloader(config)

    # Select model, optimizer, scheduler and loss function
    model = select_model(config).to(device)
    optimizer = select_optimizer(config, model)
    loss_module = nn.CrossEntropyLoss()

    # Record current time to record training time
    start_time = time.time()
    batch_count = 0
    for epoch in range(config['epochs']):
        batch_count = 0
        model.train()
        for inputs, targets in traindata_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate model output and loss
            predictions, _ = model(inputs)
            loss = loss_module(predictions, targets)

            # Execute backwards propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Train Epoch %d/%d] [Batch %d] time: %4.4f [loss: %f]" % (
                epoch, config['epochs'], batch_count, time.time() - start_time,
                loss.item()))

            batch_count += 1

        # Calculate test loss
        model.eval()
        with torch.no_grad():
            total_comparisons = 0
            total_correct = 0
            for inputs, targets in testdata_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, prediction_probabilities = model(inputs)
                predictions = torch.argmax(prediction_probabilities, dim=-1)
                total_comparisons += len(targets)
                total_correct += (predictions == targets).sum().item()

            accuracy = total_correct / total_comparisons
            print("[Test Epoch %d/%d] [corr: %f]" % (epoch, config['epochs'], accuracy))

    torch.save(model.state_dict(), str(config['save_dir']) + str(config['model_name']))
    return model


def select_dataloader(config):
    '''
    Selects a dataloader given the hyperparameters
    :config: Set of hyperparameters
    :return: DataLoader
    '''
    if config['dataset'] == "mnist":
        # Parse the list of digits to include
        if config['mnist_digits'] == "None":
            mnist_digits = None
        else:
            mnist_digits = config['mnist_digits'].split(',')
            mnist_digits = [int(digit) for digit in mnist_digits]

        # return dataloader
        return MNIST_dataloader.get_mnist_dataloaders(batch_size=config['batch_size'], digits_to_include=mnist_digits)
    else:
        raise Exception("No valid dataset selected!")


def select_model(config):
    '''
    Selects a model given the hyperparameters
    :config: Set of hyperparameters
    :return: nn.module
    '''
    if config['model'] == 'mnist_cnn':
        if config['mnist_digits'] == "None":
            output_dim = 10
        else:
            output_dim = len(config['mnist_digits'].split(','))
        return MNIST_CNN_model.MNIST_CNN(output_dim)
    else:
        raise Exception("No valid model selected!")


def select_optimizer(config, model):
    '''
    Selects an optimizer given the hyperparameters
    :config: Set of hyperparameters
    :param model: model that we want to optimize
    :return: nn.optim.optimizer
    '''
    if config['optimizer'] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config['lr'])
    else:
        raise Exception("No valid optimizer selected!")


if __name__ == "__main__":
    # Create parser to get hyperparameters from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))

    train(config, 1)

    # # Parse hyperparameters
    # parser.add_argument('--model', type=str, default='mnist_cnn', choices=['mnist_cnn'],
    #                     help='model to train')
    # parser.add_argument('--epochs', type=int, default=50,
    #                     help='number of epochs of training')
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='size of the batches')
    # parser.add_argument('--lr', type=float, default=0.1,
    #                     help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.5,
    #                     help='momentum term')
    #
    # parser.add_argument('--save_dir', type=str, default="./pretrained_models_local/",
    #                     help="directory to save the model")
    #
    #
    # # New possible hyperparameters
    # parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
    #                     help="optimizer used to train the model")
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
    #                     choices=['cuda', 'cpu'],
    #                     help="device to run the algorithm on")
    # parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist"],
    #                     help="dataset to train on")
    # parser.add_argument('--mnist_digits', type=str, default="None",
    #                     help="list of digits to include in the dataset. If nothing is given, all are included. "
    #                          "E.g. --mnist_digits=3,8 to include just 3 and 8")
    #
    # train(parser.parse_args(), 1)
