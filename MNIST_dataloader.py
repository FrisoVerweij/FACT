"""
 Contains all the utilities to load the MNIST dataset
"""
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

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
            # mnist_digits = config['mnist_digits'].split(',')
            # mnist_digits = [int(digit) for digit in mnist_digits]
            mnist_digits = config['mnist_digits']

        # return dataloader
        return get_mnist_dataloaders(batch_size=config['batch_size'], digits_to_include=mnist_digits)
    else:
        raise Exception("No valid dataset selected!")


def get_mnist_dataloaders(root='./datasets/', batch_size=64, digits_to_include: list = None):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param digits_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    if digits_to_include == None:
        mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform)
        mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform)

    else:
        digit_filter_function = lambda x: digits_to_include.index(x) if x in digits_to_include else -1

        mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform,
                                    target_transform=digit_filter_function)
        mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform,
                                   target_transform=digit_filter_function)
        train_indices = get_indices(mnist_train_dataset)
        test_indices = get_indices(mnist_test_dataset)
        mnist_train_dataset = Subset(mnist_train_dataset, train_indices, )
        mnist_test_dataset = Subset(mnist_test_dataset, test_indices)

    train_dataloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(mnist_test_dataset, shuffle=True, batch_size=batch_size, )

    return train_dataloader, test_dataloader


def get_indices(dataset):
    '''
    Returns indices of datapoint that have a non-negative label.
    :param dataset: dataset to get indices from
    :return: list: indices
    '''
    indices = []
    for i, (x, y) in enumerate(dataset):
        if y != -1:
            indices.append(i)

    return indices
