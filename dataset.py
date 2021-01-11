"""
 Contains all the utilities to load the MNIST dataset
"""
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import transforms

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


def get_mnist_overfit_dataloaders(root='./datasets/', batch_size=64, digits_to_include: list = None):
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
        mnist_train_dataset = Subset(mnist_train_dataset, train_indices)
        mnist_train_dataset = Subset(mnist_train_dataset, range(5))

        mnist_test_dataset = Subset(mnist_test_dataset, test_indices)

    train_dataloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(mnist_test_dataset, shuffle=True, batch_size=batch_size, )

    return train_dataloader, test_dataloader


def get_fmnist_dataloaders(root='./datasets/', batch_size=64, digits_to_include: list = None):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param digits_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    if digits_to_include == None:
        fmnist_train_dataset = FashionMNIST(root=root, download=True, train=True, transform=transform)
        fmnist_test_dataset = FashionMNIST(root=root, download=True, train=False, transform=transform)

    else:
        digit_filter_function = lambda x: digits_to_include.index(x) if x in digits_to_include else -1

        fmnist_train_dataset = FashionMNIST(root=root, download=True, train=True, transform=transform,
                                    target_transform=digit_filter_function)
        fmnist_test_dataset = FashionMNIST(root=root, download=True, train=False, transform=transform,
                                   target_transform=digit_filter_function)
        train_indices = get_indices(fmnist_train_dataset)
        test_indices = get_indices(fmnist_test_dataset)
        fmnist_train_dataset = Subset(fmnist_train_dataset, train_indices, )
        fmnist_test_dataset = Subset(fmnist_test_dataset, test_indices)

    train_dataloader = DataLoader(fmnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(fmnist_test_dataset, shuffle=True, batch_size=batch_size, )

    return train_dataloader, test_dataloader

def get_cifar10_dataloaders(root='./datasets/', batch_size=64, digits_to_include: list = None):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param digits_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    if digits_to_include == None:
        train_dataset = CIFAR10(root=root, download=True, train=True, transform=transform)
        test_dataset = CIFAR10(root=root, download=True, train=False, transform=transform)

    else:
        digit_filter_function = lambda x: digits_to_include.index(x) if x in digits_to_include else -1

        train_dataset = CIFAR10(root=root, download=True, train=True, transform=transform,
                                    target_transform=digit_filter_function)
        test_dataset = CIFAR10(root=root, download=True, train=False, transform=transform,
                                   target_transform=digit_filter_function)
        train_indices = get_indices(train_dataset)
        test_indices = get_indices(test_dataset)
        train_dataset = Subset(train_dataset, train_indices, )
        test_dataset = Subset(test_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, )

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
