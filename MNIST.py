"""
 Contains all the utilities to load the MNIST dataset
"""
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def get_mnist_dataloaders(root='./datasets/', batch_size=64, digits_to_include:list=None):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param digits_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform)
    mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform)

    if digits_to_include != None:
        train_indices = get_indices(mnist_train_dataset, digits_to_include)
        test_indices = get_indices(mnist_test_dataset, digits_to_include)
        mnist_train_dataset = Subset(mnist_train_dataset, train_indices)
        mnist_test_dataset = Subset(mnist_test_dataset, test_indices)

    train_dataloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader =  DataLoader(mnist_test_dataset, shuffle=True, batch_size=batch_size, )

    return train_dataloader, test_dataloader


def get_indices(dataset, digits):
    '''
    Returns indices of given digits in dataset
    :param dataset: dataset to get indices from
    :param digits: list of digits to find indices of
    :return: list: indices
    '''
    indices = []
    for i, (x, y) in enumerate(dataset):
        if y in digits:
            indices.append(i)

    return indices
