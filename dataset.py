"""
 Contains all the utilities to load the MNIST dataset
"""
import os
import random
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import transforms

from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from PIL import Image

class MNIST_dummy(MNIST):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            digits_to_include: list = None,
    ) -> None:
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.digits_to_include = digits_to_include

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(img, target, self.digits_to_include)

        return img, target

class CustomTransform(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        coin = random.randint(0, 1)
        if coin == 0:
            sample[:, 0, 0] = 0
            sample[:, 0, 1] = 0
            sample[:, 0, 2] = 0

            sample[:, 1, 0] = 0
            sample[:, 1, 1] = 0
            sample[:, 1, 2] = 0

            sample[:, 2, 0] = 0
            sample[:, 2, 1] = 0
            sample[:, 2, 2] = 0

        elif coin == 1:
            sample[:, 0, 0] = 1
            sample[:, 0, 1] = 1
            sample[:, 0, 2] = 1

            sample[:, 1, 0] = 1
            sample[:, 1, 1] = 1
            sample[:, 1, 2] = 1

            sample[:, 2, 0] = 1
            sample[:, 2, 1] = 1
            sample[:, 2, 2] = 1

        return sample


def transform_y(x, y, digits_to_include):
    if y in digits_to_include:
        if x[:, 0, 0] == 0:
            return 0
        elif x[:, 0, 0] == 1:
            return 1
        else:
            print("first pixel: ", x[:, :, :])
            exit()
    else:
        return -1


def get_mnist_dataloaders(root='./datasets/', batch_size=64, digits_to_include: list = None, size=None, dummy=False):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param digits_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    if dummy:
        transform = transforms.Compose([transforms.ToTensor(), CustomTransform()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    if digits_to_include is None:
        mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform)
        mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform)

    else:
        if dummy:
            label_transform = transform_y
            mnist_train_dataset = MNIST_dummy(root=root, download=True, train=True, transform=transform,
                                              target_transform=label_transform, digits_to_include=digits_to_include)
            mnist_test_dataset = MNIST_dummy(root=root, download=True, train=False, transform=transform,
                                             target_transform=label_transform, digits_to_include=digits_to_include)
        else:
            label_transform = lambda x: digits_to_include.index(x) if x in digits_to_include else -1
            mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform,
                                        target_transform=label_transform)
            mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform,
                                       target_transform=label_transform)

        train_indices = get_indices(mnist_train_dataset)
        test_indices = get_indices(mnist_test_dataset)
        mnist_train_dataset = Subset(mnist_train_dataset, train_indices, )
        mnist_test_dataset = Subset(mnist_test_dataset, test_indices)
    if size != None:
        mnist_train_dataset = Subset(mnist_train_dataset, range(size), )
        mnist_test_dataset = Subset(mnist_test_dataset, range(size), )
    train_dataloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(mnist_test_dataset, shuffle=False, batch_size=batch_size, )

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

    if digits_to_include is None:
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
    test_dataloader = DataLoader(mnist_test_dataset, shuffle=False, batch_size=batch_size, )

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

    if digits_to_include is None:
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
    test_dataloader = DataLoader(fmnist_test_dataset, shuffle=False, batch_size=batch_size, )

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

    if digits_to_include is None:
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
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, )

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
