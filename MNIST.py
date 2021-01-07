"""
 Contains all the utilities to load the MNIST dataset
"""

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def get_mnist_dataloaders(root='./datasets/', batch_size=64, numbers_to_include=None):
    '''
    Loads the mnist train and test set into a dataloader
    :param root: dir to save dataset
    :param batch_size: size of batch
    :param numbers_to_include: array of labels to be included into the dataset. Default=None (all labels are included)
    :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_dataset = MNIST(root=root, download=True, train=True, transform=transform)

    mnist_test_dataset = MNIST(root=root, download=True, train=False, transform=transform)
    train_dataloader = DataLoader(mnist_train_dataset, shuffle=True, batch_size=batch_size, )
    test_dataloader =  DataLoader(mnist_test_dataset, shuffle=True, batch_size=batch_size, )

    return train_dataloader, test_dataloader











