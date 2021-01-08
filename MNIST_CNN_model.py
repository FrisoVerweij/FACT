import torch.nn as nn
import torch


class MNIST_CNN(nn.Module):

    def __init__(self, output_dim):
        '''
        Initialises the architecture of the MNIST_CNN model
        :param output_dim: dimension of the output of the model
        '''
        super(MNIST_CNN, self).__init__()
        self.all_modules = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        '''
        Calculates the output of the model given the input
        :param x: input for the model
        :return: torch.tensor: self.all_modules(x)
        '''
        out = self.all_modules(x)
        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs
