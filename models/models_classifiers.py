from abc import ABC

import torch.nn as nn
import torch


class DummyClassifier(nn.Module):
    def __init__(self):
        super(DummyClassifier, self).__init__()
        self.dummyLayer = nn.Linear(1, 2)

    def forward(self, x):
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        x = torch.reshape(x, (x.shape[0], 1))
        out = self.dummyLayer(x)
        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs


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
        

class VGG11(nn.Module):
    def __init__(self, output_dim):
        super(VGG11, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        self.eval()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x_prob = torch.softmax(x, dim=-1)
        return x, x_prob