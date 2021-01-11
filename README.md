Paper reproduction

## Requirements
- Numpy
- PyTorch

## How to use
###classifier_trainloop.py###
####Interface####
To create and train a classification model, call the *classifier_trainloop.py* file using the commandline using the hyperparameters provided below.
After training, the model is saved in the folder models/

####Hyperparameters####
Specify the following parameters when running from commandline:
- model: default: mnist_cnn, choices: mnist_cnn
- epochs: default: 50
- batch_size: default: 64
- lr: default: 0.1
- momentum: default: 0.5
- optimizer: default: SGD, choices: SGD, Adam
- device: cuda if available, cpu otherwise
- dataset: default: mnist, choices: mnist
- mnist_digits: default: None	E.g. --mnist_digits=3,8