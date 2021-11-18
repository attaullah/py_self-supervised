import numpy as np
from .plant_village import  load_dataset
import torchvision
import os

base_path = '/research/repository/a19/datasets/'
if "weka6" in os.uname()[1]:
    from pathlib import Path
    home = str(Path.home())
    # base_path = home + '/Desktop/sftp/repository/a19/'
    base_path = '/home/attaullah/Desktop/sftp/research/repository/a19/datasets/'
elif "weka14" in os.uname()[1]:
    base_path = '/home/ml/Desktop/sftp/research/repository/a19/datasets/'

# base_path = "../githubs/myrepos/downsampled-plant-disease-dataset/data"  # /research/repository/a19


def mnist(name):
    if 'fashion' in name:
        train_dataset = torchvision.datasets.FashionMNIST(base_path + name, train=True, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(base_path + name, train=False, download=True)
    else:
        train_dataset = torchvision.datasets.MNIST(base_path + name, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(base_path + name, train=False, download=True)
    
    train_images = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_labels = test_dataset.targets.numpy()
    test_images = test_dataset.data.numpy()

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    return train_images, train_labels, test_images, test_labels


def cifar10(name, channel_first=False):
    train_dataset = torchvision.datasets.CIFAR10(base_path + name, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(base_path + name, train=False, download=True)

    train_images = train_dataset.data
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    test_images = test_dataset.data
    if channel_first:
        train_images = train_images.transpose(0, 3, 1, 2)
        test_images = test_images.transpose(0, 3, 1, 2)
    return train_images, train_labels, test_images, test_labels


def svhn(name):

    train_dataset = torchvision.datasets.SVHN(base_path + name, 'train', download=True)
    test_dataset = torchvision.datasets.SVHN(base_path + name, 'test', download=True)

    train_images = train_dataset.data
    train_labels = train_dataset.labels
    test_images = test_dataset.data
    test_labels = test_dataset.labels

    train_images = train_images.transpose((0, 3, 2, 1))
    test_images = test_images.transpose((0, 3, 2, 1))

    return train_images, train_labels, test_images, test_labels


def read_data_sets(name, channel_first=True):

    if 'mnist' in name:
        train_images, train_labels, test_images, test_labels = mnist('torch/' + name)
    elif name == 'cifar10':
        train_images, train_labels, test_images, test_labels = cifar10(name)
    elif 'svhn' in name:
        train_images, train_labels, test_images, test_labels = svhn(name)
    elif 'plant' in name:
        train_images, train_labels, test_images, test_labels = load_dataset(name)
    else:
        print("Dataset: ", name, " not available.")
        return

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    if channel_first:
        train_images = train_images.transpose(0, 3, 1, 2)
        test_images = test_images.transpose(0, 3, 1, 2)
    print('check ...!!!', train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    return train_images, train_labels, test_images, test_labels
