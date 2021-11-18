import numpy as np
import os
from scipy.io import loadmat
import plant_village
import _pickle as cPickle

base_path = '/path/to/'


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Args:
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.
    Returns:
      A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
          d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def cifar10(image_data_format="channel-first"):
    path = base_path + 'datasets/cifar10'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if image_data_format == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return x_train, y_train, x_test, y_test


def svhn(channel_first=True):

    train_dir = base_path + 'datasets/svhn/cropped'
    train_file = train_dir + '/train_32x32.mat'
    test_file = train_dir + '/test_32x32.mat'

    data = loadmat(train_file)
    train_images, train_labels = data['X'], data['y']
    train_labels[train_labels == 10] = 0

    data = loadmat(test_file)
    test_images, test_labels = data['X'], data['y']
    test_labels[test_labels == 10] = 0
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    if channel_first:
        train_images = train_images.transpose(0, 3, 2, 1)
        test_images = test_images.transpose(0, 3, 2, 1)
    else:
        train_images = train_images.transpose(3, 0, 1, 2)
        test_images = test_images.transpose(3, 0, 1, 2)

    return train_images, train_labels, test_images, test_labels


def read_data_sets(name):

    if name == 'cifar10':
        train_images, train_labels, test_images, test_labels = cifar10()
    elif 'svhn' in name:
        train_images, train_labels, test_images, test_labels = svhn()
    else:
        # 'plant_' in name:
        train_images, train_labels, test_images, test_labels = plant_village.load_dataset(name)

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    return train_images, train_labels, test_images, test_labels
