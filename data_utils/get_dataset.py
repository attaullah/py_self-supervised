import numpy as np
from .dataset import DataSet, SemiDataSet
from .dataset_config import data_details
# import data_loader
from .data_loader_py import read_data_sets as get_dataset


def read_data_sets(name, one_hot=False, semi=True, scale=False, shuffle=True, channel_first=True):
    class DataSets(object):
        pass
    data_sets = DataSets()

    train_images, train_labels, test_images, test_labels = get_dataset(name, channel_first=channel_first)
    if scale:
        test_images = np.multiply(test_images, 1.0 / 255.0)
        train_images = np.multiply(train_images, 1.0 / 255.0)

    n_labeled, selection_percentile, sigma = data_details(name)
    n_classes = len(np.unique(test_labels))
    if semi:
        data_sets.train = SemiDataSet(train_images, train_labels, n_labeled, one_hot=one_hot, n_classes=n_classes,
                                      shuffle=shuffle)
    else:
        n_labeled = train_images.shape[0]  # in case of all-labelled examples
        data_sets.train = DataSet(train_images, train_labels, one_hot=one_hot, shuffle=shuffle)
    data_sets.test = DataSet(test_images, test_labels, one_hot=one_hot, shuffle=shuffle, n_classes=n_classes)

    class Config(object):
        pass
    # dataset attributes
    data_config = Config()
    data_config.name = name
    data_config.channels = train_images.shape[1] if channel_first else train_images.shape[-1]
    data_config.size = train_images.shape[2]
    data_config.nc = n_classes
    data_config.n_label = n_labeled
    data_config.sp = selection_percentile
    data_config.sigma = sigma
    data_config.semi = semi

    return data_sets, data_config

