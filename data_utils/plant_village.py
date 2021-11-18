import os
import sys
import numpy as np

# path to PlantVillage dataset containing train and test directories or plant**.npz for 32,64,and 96
base_path = "../githubs/myrepos/downsampled-plant-disease-dataset/data"  # /research/repository/a19


def load_dataset(name):
    file_path = base_path + '/' + name + '.npz'
    if os.path.exists(file_path):
        npzfile = np.load(file_path)
        train_images, train_labels,  = npzfile['train_images'], npzfile['train_labels']
        test_images, test_labels = npzfile['test_images'], npzfile['test_labels']
        return train_images, train_labels, test_images, test_labels
    else:
        print("dataset not found at: ", file_path)
        sys.exit(1)
