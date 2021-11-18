import numpy as np
import tensorflow
import tensorflow.keras


def geometric_transform(image, auxiliary_labels=6, one_hot=True):
    """
    Applies six transformations: rotation by 0,90, 180, 270 and flip left-right and flip upside-down.
    :param image: input image
    :param auxiliary_labels: number of auxiliary labels
    :param one_hot: apply one-hot encoding to labels
    :return: six images with all six transformations applied and auxiliary labels 0...auxiliary_labels-1
    """
    _, h, w, c = image.shape
    image = np.reshape(image, (h, w, c))
    labels = np.empty((auxiliary_labels,), dtype='uint8')
    images = np.empty((auxiliary_labels, h, w, c), dtype='float32')
    for i in range(auxiliary_labels):
        if i <= 3:
            t = np.rot90(image, i)
        elif i == 4:
            t = np.fliplr(image)
        else:
            t = np.flipud(image)
        images[i] = t
        labels[i] = i
    if one_hot:
        return images, np.eye(auxiliary_labels)[labels]
    else:
        # print('one_hot = ', one_hot, labels.shape)
        labels = labels.squeeze()
        # print('one_hot = ', one_hot, labels.shape)
        return images, labels


def combined_generator(super_iter, self_iter, batch_size, one_hot=True):
    """
    Utility function to load data into required Keras model format for training on supervised batch and self-supervised
    batch.
    :param super_iter: supervised data generator based on labelled training images
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    """
    super_batch = batch_size * 1
    self_batch = batch_size
    while True:
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[geometric_transform(next(self_iter), one_hot=one_hot)
                               for _ in range(self_batch)])

        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.vstack(x_self)
        y_self = np.vstack(y_self)
        if not one_hot:
            y_self = y_self.ravel()
        yield [x_self, x_super], [y_self, y_super]


def combined_pseudo_generator(labeled_data, pseudo_data, batch_size):

    while True:
        x_labeled, y_labeled = zip(*[next(labeled_data) for _ in range(batch_size)])
        x_pseudo, y_pseudo = zip(*[next(pseudo_data) for _ in range(batch_size)])

        x_labeled = np.vstack(x_labeled)
        y_labeled = np.vstack(y_labeled)
        x_pseudo = np.vstack(x_pseudo)
        y_pseudo = np.vstack(y_pseudo)
        # print("before yield ", x_labeled.shape, y_labeled.shape, x_pseudo.shape, y_pseudo.shape)
        yield [x_labeled, y_labeled], [x_pseudo, y_pseudo]


def self_supervised_data_generator(self_iter, batch_size, one_hot=True):
    """
    Utility function to load data into required Keras model format.
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    """
    self_batch = batch_size
    while True:
        x_self, y_self = zip(*[geometric_transform(next(self_iter), one_hot=one_hot)
                               for _ in range(self_batch)])
        x_self = np.vstack(x_self)
        if not one_hot:
            y_self = np.array(y_self)
            y_self = y_self.ravel()
        y_self = np.vstack(y_self)
        yield x_self, y_self


class RotationalGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, x_train, batch_size=32, one_hot=True, shuffle=True):

        self.X_train = x_train
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X_train))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.X_train) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: of batches
        :return: images and labels for a batch
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        x = self.X_train[batch_ids]
        x_self, y = zip(*[geometric_transform(x, one_hot=self.one_hot)
                        for _ in range(self.batch_size)])
        x = np.vstack(x_self)
        if not self.one_hot:
            y = np.array(y)
            y = y.ravel()
        y = np.vstack(y)
        # for i in range(self.batch_size):
        #     x[i] = self.augmentations.random_transform(x[i])
        return x, y


class RotationalCombinedGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, lab_images, lab_labels, x_train, augmentations=None, batch_size=32, one_hot=True, shuffle=True):

        self.lab_images = lab_images
        self.lab_labels = lab_labels
        self.X_train = x_train
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.self_indexes = np.arange(len(self.X_train))
        self.lab_indexes = np.arange(len(self.lab_images))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.X_train) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: of batches
        :return: images and labels for a batch
        """
        # Generate indexes of the batch
        indexes = self.self_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        lab_indexes = self.lab_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes, lab_indexes)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.self_indexes)
            np.random.shuffle(self.lab_indexes)

    def __data_generation(self, batch_ids, lab_batch_ids):
        x_lab, y_lab = self.lab_images[lab_batch_ids], self.lab_labels[lab_batch_ids]
        for i in range(self.batch_size):
            x_lab[i] = self.augmentations.random_transform(x_lab[i])
        x = self.X_train[batch_ids]
        x_self, y = zip(*[geometric_transform(x, one_hot=self.one_hot)
                        for _ in range(self.batch_size)])
        x_self = np.vstack(x_self)
        if not self.one_hot:
            y = np.array(y)
            y = y.ravel()
        y = np.vstack(y)

        return [x_self, x_lab], [y, y_lab]
