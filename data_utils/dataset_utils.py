import numpy as np
from absl import flags
import torch
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
import torchvision.transforms as T

FLAGS = flags.FLAGS
torch.manual_seed(1)
# https://github.com/vahidk/tfrecord
# https://patrykchrabaszcz.github.io/Imagenet32/
def get_dataset(dataset=FLAGS.d):
    n_classes = 10
    if 'mnist' in dataset:
        n_label = 100 if FLAGS.nl == 0 else FLAGS.nl

        import data_loader1 as data_loader
        # selection_percentile = 0.1 if FLAGS.sp == 0. else FLAGS.sp
        size = 28
        channels = 1
        # sigma = 1.8 if FLAGS.s == 0. else FLAGS.s
        # if 'fashion' in dataset:
        #     sigma = 3.2 if FLAGS.s == 0. else FLAGS.s
        #     dataset = 'fashion_mnist'
    else:
        n_label = 4000 if FLAGS.nl == 0 else FLAGS.nl
        # if FLAGS.dl in "my":
        import data_loader1 as data_loader
        # selection_percentile = 0.05 if FLAGS.sp == 0. else FLAGS.sp
        size = 32
        channels = 3
        # sigma = 1.2 if FLAGS.s == 0. else FLAGS.s
        if 'cifar' in dataset:
            pass
            # FLAGS.zca = True
        if 'svhn' in dataset:
            n_label = 1000 if FLAGS.nl == 0 else FLAGS.nl
            # if FLAGS.dl in "my":
            import data_loader1 as data_loader
            FLAGS.d = 'svhn_cropped'
            # from utils import label_spreading as llgc_meta
            # sigma = 2.4 if FLAGS.s == 0. else FLAGS.s
        elif '100' in dataset:
            n_label = 10000 if FLAGS.nl == 0 else FLAGS.nl
            # if FLAGS.dl in "my":
            import data_loader1 as data_loader
            # selection_percentile = 0.02
            n_classes = 100
        elif 'imagenet' in dataset:
            n_label = 10000 if FLAGS.nl == 0 else FLAGS.nl
            n_classes = 1000
        elif 'plant' in dataset:
            n_label = 380 if FLAGS.nl == 0 else FLAGS.nl
            import data_loader1 as data_loader
            # selection_percentile = 0.02 if FLAGS.sp == 0. else FLAGS.sp
            if '32' in dataset:
                size = 32
            elif '64' in dataset:
                size = 64
            elif '96' in dataset:
                size = 96
                # if FLAGS.bs ==100:
                #     FLAGS.bs = 64
            elif '128' in dataset:
                size = 128
                # FLAGS.bs = 64
            channels = 3
            n_classes = 38
            # FLAGS.dl = "my"

    # FLAGS.sp = selection_percentile
    FLAGS.nc = n_classes
    # FLAGS.s = sigma
    return size, channels, n_label


def set_dataset(dataset=FLAGS.d, one_hot=False, scale=False):
    semi = FLAGS.semi
    shuffle = True  # FLAGS.shuffle
    zca = False  # FLAGS.zca
    size, channels, n_label = get_dataset(dataset)
    import data_loader1 as data_loader
    # import data_loader_py as data_loader
    dso = data_loader.read_data_sets(dataset, n_labeled=n_label, scale=scale, one_hot=one_hot, semi=semi,
                                     shuffle=shuffle)
    FLAGS.nl = dso.train.labeled_ds.num_examples
    return dso, size, channels


class MyDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x,y,test_data=False):
        'Initialization'
        data = x.astype(np.float64)
        data = 255 * data
        img = data.astype(np.uint8)
        self.x = img
        self.y = y
        self.is_test = test_data
        basic_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                        np.array([63.0, 62.1, 66.7]) / 255.0),

        ])
        if self.is_test:
            self.transform = basic_transform
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Pad(4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.RandomCrop(32),
                T.ToTensor(),
                T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                            np.array([63.0, 62.1, 66.7]) / 255.0),

            ])
    def  __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image = self.x[index]
        label = self.y[index]
        X = self.transform(image)
        return X, label


