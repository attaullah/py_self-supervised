import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import multiprocessing
import numpy as np


def create_my_datasets(img, y=None, is_train=True, name="cifar10"):
    # data = x.astype(np.float64)
    # data = 255 * data
    # img = x.astype(np.uint8)
    # print(np.max(img[0,0,:,0]))
    # print("ALb transform for ", name, " train ", is_train)
    train_transform, val_transform = alb_transform(name)  # get_transforms
    if y is not None:
        return SSL(img, y)
        # if is_train:
        #     return AlbTransDataset(img, y, train_transform)  # CustomTensorDataset
        # return AlbTransDataset(img, y, val_transform)
    else:
        return SSLUNLAB(img)


def create_data_loader(images, labels=None, bs=128, shuffle=False, nthread=8, is_train=True, name="cifar10"):
    dataset = create_my_datasets(images, labels, is_train, name=name)
    nthread = min(nthread, multiprocessing.cpu_count()) #
    loader = DataLoader(dataset, bs, shuffle=shuffle, num_workers=nthread, pin_memory=True)
    return loader


class AlbTransDataset(Dataset):
    def __init__(self, x,y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def alb_transform(name="cifar10"):
    # print("ALb transform for ", name)
    if name == "svhn":
        train_transform = A.Compose(
            [
                A.Affine(translate_percent=(0.125, 0.125)),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Affine(translate_percent=(0.125, 0.125)),
                A.HorizontalFlip(p=0.5),

                ToTensorV2(),
            ]
        )
    val_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def get_transforms(name="cifar10"):
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if name == "svhn":
        print("svhn not horizontal flip")
        train_transforms = T.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomAffine(degrees=0, scale=(.125, 0.125)),
            T.ToTensor(),
            #
        ])
    else:
        train_transforms = T.Compose([
            T.ToTensor(),
            # transforms.ToPILImage(),
            T.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=0, scale=(.125, 0.125)),
        ])

    val_transforms = T.Compose([
        # transforms.ToPILImage(),
        T.ToTensor(),
        # transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x, y, transform=None):
        x = x.astype(np.uint8)
        self.x = x
        self.y = y

        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform:
            x = self.transform(x)

        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.y)


class CustomDataset(Dataset):
    def __init__(self, x, y, is_train=True):
        self.x = x
        self.y = y
        size = x.shape[2]
        self.train = is_train

        basic_transform = T.Compose([
            # T.ToPILImage(),
            T.ToTensor(),
        ])
        if self.train:
            self.transform = T.Compose([
                T.ToPILImage(),
                # T.Pad(4, padding_mode='reflect'),
                # T.RandomHorizontalFlip(),
                T.RandomCrop(size),
                T.ToTensor(),

            ])
        else:
            self.transform = basic_transform

    def  __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Select sample
        image = self.x[index]
        label = self.y[index]
        X = self.transform(image)
        return X, label


class IODataset(Dataset):

    def __init__(self, x, is_train=True):
        self.x = x
        self.train = is_train

        basic_transform = T.Compose([
            T.ToTensor(),
        ])
        if self.train:
            self.transform = A.Compose(
                [
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    def  __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image = self.x[index]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image


class SSL:
    def __init__(self, x, y):
        x = x.astype(np.uint8)
        self.x = x
        self.y = y
        print("type of dataset", type(self.x[0,0,0,0]), self.x.shape)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        image = (image/255. - 0.5)/0.5
        return image, label

    def __len__(self):
        return len(self.y)


class SSLUNLAB:
    def __init__(self, x):
        x = x.astype(np.uint8)
        self.x = x
        # self.y = y
        print("type of dataset", type(self.x[0,0,0,0]), self.x.shape)

    def __getitem__(self, idx):
        image = self.x[idx]
        # label = self.y[idx]
        image = (image/255. - 0.5)/0.5
        return image

    def __len__(self):
        return len(self.x)

