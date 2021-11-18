import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, size=32, channels=3, emb_size=4, num_classes=10):
        super(SimpleModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=7, stride=2, padding=2,)
        self.bn1 = nn.BatchNorm2d(32)
        self.mp = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2,)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2,)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=2, )
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, emb_size, kernel_size=1, stride=1, padding=1,)
        self.bn5 = nn.BatchNorm2d(emb_size)
        self.flatten = nn.Flatten()
        self.classes = num_classes
        if self.classes != 16:
            self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.mp(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.mp(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.mp(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.mp(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.bn5(out)
        out = self.mp(out)

        out = self.flatten(out)
        if self.classes != 16:
            out = self.fc(out)
        return out


class SSDL(nn.Module):
    def __init__(self, channels=3, emb_size=64, num_classes=-1, transform_fn=None):
        super(SSDL, self).__init__()
        self.transform_fn = transform_fn
        self.relu = nn.ReLU()  # inplace=True)
        self.conv1 = nn.Conv2d(channels, 192, kernel_size=5, stride=1, padding=2,)# bias=False)
        self.bn1 = nn.BatchNorm2d(192)
        self.conv2 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=2, )#bias=False)
        self.bn2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=2, )#bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.mp = nn.MaxPool2d(3, 2)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2, )#bias=False)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=2, )#bias=False)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=2, )#bias=False)
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=2, )#bias=False)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, emb_size, kernel_size=1, stride=1, padding=2, )#bias=False)
        self.bn8 = nn.BatchNorm2d(emb_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classes = num_classes
        if self.classes > 0:
            self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.mp(out)  # 3

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.bn5(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.bn6(out)
        out = self.mp(out)  # 6

        out = self.conv7(out)
        out = self.relu(out)
        out = self.bn7(out)
        out = self.conv8(out)
        # out = self.relu(out) # 8
        out = self.bn8(out)
        out = self.gap(out)
        out = self.flatten(out)
        if self.classes > 0:
            out = self.fc(out)
        return out

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class MyModel(nn.Module):
    def __init__(self, channels=1, emb_size=4, num_classes=10, transform_fn=None):
        super(MyModel, self).__init__()
        self.transform_fn = transform_fn
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=1,)
        self.bn1 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, )
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, )
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, )
        self.bn5 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classes = num_classes
        if self.classes <= 38:
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.mp(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.mp(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.bn5(out)
        # print(' model size ==   ', out.size())
        out = self.gap(out)
        out = self.flatten(out)
        if self.classes <= 38:
            out = self.fc(out)
        return out