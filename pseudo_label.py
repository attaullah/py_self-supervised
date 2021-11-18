import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math, time, json, os
from models.model import  get_network, get_optimizer

from torchmetrics import MeanMetric, Accuracy
from data_utils.DataLoaders import create_data_loader
from tqdm import tqdm
import pkbar
import torch
import pandas as pd
from torchsummary import summary
import torch
import random
import numpy as np
from data_utils import get_dataset
from data_utils.DataLoaders import CustomDataset
device = torch.device("cuda")

iterations = 500000
warmup = 200000
lr_decay_iter = 400000
lr_decay_factor = 0.2
validation = 1500


class PL(nn.Module):
    def __init__(self, threshold=0.95, n_classes=10):
        super().__init__()
        self.th = threshold
        self.n_classes = n_classes

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0]  # reduce_any
        lt_mask = 1 - gt_mask  # logical not
        p_target = gt_mask[:, None] * 10 * onehot_label + lt_mask[:, None] * y_probs
        model.update_batch_stats(False)
        output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        model.update_batch_stats(True)
        return loss

    def __make_one_hot(self, y ):
        return torch.eye(self.n_classes)[y].to(y.device)


class transform:
    def __init__(self, flip=True, r_crop=True, g_noise=False):
        self.flip = flip
        self.r_crop = r_crop
        self.g_noise = g_noise
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
            self.flip, self.r_crop, self.g_noise
        ))

    def __call__(self, x):
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
        if self.r_crop:
            h, w = x.shape[-2:]
            x = F.pad(x, [2,2,2,2], mode="reflect")
            l, t = random.randint(0, 4), random.randint(0,4)
            x = x[:,:,t:t+h,l:l+w]
        if self.g_noise:
            n = torch.randn_like(x) * 0.15
            x = n + x
        return x


def get_datasets(dso, dataloader="my", semi=True, dataset=""):
    if dataloader == "my":
        # dso, data_config = get_dataset.read_data_sets(dataset, one_hot=False, semi=semi, scale=True)
        print("!! check label imag shape", dso.train.labeled_ds.images.shape, np.max(dso.test.images[10,0,0,0]))
        l_train_dataset = TensorDataset(torch.Tensor(dso.train.labeled_ds.images),
                                        torch.Tensor(dso.train.labeled_ds.labels))
        u_train_dataset = TensorDataset(torch.Tensor(dso.train.unlabeled_ds.images),
                                        torch.Tensor(np.zeros_like(dso.train.unlabeled_ds.labels) - 1))
        test_dataset = TensorDataset(torch.Tensor(dso.test.images), torch.Tensor(dso.test.labels))
        transform_fn = None
    else:  # dataloader == "custom":

        dso, data_config = get_dataset.read_data_sets(dataset, one_hot=False, semi=semi, scale=False,
                                                      channel_first=False)
        print("!! check label imag shape", dso.train.labeled_ds.images.shape)
        test_dataset = CustomDataset(dso.test.images, dso.test.labels)
        l_train_dataset = CustomDataset(dso.train.labeled_ds.images, dso.train.labeled_ds.labels)  # create your datset
        u_train_dataset = CustomDataset(dso.train.unlabeled_ds.images,
                                        (np.zeros_like(dso.train.unlabeled_ds.labels) - 1))
        transform_fn = None  # dataset includes transforms
    return l_train_dataset, u_train_dataset, test_dataset


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_data_loaders(dso, dl="my", t_type="original", alg="PL", bs=128, steps=500000):
    l_train_dataset, u_train_dataset, test_dataset = get_datasets(dso, dataloader=dl)
    print("datasets size:: ",len(l_train_dataset),len(u_train_dataset), len(test_dataset))
    if t_type == "original":
        if alg != "supervised":
            # batch size = 0.5 x batch size
            l_loader = DataLoader(l_train_dataset, bs // 2, drop_last=True,
                                  sampler=RandomSampler(len(l_train_dataset), steps * bs // 2))
        else:
            l_loader = DataLoader(l_train_dataset, bs, drop_last=True,
                                  sampler=RandomSampler(len(l_train_dataset), steps * bs))
        u_loader = DataLoader(u_train_dataset, bs // 2, drop_last=True,
                              sampler=RandomSampler(len(u_train_dataset), steps * bs // 2))
    else:
        l_loader = DataLoader(l_train_dataset, bs, drop_last=True)
        u_loader = DataLoader(u_train_dataset, bs // 2, drop_last=True, )
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

    return l_loader, u_loader, test_loader


def set_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3):
    ssl_obj = PL(n_classes=data_config.nc)
    model, _ = get_network(arch, data_config.size, data_config.channels,num_classes=data_config.nc)
    optimizer = get_optimizer(opt, lr, model.parameters())
    # model, optimizer, criterion = get_model(arch, data_config, weights, loss_type, opt, lr)
    summary(model, input_size=(data_config.channels, data_config.size, data_config.size), device="cpu")  # .to(device)
    # model = ssdl.SSDL(3, 64, dataset_cfg["num_classes"], transform_fn=transform_fn)
    # optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

    return model, ssl_obj, optimizer, F.cross_entropy


def train_orig(model, ssl_obj, optimizer, l_loader, u_loader, test_loader,  alg="PL", ):
    print()
    consis_coef = 1  # for PL
    iteration = 0
    s = time.time()
    for l_data, u_data in zip(l_loader, u_loader):
        iteration += 1
        l_input, target = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()
        if alg != "supervised":  # for ssl algorithm
            u_input, dummy_target = u_data
            u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

            target = torch.cat([target, dummy_target], 0)
            unlabeled_mask = (target == -1).float()

            inputs = torch.cat([l_input, u_input], 0)
            outputs = model(inputs)

            # ramp up exp(-5(1 - t)^2)
            coef = consis_coef * math.exp(-5 * (1 - min(iteration/warmup, 1))**2)
            ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef
        else:
            outputs = model(l_input)
            coef = 0
            ssl_loss = torch.zeros(1).to(device)

        # supervised loss
        cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()

        loss = cls_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # display
        if iteration == 1 or (iteration % 100) == 0:
            wasted_time = time.time() - s
            rest = (iterations - iteration)/100 * wasted_time / 60
            print("iteration [{}/{}] cls loss : {:.4f}, SSL loss : {:.4f}, coef : {:.5e}, lr : {}".format(iteration,
                  iterations, cls_loss.item(), ssl_loss.item(), coef,  optimizer.param_groups[0]["lr"]),
                  "\r", end="")
            s = time.time()

        # validation
        if (iteration % validation) == 0 or iteration == iterations:
            sum_acc = 0.
            s = time.time()
            for j, data in enumerate(test_loader):
                input, target = data
                input, target = input.to(device).float(), target.to(device).long()
                output = model(input)
                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()
                if ((j+1) % 10) == 0:
                    d_p_s = 100/(time.time()-s)
                    s = time.time()

            test_acc = sum_acc / float(len(test_loader.dataset))
            print("test accuracy : {:.4f}".format(test_acc))
            model.train()
            s = time.time()
        # lr decay
        if iteration == lr_decay_iter:
            optimizer.param_groups[0]["lr"] *= lr_decay_factor
    return test_acc


def train_pbar(model, ssl_obj, optimizer, lab_loader, unlab_loader, epochs=10, bs=128, verbose=True, dev=device,
               t_loader=None, print_freq=10, alg="PL"):
    log = []
    consis_coef = 1
    step = 0
    num_of_batches_per_epoch = np.ceil(len(unlab_loader.dataset) / bs)
    train_per_epoch = num_of_batches_per_epoch
    if alg == "supervised":
        train_per_epoch /= 2

    if verbose:
        loop_range = range(epochs)
    else:
        loop_range = tqdm(range(epochs))
    accuracy_metric = Accuracy().to(dev)
    loss_metric = MeanMetric().to(dev)
    loss_ssl_metric = MeanMetric().to(dev)
    model.to(dev)
    for epoch in loop_range:
        accuracy_metric.reset()
        loss_metric.reset()
        loss_ssl_metric.reset()
        model.train()
        kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=epochs, width=8, always_stateful=True)

        for i, (l_data, u_data) in enumerate(zip(lab_loader, unlab_loader)):
            step += 1
            l_input, target = l_data
            l_input, target = l_input.to(device).float(), target.to(device).long()
            if alg != "supervised":  # for ssl algorithm
                u_input, dummy_target = u_data
                u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

                target = torch.cat([target, dummy_target], 0)
                unlabeled_mask = (target == -1).float()

                inputs = torch.cat([l_input, u_input], 0)
                outputs = model(inputs)

                # ramp up exp(-5(1 - t)^2)
                coef = consis_coef * math.exp(-5 * (1 - min(step / warmup, 1)) ** 2)
                ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef
            else:
                outputs = model(l_input)
                coef = 0
                ssl_loss = torch.zeros(1).to(device)

            # supervised loss
            cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()

            loss = cls_loss + ssl_loss
            loss_ssl_metric.update(ssl_loss)
            loss_metric.update(ssl_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == lr_decay_iter:
                optimizer.param_groups[0]["lr"] *= lr_decay_factor
            # display
            if verbose:
                kbar.update(i, values=[("sup-loss", loss_metric.compute()), ("ssl-loss", loss_ssl_metric.compute()),
                                       ("wt", coef)])
                template = "iteration [{}/{}] cls loss : {:.4f}, SSL loss : {:.4f}, coef : {:.5e}, lr : {}"
                # print(template.format(i, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef,
                #                       optimizer.param_groups[0]["lr"]), "\r", end="")

        # validation
        if (epoch+1) % print_freq == 0 and verbose:
            if t_loader is not None:
                val_loss, test_acc = evaluate(model, t_loader, dev=dev)
                kbar.add(1, values=[("val_loss", val_loss), ("val_acc", test_acc)])
                print("test accuracy : {:.4f}".format(test_acc))

    return test_acc


def evaluate(net, imgs, lbls=None, loss_fn=None, verbose=False, dev=None, bs=128):
    if isinstance(imgs, DataLoader):
        dl = imgs
    else:
        dl = create_data_loader(imgs, lbls, bs=bs)
    net.to(dev)
    net.eval()
    num_of_batches_per_epoch = np.ceil(len(dl.dataset) / dl.batch_size)
    kbar = pkbar.Kbar(target=num_of_batches_per_epoch, epoch=None, num_epochs=None, width=8, always_stateful=False)

    acc = Accuracy().to(dev)
    loss = MeanMetric().to(dev)
    test_loss = 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(dl):
            data, target = data.to(dev), target.to(dev)
            target = target.type(torch.LongTensor).to(dev)
            output = net(data)
            _, preds = torch.max(output.data, 1)
            # if loss_fn:
            test_loss = F.cross_entropy(output, target, reduction="none", ignore_index=-1).mean()  #(output, target)
            loss.update(test_loss)
            acc.update(preds, target)
            if verbose:
                kbar.update(i, values=[("val_loss", loss.compute()), ("val_acc", acc.compute())])

    return loss.compute().cpu().numpy().squeeze(), acc.compute().cpu().numpy()


def start_training(model, dso, epochs=100, semi=True, bs=100, verb=True, train_type="original", alg="PL", dl="my"):
    # images, labels = dso.train.labeled_ds.images, dso.train.labeled_ds.labels
    l_loader, u_loader, test_loader = get_data_loaders(dso, dl=dl, t_type=train_type, alg=alg, bs=bs, steps=iterations)
    model, ssl_obj, optimizer, criterion = model
    if train_type == "original":
        train_orig(model, ssl_obj, optimizer, l_loader, u_loader, test_loader, alg)
    else:
        train_pbar(model, ssl_obj, optimizer, l_loader, u_loader, epochs=epochs, verbose=verb, t_loader=test_loader,
                   alg=alg)


