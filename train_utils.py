import os
import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils import feature_scaling
from utils.shallow_classifiers import shallow_clf_accuracy
from scipy.spatial.distance import cdist
from models.model import get_model
from data_utils import get_dataset
import time
from torchmetrics import MeanMetric, Accuracy
from data_utils.DataLoaders import create_data_loader
from tqdm import tqdm
import pkbar
import torch
import pandas as pd
from torchsummary import summary
from torch.utils.data import DataLoader
import platform


device = torch.device("cuda")
template1 = "Labeled= {} selection={}% iterations= {}"
template2 = 'total selected based on percentile {} having accuracy {:.2f}%'
template3 = 'Length of predicted {},  unlabeled {}'


def set_dataset(dataset, lt, semi=True, scale=False, channel_first=False):
    one_hot = True if lt.lower() == "arcface" else False
    dso, data_config = get_dataset.read_data_sets(dataset, one_hot, semi, scale=scale, channel_first=channel_first)
    return dso, data_config


def set_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3,lr_sched=None):
    model, optimizer, criterion, lr_sched = get_model(arch, data_config, weights, loss_type, opt, lr, lr_sched)
    summary(model, input_size=(data_config.channels, data_config.size, data_config.size), device='cpu')
    return model, optimizer, criterion, lr_sched


def get_log_name(flags, data_config, prefix=""):
    path = prefix + flags.lt + '_logs/' + flags.dataset + '/' + flags.network + '/'

    log_name = str(data_config.n_label) + '-'
    weights = '-w' if flags.weights else ''
    if flags.lt != "cross-entropy":
        log_name += flags.lbl + '-'

    log_name += flags.opt.lower() + weights
    if flags.self_training:
        log_name = log_name + '-self-training-'
        if flags.lt != "cross-entropy":
            log_name += flags.confidence_measure

    return path, log_name


def predict(model, imgs, verbose=False, dev=None,  bs=128):
    if isinstance(imgs, DataLoader):
        dl = imgs
    else:
        dl = create_data_loader(imgs, bs=bs, is_train=False)

    model.to(dev)
    model.eval()
    num_of_batches_per_epoch = np.ceil(len(dl.dataset) / dl.batch_size)
    kbar = pkbar.Kbar(target=num_of_batches_per_epoch, epoch=None, num_epochs=None, width=8, always_stateful=False)

    with torch.no_grad():
        for i, data in enumerate(dl):
            data = data.to(dev).float()
            output = model(data)
            current = output.data.cpu().numpy()
            if i == 0:
                es = current.shape[1]
                outputs1 = np.zeros((len(imgs), es))
            up_limit = min((i+1)*bs, i*bs+len(current))
            outputs1[i*bs:up_limit, :] = current
            if verbose:
                kbar.update(i,)
    return outputs1


def evaluate(model, imgs, lbls=None, loss_fn=None, verbose=False, dev=None, bs=128):
    # print("EVAL func::  device = ", dev, " labels shape   ", lbls.shape)
    if isinstance(imgs, DataLoader):
        dl = imgs
    else:
        dl = create_data_loader(imgs, lbls, bs=bs)
    model.to(dev)
    model.eval()
    num_of_batches_per_epoch = np.ceil(len(dl.dataset) / dl.batch_size)
    kbar = pkbar.Kbar(target=num_of_batches_per_epoch, epoch=None, num_epochs=None, width=8, always_stateful=False)

    acc = Accuracy(task="multiclass",num_classes=10).to(dev)
    loss = MeanMetric().to(dev)
    test_loss = 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(dl):
            data, target = data.to(dev).float(), target.to(dev).long()
            # target = target.type(torch.LongTensor).to(dev)
            output = model(data)
            _, preds = torch.max(output.data, 1)
            if loss_fn:
                test_loss = loss_fn(output, target)
            loss.update(test_loss)
            acc.update(preds, target)
            if verbose:
                kbar.update(i, values=[("val_loss", loss.compute()), ("val_acc", acc.compute())])

    return loss.compute().cpu().numpy().squeeze(), acc.compute().cpu().numpy()


def compute_supervised_accuracy(model, imgs, lbls, ret_labels=False, v=False, bs=128):
    if ret_labels:
        pred = predict(model, imgs, verbose=v, bs=bs)
        pred_lbls = np.argmax(pred, 1)
        accuracy = accuracy_score(lbls, pred_lbls)
        return accuracy, pred_lbls
    else:
        accuracy = evaluate(model, imgs, lbls, verbose=v, bs=bs, dev=device)
        accuracy = np.round(accuracy[1] * 100., 2)
        return accuracy


def get_network_embeddings(model, input_images,  bs=128):
    return predict(model, input_images, bs=bs, dev=device)


def get_network_output(model, input_images, lt="", scaling=False, v=False, bs=128):

    if 'ntropy' in lt:
        feat = predict(model, input_images, verbose=v, bs=bs, dev=device)
    else:
        feat = get_network_embeddings(model, input_images, bs=bs)
    if scaling:
        feat, _, _ = feature_scaling(feat)
    return feat


def compute_embeddings_accuracy(model, imgs, lbls, test_imgs, test_lbls, labelling="knn", loss_type="",
                                ret_labels=False, scaling=True):

    if lbls.ndim > 1:  # for Arcface, convert one-hot encodings to simple
        lbls = np.argmax(lbls, 1)
        test_lbls = np.argmax(test_lbls, 1)
    labels, accuracy = shallow_clf_accuracy(get_network_output(model, imgs, loss_type, scaling=scaling), lbls,
                                            get_network_output(model, test_imgs, loss_type, scaling=scaling),
                                            test_lbls, labelling)
    accuracy = np.round(accuracy * 100., 2)
    if ret_labels:
        return accuracy, labels
    return accuracy


def compute_embeddings_acc_loaders(model, lab_loader, test_loader, labelling="knn", ret_labels=False, scaling=True):

    labels, accuracy = shallow_clf_accuracy(get_network_output(model, lab_loader.dataset.x, scaling=scaling),
                                            lab_loader.dataset.y,
                                            get_network_output(model, test_loader.dataset.x, scaling=scaling),
                                            test_loader.dataset.y, labelling)
    accuracy = np.round(accuracy * 100., 2)
    if ret_labels:
        return accuracy, labels
    return accuracy


def compute_accuracy(model, train_images, train_labels, test_images, test_labels, loss_type="cross-entropy",
                     labelling="knn"):
    if 'tropy' in loss_type:
        ac = compute_supervised_accuracy(model, test_images, test_labels)
    else:
        ac = compute_embeddings_accuracy(model, train_images, train_labels, test_images, test_labels,
                                         loss_type=loss_type, labelling=labelling)
    return ac


def log_accuracy(model, dso, loss_type="", semi=True, labelling="knn"):
    if semi:
        acc = compute_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images,
                               dso.test.labels, loss_type=loss_type, labelling=labelling)
    else:
        acc = compute_accuracy(model, dso.train.images, dso.train.labels, dso.test.images, dso.test.labels,
                               loss_type=loss_type, labelling=labelling)
    return acc


def start_training(model, dso, epochs=100, semi=True, bs=100, verb=True, name="cifar10", lr_sched=None):
    if semi:  # N-labelled
        images, labels = dso.train.labeled_ds.images, dso.train.labeled_ds.labels
    else:  # all-labelled examples
        images, labels = dso.train.images, dso.train.labels,

    do_training(model, images, labels, dso.test.images, dso.test.labels, train_iter=epochs, batch_size=bs, verb=verb,
                name=name, lr_sched=lr_sched)


def do_training(model, images, labels, test_images, test_labels, train_iter=10, batch_size=100, verb=True, vf=20,
                iter='', name="cifar10", lr_sched=None):
    os.makedirs("./csvs/", exist_ok=True)
    csv_path = "./csvs/{}-{}-supervised-{}-{}.csv".format(iter, str(len(labels)), time.strftime("%d-%m-%Y-%H%M%S"),
                                                          platform.uname()[1])
    print("saving losses at ", csv_path)
    test_generator = create_data_loader(test_images, test_labels, batch_size, is_train=False)
    train_generator = create_data_loader(images, labels, bs=batch_size, name=name)
    model, optimizer, criterion = model
    history = train_pbar(train_generator, model, optimizer, criterion, epochs=train_iter, testloader=test_generator,
                         print_freq=vf, verbose=verb, lr_sched=lr_sched, dev=device, csv_path=csv_path)

    return history, csv_path


def train_pbar(train_loader, model, optimizer, criterion, epochs=200, testloader=None, print_freq=50,  verbose=True,
               lr_sched=None, dev=None, csv_path=None):
    type_of_loss = type(criterion)
    if "CrossEntropy" in str(type_of_loss):
        ce = True
    else:
        ce = False
    batch_size = train_loader.batch_size
    log = []
    num_of_batches_per_epoch = np.ceil(len(train_loader.dataset) / batch_size)
    train_per_epoch = num_of_batches_per_epoch

    if verbose:
        loop_range = range(epochs)
    else:
        loop_range = tqdm(range(epochs))
    accuracy = Accuracy(task="multiclass",num_classes=10).to(dev)
    loss = MeanMetric().to(dev)
    model.to(dev)
    for epoch in loop_range:
        accuracy.reset()
        loss.reset()
        model.train()
        # if verbose:
        kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=epochs, width=8, always_stateful=True)
        # training
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(dev).float()
            outputs = model(inputs)
            targets = targets.to(dev).long()
            train_loss = criterion(outputs, targets)
            _, preds = torch.max(outputs.data, 1)
            acc = accuracy(preds, targets)
            loss.update(train_loss)
            # step
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if verbose:
                # Update after each batch
                if ce:
                    kbar.update(i, values=[("loss", loss.compute()), ("acc", accuracy.compute())])
                else:
                    kbar.update(i, values=[("loss", loss.compute())])
        if lr_sched is not None:
            lr_sched.step()

        if epoch % print_freq == 0 and verbose:
            # validation
            if testloader is not None:
                if ce:
                    val_loss, test_accuracy = evaluate(model, testloader, loss_fn=criterion, dev=dev)
                else:  # for triplet
                    test_accuracy = compute_embeddings_accuracy(model, train_loader.dataset.x, train_loader.dataset.y,
                                                                testloader.dataset.x, testloader.dataset.y)
                    val_loss = 0.

                if lr_sched is not None and lr_sched:
                    lr = lr_sched.get_last_lr()
                else:
                    lr = optimizer.param_groups[0]['lr']
            # Add validation metrics
                if ce:
                    kbar.add(1, values=[("val_loss", val_loss), ("val_acc", test_accuracy)])
                else:
                    kbar.add(1, values=[("val_knn_acc", test_accuracy)])
                if csv_path:
                    import csv
                    headers = ['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc']
                    row = [epoch+1, lr, loss.compute().item(), accuracy.compute().item(),
                          val_loss, test_accuracy]
                    # print(row)
                    with open(csv_path, 'a') as f:
                        # file_is_empty = os.stat(csv_path).st_size == 0
                        writer = csv.writer(f, lineterminator='\n')
                        if epoch == 0:
                            writer.writerow(headers)
                        writer.writerow(row)

                    tmp = pd.Series([epoch+1, lr, loss.compute().item(), accuracy.compute().item(),
                          val_loss, test_accuracy], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])
                    log.append(tmp)
                    # tmp.to_csv(csv_path, mode='a')
                    # log.to_csv(csv_path, index=False)
    return log


def start_self_learning(model, dso, dc, lt, i, mti, bs, logger, lr_sched=None):
    self_learning(model, dso, lt, logger,  i, dc.sp, mti, bs, lr_sched=lr_sched)


def pseudo_label_selection(imgs, pred_lbls, scores, orig_lbls, p=0.05):
    to_select = int(len(pred_lbls) * p)
    pseudo_images = []
    pseudo_labels = []
    orig_lbls_selected = []
    number_classes = np.unique(pred_lbls)
    per_class = to_select // len(number_classes)
    args = np.argsort(scores)
    indices = []
    for key in number_classes:  # for all classes
        selected = 0
        for index in args:
            if pred_lbls[index] == key:
                pseudo_images.append(imgs[index])
                pseudo_labels.append(pred_lbls[index])
                indices.append(index)
                orig_lbls_selected.append(orig_lbls[index])
                selected += 1
                if per_class == selected:
                    break
    orig_lbls_selected = np.array(orig_lbls_selected)
    pseudo_labels = np.array(pseudo_labels)
    if orig_lbls_selected.ndim > 1:
        acc = accuracy_score(np.argmax(orig_lbls_selected, 1), pseudo_labels) * 100.
    else:
        acc = accuracy_score(orig_lbls_selected, pseudo_labels) * 100.
    return np.array(pseudo_images), pseudo_labels, indices, acc


def assign_labels(model, train_labels, train_images, unlabeled_imgs, unlabeled_lbls, lt="cross-entropy"):

    if unlabeled_lbls.ndim > 1:  # if labels are one-hot encoded
        train_labels = np.argmax(train_labels, 1)
        unlabeled_lbls = np.argmax(unlabeled_lbls, 1)

    if lt == "cross-entropy":
        test_image_feat = get_network_output(model[0], unlabeled_imgs, lt)  # model.predict(unlabeled_imgs)
        pred_lbls = np.argmax(test_image_feat, 1)
        calc_score = np.max(test_image_feat, 1)
        calc_score = calc_score * -1.  # negate probs for same notion as distance
    else:   # for other loss functions
        # default to 1-NN distance as confidence score
        pred_lbls = []
        calc_score = []
        k = 1
        test_image_feat = get_network_output(model[0], unlabeled_imgs, lt)
        current_labeled_train_feat = get_network_output(model[0], train_images, lt)
        for j in range(len(test_image_feat)):
            search_feat = np.expand_dims(test_image_feat[j], 0)
            # calculate the sqeuclidean similarity and sort
            dist = cdist(current_labeled_train_feat, search_feat, 'sqeuclidean')
            rank = np.argsort(dist.ravel())
            pred_lbls.append(train_labels[rank[:k]])
            calc_score.append(dist[rank[0]])

    pred_lbls = np.array(pred_lbls)
    pred_lbls = pred_lbls.squeeze()
    pred_acc = accuracy_score(unlabeled_lbls, pred_lbls)*100.
    # print('predicted accuracy {:.2f} %'.format(pred_acc))
    calc_score = np.array(calc_score)
    pred_score = calc_score.squeeze()
    return pred_lbls, pred_score, pred_acc


def self_learning(model, mdso, lt,  logger, num_iterations=25, percentile=0.05, epochs=200, bs=100, lr_sched=None):

    # Initial labeled data
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # Initial unlabeled data
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels
    if lbls.ndim > 1:
        n_classes = len(np.unique(np.argmax(lbls, 1)))
    else:
        n_classes = len(np.unique(lbls))
    n_label = len(lbls)

    logger.info(template1.format(n_label, 100 * percentile, num_iterations))
    logger.info("i-th meta-iteration, unlabelled accuracy, pseudo-label accuracy,test accuracy")

    for i in range(num_iterations):
        print('=============== Meta-iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        # 1- training
        do_training(model, imgs, lbls, mdso.test.images, mdso.test.labels, epochs, bs, iter=str(i+1), lr_sched=lr_sched)
        # 2- predict labels and confidence score
        pred_lbls, pred_score, unlabeled_acc = assign_labels(model, mdso.train.labeled_ds.labels,
                                                             mdso.train.labeled_ds.images, unlabeled_imgs,
                                                             unlabeled_lbls, lt)
        # 3- select top p% pseudo-labels
        pseudo_label_imgs, pseudo_labels, indices_of_selected, pseudo_labels_acc = \
            pseudo_label_selection(unlabeled_imgs, pred_lbls, pred_score, unlabeled_lbls, percentile)
        # 4- merging new labeled for next loop iteration
        imgs = np.concatenate([imgs, pseudo_label_imgs], axis=0)
        if lbls.ndim > 1:  # if one-hot encoded
            pseudo_labels = np.eye(n_classes)[pseudo_labels]
        lbls = np.concatenate([lbls, pseudo_labels], axis=0)
        # 5- remove selected pseudo-labelled data from unlabelled data
        unlabeled_imgs = np.delete(unlabeled_imgs, indices_of_selected, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, indices_of_selected, 0)

        #####################################################################################
        #  print/save accuracies and other information
        test_acc = compute_accuracy(model[0], mdso.train.labeled_ds.images, mdso.train.labeled_ds.labels, mdso.test.images,
                                    mdso.test.labels, lt)
        print(template2.format(len(indices_of_selected), pseudo_labels_acc))
        print(template3.format(len(lbls) - n_label, len(unlabeled_lbls)))
        print("Acc: unlabeled: {:.2f} %,  test  {:.2f} %".format(unlabeled_acc, test_acc))
        # ith meta-iteration, unlabelled accuracy, pseudo-label accuracy, test accuracy
        logger.info("{},{:.2f},{:.2f},{:.2f}".format(i + 1, unlabeled_acc, pseudo_labels_acc, test_acc))
        #####################################################################################

    return imgs, lbls
