"""Contains the code for the exemplar cnn sub task."""
import time
# https://github.com/Wuschelbueb/AML19-SelfSupervised/blob/master/exemplar_cnn.py
# https://github.com/KevinMusgrave/pytorch-metric-learning
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
from utils2 import linear_test_accuracy
from custom_utils import create_data_loader
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter, \
    RandomResizedCrop, RandomRotation, RandomAffine, Compose, Resize, ToTensor, \
    Normalize, ToPILImage, RandomVerticalFlip
DEVICE = torch.device("cuda")


def log_accuracy(flags,dso, model, log,  extra=''):
    if flags.so:
        type_str = 'supervised'
        if 'test' in extra.lower():
            dl = create_data_loader(dso.test.images, dso.test.labels, is_train=False, shuffle=False)
        else:
            dl = create_data_loader(dso.train.unlabeled_ds.images, dso.train.unlabeled_ds.labels, is_train=False)
        acc = evaluate_model(dl, model)
    else:
        type_str = flags.lbl
        if 'test' in extra.lower():
            acc = linear_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels,
                                  dso.test.images, dso.test.labels, lbling=flags.lbl)
        else:
            acc = linear_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels,
                                  dso.train.unlabeled_ds.images, dso.train.unlabeled_ds.labels, lbling=flags.lbl)
    log.info('{} {} accuracy  is {:.2f} %'.format(extra, type_str, acc))


def evaluate_model(testloader, net, use_cuda=True):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    from torch.autograd import Variable
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs,_ = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
    return acc.numpy()


def supervised_accuracy(images, labels, net, use_cuda=True):
    testloader = create_data_loader(images, labels, is_train=False, shuffle=False)
    return evaluate_model(testloader,net)


def linear_accuracy(model, l_imgs, l_lbls, t_imgs, t_lbls, lbling='lda',ret_labels=False):
    labels, accuracy = linear_test_accuracy(predict(l_imgs, model), l_lbls, predict(t_imgs, model), t_lbls, lbling,
                                            verbose=False, fn=True)
    if ret_labels:
        return accuracy, labels
    return accuracy


def predict(images, net, use_cuda=True):
    net.eval()
    # net.training = False
    testloader = create_data_loader(images, is_train=False, shuffle=False)
    predictions = list()
    from torch.autograd import Variable
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testloader):
            if use_cuda:
                inputs = inputs.to(DEVICE)
            inputs = Variable(inputs)
            outputs,_ = net(inputs)
            predictions.append(outputs.cpu().data.numpy())

        predictions = np.vstack(predictions)

    return predictions


def transform_image(image, transformation):
    """Randomly transforms one image."""
    image = image.cpu()
    transform = ToPILImage()
    img = transform(image)

    if transformation == 0:
        return horizontal_flip(img)
    if transformation == 1:
        return vertical_flip(img)
    if transformation == 2:
        return random_rotation(img, 90)
    if transformation == 3:
        return random_rotation(img, 180)
    if transformation == 4:
        return random_rotation(img, 270)
    if transformation == 5:
        return random_rotation(img, 0)


def horizontal_flip(image):
    """Flip image horizontally."""
    transform = Compose([
        RandomHorizontalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomHorizontalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def vertical_flip(image):
    """Flip image horizontally."""
    transform = Compose([
        RandomVerticalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomVerticalFlip(p=1.0),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def random_crop(image):
    """Crop Image."""
    transform = Compose([
        RandomCrop((20, 20)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomCrop((20, 20)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def color_jitter(image):
    """Apply color jitter."""
    transform = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def random_resized_crop(image):
    """Randomly resize and crop image."""
    transform = Compose([
        RandomResizedCrop(40, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomResizedCrop(40, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def random_rotation(image, degrees):
    """Randomly rotate image."""
    transform = Compose([
        RandomRotation(degrees),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomRotation(degrees),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def random_affine_transformation(image):
    """Applies a random affine transformation to the image."""
    transform = Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    transform_deepfashion = Compose([
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10),
        Resize(32),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = None
    if image.mode == 'L':
        img = transform(image)
    if image.mode == 'RGB':
        img = transform_deepfashion(image)
    img = img.to(DEVICE)
    return img


def self_mb(images):
    images_transformed, labes_transformed = [], []
    for index, img in enumerate(images):
        transformed_imgs = [
            # img,
            transform_image(img, 0),
            transform_image(img, 1),
            transform_image(img, 2),
            transform_image(img, 3),
            transform_image(img, 4),
            transform_image(img, 5),
        ]
        # len_transformed_imgs = len(transformed_imgs)
        transformed_labels = torch.LongTensor(range(len(transformed_imgs)))  # .tolist()
        stack = torch.stack(transformed_imgs, dim=0)

        images_transformed.append(stack)
        labes_transformed.append(transformed_labels)
        # image_index += 1

    images = torch.cat(images_transformed, dim=0).to(DEVICE)
    labels = torch.cat(labes_transformed, dim=0).to(DEVICE)
    return images, labels


def finetune_triplet(train_loader, model, optimizer, criterion,  epochs=200, testloader=None, print_freq=50,
                  verbose=True, lr_sched=None):
    import pkbar
    batch_size = 128
    if verbose:
        loop_range = range(epochs)
    else:
        loop_range = tqdm(range(epochs))
    log = []
    csv = False
    num_of_batches_per_epoch = np.ceil(len(train_loader.dataset)/ batch_size)
    train_per_epoch = num_of_batches_per_epoch

    for epoch in loop_range:
        model.train()
        if verbose:
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=epochs, width=8, always_stateful=True)
        # training
        for i, (inputs, targets) in enumerate(train_loader):
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            embeddings, _ = model(inputs)
            # indices_tuple = mining_func(embeddings, targets)
            train_loss = criterion(embeddings, targets)  # indices_tuple)

            # step
            train_loss.backward()
            optimizer.step()
            if verbose:
                ############################# Update after each batch ##################################
                kbar.update(i, values=[("train loss", train_loss),])  # ("train_accuracy", train_accuracy)
                ########################################################################################
        # lr_sched.step()

        if (epoch+1) % print_freq == 0 and verbose:
            # validation
            if testloader is not None:
                val_loss = 0
                for j, (inputs, targets) in enumerate(testloader):
                    targets = targets.cuda(non_blocking=True)
                    inputs= inputs.cuda(non_blocking=True)
                    outputs,_  = model(inputs)
                    targets = targets.type(torch.LongTensor).to(device)
                    val_loss = criterion(outputs, targets)

                test_accuracy = linear_accuracy(model, train_loader.dataset.x,train_loader.dataset.y,
                                                testloader.dataset.x,testloader.dataset.y)

            ################################ Add validation metrics ###################################
                kbar.add(1, values=[("val_loss", val_loss), ("val_accuracy", test_accuracy),
                                    ("lr",optimizer.param_groups[0]['lr'])])


def start_joint_train(flags, model, optimizer, criterion, dso):

    train_dl1 = create_data_loader(dso.train.labeled_ds.images, dso.train.labeled_ds.labels)
    if flags.semi:
        x = np.concatenate((dso.train.labeled_ds.images,dso.train.unlabeled_ds.images))
    else:
        x = dso.train.labeled_ds.images
    train_dl2 = create_data_loader(x)
    dl_test = create_data_loader(dso.test.images, dso.test.labels, shuffle=False, is_train=False)

    if flags.so:
        joint_train(model, optimizer, criterion, train_dl1,train_dl2, flags.ii, dl_test, flags.vf, verbose=flags.verbose)
    else:
        loss_func = criterion
        joint_train_triplet(model, optimizer, loss_func, train_dl1, train_dl2, flags.ii, dl_test, flags.vf,
                    verbose=flags.verbose)


def joint_train(model, optimizer, loss_fn, train_loader1, train_loader2, num_epochs=20,  test_dl=None, print_freq=50,
                verbose=True):
    """Train the model"""

    train_losses, train_accuracies = [], []
    r_train_losses, r_train_accuracies = [], []
    batch_size = 128
    if verbose:
        loop_range = range(num_epochs)
    else:
        loop_range = tqdm(range(num_epochs))
    import pkbar
    num_of_batches_per_epoch = np.ceil(len(train_loader2.dataset) / batch_size)
    train_per_epoch = num_of_batches_per_epoch
    from itertools import cycle
    for epoch in loop_range:
        # scheduler.step()
        model.train()

        running_loss = []
        running_corrects = 0
        r_running_loss = []
        r_running_corrects = 0
        total_count = 0
        r_total_count = 0
        if verbose:
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=num_epochs, width=8, always_stateful=True)

        i = 0
        for item1, item2 in zip(train_loader2, cycle(train_loader1)):
            image_batch1 = item1
            image_batch2, labels2 = item2
            labels2 = labels2.cuda(non_blocking=True)
            image_batch2 = image_batch2.cuda(non_blocking=True)
            labels2 = labels2.type(torch.LongTensor).to(DEVICE)
            images = image_batch1.to(DEVICE)

            images, labels = self_mb(images)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            _, r_outputs = model(images)
            outputs, _ = model(image_batch2)
            _, preds = torch.max(outputs.data, 1)
            _, r_preds = torch.max(r_outputs.data, 1)
            loss1 = loss_fn(outputs, labels2)
            r_loss = loss_fn(r_outputs, labels)
            loss= loss1 + r_loss
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            loss1_value = loss1.item()
            rloss_value = r_loss.item()
            running_loss.append(loss1_value)
            r_running_loss.append(rloss_value)
            running_corrects += torch.sum(preds == labels2.data).to(torch.float32)
            r_running_corrects += torch.sum(r_preds == labels.data).to(torch.float32)
            total_count += preds.size(0)
            r_total_count += r_preds.size(0)

            i+=1
            if verbose:
                kbar.update(i,
                            values=[("sup train_loss", loss1_value), (" sup train_accuracy", (100.0 * running_corrects) / total_count),
                                    ("self train_loss", rloss_value),
                                    (" self train_accuracy", (100.0 * r_running_corrects) / r_total_count)])

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append((100.0 * running_corrects) / total_count)  # (len(train_loader2.dataset)))
        r_train_losses.append(np.mean(np.array(r_running_loss)))
        r_train_accuracies.append((100.0 * r_running_corrects) / total_count)  # (len(train_loader2.dataset)))

        if (epoch+1) % print_freq == 0 and verbose:
            # validation
            if test_dl is not None:
                val_loss, test_accuracy = test(model, loss_fn, test_dl)

            ################################ Add validation metrics ###################################
                kbar.add(1, values=[("val_loss", val_loss), ("val_accuracy", test_accuracy),#("lr",lr_sched.get_lr()[0])
                         ])
    return model, train_losses, train_accuracies


def joint_train_triplet(model, optimizer, loss_fn, train_loader1, train_loader2, num_epochs=20,
                        test_dl=None, print_freq=50, verbose=True):
    """Train the model"""

    train_losses, train_accuracies = [], []
    r_train_losses, r_train_accuracies = [], []
    batch_size = 128
    if verbose:
        loop_range = range(num_epochs)
    else:
        loop_range = tqdm(range(num_epochs))
    import pkbar
    num_of_batches_per_epoch = np.ceil(len(train_loader2.dataset) / batch_size)
    train_per_epoch = num_of_batches_per_epoch
    self_loss = CrossEntropyLoss()
    from itertools import cycle
    for epoch in loop_range:
        # scheduler.step()
        model.train()

        running_loss = []
        r_running_loss = []
        r_running_corrects = 0

        if verbose:
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=num_epochs, width=8, always_stateful=True)

        i = 0
        for item1, item2 in zip(train_loader2, cycle(train_loader1)):
            image_batch1 = item1
            image_batch2, labels2 = item2
            labels2 = labels2.cuda(non_blocking=True)
            image_batch2 = image_batch2.cuda(non_blocking=True)
            labels2 = labels2.type(torch.LongTensor).to(DEVICE)
            images = image_batch1.to(DEVICE)

            images, labels = self_mb(images) # self-supervised mini batch

            optimizer.zero_grad()
            # forward
            embeddings, _ = model(image_batch2)
            # indices_tuple = mining_func(embeddings, labels2)
            loss1 = loss_fn(embeddings, labels2)  # , indices_tuple)
            _, r_outputs = model(images)
            # print('!!!!! test   ', labels.size(), '  ', r_outputs.size())
            _, r_preds = torch.max(r_outputs.data, 1)
            r_loss = self_loss(r_outputs, labels)
            loss = loss1 + r_loss
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss1.item())
            r_running_loss.append(r_loss.item())
            r_running_corrects += torch.sum(r_preds == labels.data).to(torch.float32)
            i+=1

        train_losses.append(np.mean(np.array(running_loss)))
        r_train_losses.append(np.mean(np.array(r_running_loss)))
        r_train_accuracies.append((100.0 * r_running_corrects) / (len(train_loader2.dataset)))
        if verbose:
            kbar.update(i, values=[("sup train_loss", train_losses[-1]), ("self train_loss", r_train_losses[-1]),
                                   (" self train_accuracy", r_train_accuracies[-1])])
        if (epoch+1) % print_freq == 0 and verbose:
            if test_dl is not None:
                val_loss,  test_accuracy = test(model, loss_fn, test_dl)
                test_accuracy = linear_accuracy(model, train_loader1.dataset.x, train_loader1.dataset.y,
                                                    test_dl.dataset.x, test_dl.dataset.y)
            ################################ Add validation metrics ###################################
                kbar.add(1, values=[("val_loss", val_loss), ("val_accuracy", test_accuracy),#("lr",lr_sched.get_lr()[0])
                         ])
    return model, train_losses, train_accuracies

def start_fine_tune(model, optimizer, loss, dso, flags):
    x = dso.train.labeled_ds.images
    y = dso.train.labeled_ds.labels
    train_dl = create_data_loader(x, y)
    print(' fine tune labld len', len(x))
    dl_test = create_data_loader(dso.test.images, dso.test.labels, is_train=False, shuffle=False)
    if flags.so:
        fine_tune(model, optimizer,loss, train_dl, flags.ii, dl_test, flags.verbose, flags.vf)
    else:
        loss, mining_func = loss
        finetune_triplet(train_dl, model, optimizer, loss, mining_func, flags.ii, dl_test, flags.verbose, flags.vf)


def start_pre_train(model, optimizer, loss, dso, flags):
    if flags.semi:
        x = np.concatenate((dso.train.labeled_ds.images, dso.train.unlabeled_ds.images))
    else:
        x = dso.train.labeled_ds.images
    train_dl = create_data_loader(x)
    pre_train(model, optimizer, loss, train_dl, flags.pti, flags.verbose, flags.bs)


def pre_train(model, optimizer, loss_fn, train_loader, num_epochs=20, verbose=False, batch_size=128):
    """Train the model"""

    train_losses, train_accuracies = [], []

    if verbose:
        loop_range = range(num_epochs)
    else:
        loop_range = tqdm(range(num_epochs))
    import pkbar
    num_of_batches_per_epoch = np.ceil(len(train_loader.dataset) / batch_size)
    train_per_epoch = num_of_batches_per_epoch
    from itertools import cycle
    for epoch in loop_range:
        # scheduler.step()
        model.train()

        running_loss = []
        running_corrects = 0

        len_transformed_imgs = 0.0
        image_index = 0
        if verbose:
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=num_epochs, width=8, always_stateful=True)

        i = 0
        for item1 in train_loader:
            image_batch1 = item1
            images = image_batch1.to(DEVICE)
            # images_transformed, labes_transformed = [], []
            #
            # for index, img in enumerate(images):
            #     transformed_imgs = [
            #         transform_image(img, 0),
            #         transform_image(img, 1),
            #         transform_image(img, 2),
            #         transform_image(img, 3),
            #         transform_image(img, 4),
            #         transform_image(img, 5),
            #     ]
            #     len_transformed_imgs = len(transformed_imgs)
            #     transformed_labels = torch.LongTensor(range(len(transformed_imgs))) # .tolist()
            #     stack = torch.stack(transformed_imgs, dim=0)
            #     images_transformed.append(stack)
            #     labes_transformed.append(transformed_labels)
            #     image_index += 1
            #
            # images = torch.cat(images_transformed, dim=0).to(DEVICE)
            # labels = torch.cat(labes_transformed, dim=0).to(DEVICE)
            images, labels = self_mb(images)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            _, r_outputs = model(images)
            _, r_preds = torch.max(r_outputs.data, 1)
            r_loss = loss_fn(r_outputs, labels)
            # backward + optimize only if in training phase
            r_loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(r_loss.item())
            running_corrects += torch.sum(r_preds == labels.data).to(torch.float32)
            i+=1

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append((100.0 * running_corrects) / (len_transformed_imgs * len(train_loader.dataset)))

        if verbose:
            kbar.update(i, values=[("self train loss", train_losses[-1]), (" self train_accuracy", train_accuracies[-1]),
                                   ])

    return model, train_losses, train_accuracies


def calculate_test_accuracy(model, dso, supervised=True):
    if supervised:
        return  supervised_accuracy(dso.test.images, dso.test.labels, model, use_cuda=True)
    return linear_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images,
                           dso.test.labels)


def double_dataloader():
    dataloaders1 = DataLoader(DummyDataset(0, 100), batch_size=10, shuffle=True)
    dataloaders2 = DataLoader(DummyDataset(0, 200), batch_size=10, shuffle=True)
    num_epochs = 10

    for epoch in range(num_epochs):
        dataloader_iterator = iter(dataloaders1)

        for i, data1 in enumerate(dataloaders2):
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloaders1)
                data2 = next(dataloader_iterator)

        do_cool_things()


def train_step(model, optimizer, loss_fn, train_loader,  epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def fine_tune(model, optimizer,loss_fn,  train_loader, num_epochs=200, val_loader=None, verbose=True,print_freq=50):
    """Fine tune the model"""
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    import pkbar
    if verbose:
        loop_range = range(num_epochs)
    else:
        loop_range = tqdm(range(num_epochs))
    batch_size = 128
    num_of_batches_per_epoch = np.ceil(len(train_loader.dataset) / batch_size)
    train_per_epoch = num_of_batches_per_epoch

    for epoch in loop_range:
        # scheduler.step()
        model.train()
        running_loss = []
        running_corrects_train = 0.0
        if verbose:
            kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=num_epochs, width=8, always_stateful=True)

        for images, labels in train_loader:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.type(torch.LongTensor).to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs, _ = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_corrects_train += torch.sum(preds == labels.data).to(torch.float32)

        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append(100.0 * running_corrects_train / len(train_loader.dataset))
        if verbose:
            kbar.update(epoch, values=[("train_loss", train_losses[-1]), ("train_accuracy", train_accuracies[-1])])
        if (epoch + 1) % print_freq == 0 and verbose:
            val_loss, val_accuracy = test(model, loss_fn, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            kbar.add(1, values=[("val_loss", val_loss), ("val_accuracy", val_accuracy),])

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def test(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target.type(torch.LongTensor).to(DEVICE)
            output, _ = model(data) # skip pseudo label output
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, 100. * correct / len(test_loader.dataset)


def meta_learning(model, optimizer, criterion, dso, flags, logging):
    self_learning_modular(model, optimizer, criterion, dso, labeled=flags.nl, num_iterations=flags.i,
                          percentile=flags.sp, network_updates=flags.mti, bs=flags.bs, logging=logging,
                          verbose=flags.verbose, supervised=flags.so, joint=flags.joint)


def do_training(model, optimizer, criterion, imgs, lbls, dso, network_updates, verb, semi=True, joint=True,
                supervised=True):
    dl = create_data_loader(imgs, lbls, shuffle=True)
    dl_test = create_data_loader(dso.test.images, dso.test.labels, shuffle=False, is_train=False)
    if semi:
        x = np.concatenate((dso.train.labeled_ds.images,dso.train.unlabeled_ds.images))
    else:
        x = dso.train.labeled_ds.images
    train_dl2 = create_data_loader(x)
    if joint:
        if supervised:
            joint_train(model, optimizer, criterion, dl, train_dl2, network_updates, dl_test, verbose=verb)
        else:
            loss_func = criterion
            joint_train_triplet(model, optimizer, loss_func, dl, train_dl2, network_updates, dl_test, verbose=verb)
    else:
        if supervised:
            fine_tune(model, optimizer, criterion, dl, network_updates, dl_test, verb, print_freq=50)
        else:
            loss_func = criterion
            finetune_triplet(dl, model, optimizer, loss_func, network_updates, dl_test, 50, verb)
    return 0

def get_confidence(model, mdso, unlabeled_imgs, unlabeled_lbls, lbl='knn', so=True):
        # te_acc_nn = 0
        # true_test_labels = unlabeled_lbls
        train_labels = mdso.train.labeled_ds.labels
        if unlabeled_lbls.ndim > 1:
            train_labels = np.argmax(train_labels, 1)
            unlabeled_lbls = np.argmax(unlabeled_lbls, 1)
        # LDA or other linear models
        if lbl != 'knn':
            clf = get_linear_clf()
            test_image_feat = get_network_output(model, unlabeled_imgs)
            current_labeled_train_feat = get_network_output(model, mdso.train.labeled_ds.images)
            clf.fit(current_labeled_train_feat, train_labels)
            pred_lbls = clf.predict(test_image_feat)
            if lbl == 'lda':
                calc_score = clf.decision_function(test_image_feat)
            else:
                calc_score = clf.predict_proba(test_image_feat)
            calc_score = np.max(calc_score, 1)
            calc_score = calc_score * -1.  # negate probs for same notion as distance
        elif so:
            test_image_feat = predict(unlabeled_imgs, model)
            # print('!!test ', test_image_feat.shape)
            pred_lbls = np.argmax(test_image_feat, 1)
            calc_score = np.max(test_image_feat, 1)
            calc_score = calc_score * -1.  # negate probs for same notion as distance
        else:  # default to KNN with k=1 distance as confidence score
            pred_lbls = []
            calc_score = []
            k = 1
            test_image_feat = predict(unlabeled_imgs, model)
            current_labeled_train_feat = predict(mdso.train.labeled_ds.images, model)
            for j in range(len(test_image_feat)):
                search_feat = np.expand_dims(test_image_feat[j], 0)
                # calculate the sqeuclidean similarity and sort
                dist = cdist(current_labeled_train_feat, search_feat, 'sqeuclidean')
                rank = np.argsort(dist.ravel())
                pred_lbls.append(train_labels[rank[:k]])
                calc_score.append(dist[rank[0]])

        pred_lbls = np.array(pred_lbls)
        labels = pred_lbls.squeeze()
        print('predicted accuracy {:.2f} %'.format(accuracy_score(unlabeled_lbls, pred_lbls) * 100.))
        calc_score = np.array(calc_score)
        score = calc_score.squeeze()
        return labels, score


def select_top_k(imgs, lbls, scores, orig_lbls=None, k=4000):
    sampled_images = []
    sampled_labels = []
    orig_lbls_selected = []
    number_classes = np.unique(lbls)
    per_class = k // len(number_classes)
    args = np.argsort(scores)
    indices = []
    for key in number_classes:
        selected = 0
        for index in args:
            if lbls[index] == key:
                sampled_images.append(imgs[index])
                sampled_labels.append(lbls[index])
                indices.append(index)
                if orig_lbls is not None:
                    orig_lbls_selected.append(orig_lbls[index])
                selected += 1
                if per_class == selected:
                    break
    if orig_lbls is not None:
        orig_lbls_selected = np.array(orig_lbls_selected)
        sampled_labels = np.array(sampled_labels)
        print('selection accuracy  {:.2f}%'.format(accuracy_score(orig_lbls_selected, sampled_labels) * 100.))
    return np.array(sampled_images), sampled_labels, indices


def self_learning_modular(model, optimizer, criterion, mdso, labeled, num_iterations, percentile, network_updates=200,
                          bs=128,logging=None, verbose=False, supervised=True, joint=True):
    # predicted_labeled_imgs = mdso.train.labeled_ds.images
    predicted_labeled_lbls = mdso.train.labeled_ds.labels
    predicted_original_lbls = mdso.train.labeled_ds.labels
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # predicted_labeled_imgs_inds = []
    n_label = labeled
    template = 'overall accuracy of predicted total {} is {:.2f}%'
    template2 = 'total selected based on percentile {} having accuracy {:.2f}%'
    template3 = "Labeled= {},LLGC= {} selection={}% iterations= {}"
    template4 = 'Length of predicted {}, labeled  {},  unlabeled {}'
    pred_str = 'supervised ' # if .so else .lbl
    # percentile = .sp if percentile == 0. else percentile
    logging.info(template3.format(labeled, False, 100 * percentile, num_iterations))
    for i in range(num_iterations):
        print('=============== iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        adaptive_network_updates = 2000 * bs / len(lbls)
        do_training(model, optimizer, criterion, imgs, lbls, mdso, network_updates,
                    verb=verbose, supervised=supervised, joint=joint)

        pred_lbls, calc_score = get_confidence(model, mdso, unlabeled_imgs, unlabeled_lbls, so=supervised)
        unlab_acc = accuracy_score(unlabeled_lbls, pred_lbls) * 100.
        print('==>', pred_str, ' predicting for  lbls ', unlabeled_lbls.shape, '=', np.round(unlab_acc, 2))
        to_select = int(len(unlabeled_lbls) * percentile)
        selected_imgs, selected_lbls, ind = select_top_k(unlabeled_imgs, pred_lbls, calc_score, unlabeled_lbls, to_select)
        # ind = np.argsort(calc_score)
        total_selected = len(ind)  # int(len(unlabeled_lbls) * percentile)

        original_lbls_selected = unlabeled_lbls[ind]
        select_acc = accuracy_score(original_lbls_selected, selected_lbls) * 100.  # pred_lbls[ind]
        print(template2.format(total_selected, select_acc))
        # selection based on top k func
        new_labeled_imgs = unlabeled_imgs[ind]  # selected_imgs
        # new_labeled_imgs = new_labeled_imgs.squeeze()
        new_labeled_lbls = pred_lbls[ind]  # [ind,0]

        imgs = np.concatenate([imgs, new_labeled_imgs], axis=0)
        lbls = np.concatenate([lbls, new_labeled_lbls], axis=0)
        labeled = len(imgs)
        predicted_labeled_lbls = np.concatenate((predicted_labeled_lbls, new_labeled_lbls))
        predicted_original_lbls = np.concatenate((predicted_original_lbls, original_lbls_selected))
        # overall accuracy of predicted
        acc = accuracy_score(predicted_original_lbls[n_label:], predicted_labeled_lbls[n_label:]) * 100.
        print(template.format(len(predicted_labeled_lbls[n_label:]), acc))
        test_acc = calculate_test_accuracy(model, mdso, supervised)
        # remove selected
        unlabeled_imgs = np.delete(unlabeled_imgs, ind, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, ind, 0)
        print(template4.format(labeled - n_label, labeled, len(unlabeled_lbls)))
        logging.info("{},{:.2f},{:.2f}".format(i+1, unlab_acc, test_acc))  # ith,unlab,test accuracy

    return imgs, lbls, predicted_original_lbls
