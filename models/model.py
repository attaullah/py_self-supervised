import sys
import torch.nn as nn
from .wrn import WideResNet
from losses.triplet import triplet_loss
from torch.nn import CrossEntropyLoss
# from pytorch_metric_learning import losses, miners, reducers, regularizers
import torch
import torchvision.models as vision_models
from torch.optim import SGD, Adam, RMSprop
import torch.nn.functional as F
from data_utils.transforms import transform


def get_network(arch, size=32, channels=3, num_classes=-1, weights=False, d_set="cifar10"):
    es = -1
    if "svhn" in d_set:
        transforms = transform(False,True,False)
    else:
        transforms = transform()
    if 'simple' in arch:
        from models import SimpleModel
        base = SimpleModel(size=size, channels=channels)
    elif 'ssdl' in arch:
        from .models import SSDL
        base = SSDL(num_classes=num_classes, transform_fn=transforms)
    elif 'my' in arch:
        from models import MyModel
        base = MyModel(channels)
    elif 'resnet18' in arch:
        es = 64
        base = vision_models.resnet18(pretrained=weights)  #
    elif 'resnet34' in arch:
        es = 64
        base = vision_models.resnet34(pretrained=weights)  #
    elif 'resnet50' in arch:
        es = 64
        base = vision_models.resnet50(pretrained=weights)  #
    elif 'vit' in arch:
        from .vit import ViT
        es = 0
        base = ViT(image_size=size,patch_size=8,num_classes=num_classes,dim=256,depth=8,heads=6,mlp_dim=256,dim_head=256,dropout=0.1, emb_dropout=0.1)
    elif 'vgg16' in arch:
        base = vision_models.vgg16(pretrained=weights)
        num_features = base.classifier[6].in_features
        features = list(base.classifier.children())[:-1]  # Remove last layer
        # features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with some number of  outputs
        base.classifier = nn.Sequential(*features)  # Replace the model classifier
        es = 256
    elif 'wrn' in arch:
        es = -1
        dwr = arch.split('-')  # 'wrn-d-w'
        depth, widen_factor = int(dwr[1]), int(dwr[2])
        if weights:
            base = WideResNet(depth=depth, widen_factor=widen_factor, num_classes=1000, transform_fn=transforms)
            if depth == 28 and widen_factor == 2:
                base.load_state_dict(torch.load('./weights/wrn-28-2/imagenet32/ep-50-acc-41.79-weights.pth'))
                base.fc.out_features = num_classes
        else:
            base = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, transform_fn=transforms)
    else:
        base = WideResNet(depth=28, widen_factor=2, num_classes=1000)
        if weights:
            base.load_state_dict(torch.load('./weights/imagenet32/wrn-28-2/ep-50-acc-41.79-weights.pth'))
            # base.fc.out_features = num_classes
    return base, es


def get_optimizer(opt, lr, params,  wd=5e-4):
    if opt.lower() == 'adam':
        return Adam(params, lr)
    elif opt.lower() == 'adamw':
        return Adam( params, lr, weight_decay=wd)
    elif opt.lower() == 'rmsprop':
        return RMSprop( params, lr,)
    elif opt.lower() in ['sgd', 'sgdn']:
        return SGD(params, lr, momentum=0.9)
    else:
        print('optimizer not implemented')
        sys.exit(1)


def get_lr_scheduler(optim, name="cosine", n_epochs=200):
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
        print("Using CosineLR scheduler")
    else: 
        print("Not using LR scheduler")
    return sched


def get_model(arch, data_config,  weights=False, loss_type="cross-entropy", opt="adam", lr=1e-3, lr_sched=None, margin=1.):

    base_model, es = get_network(arch, data_config.size, data_config.channels, num_classes=data_config.nc,
                                 weights=weights, d_set=data_config.name)  # flags.w
    model = nn.Sequential()
    model.add_module("base", base_model)
    # print("Get model:: ", data_config.nc, es)
    if es > 0:
        model.add_module("dropout", nn.Dropout(0.2))
        model.add_module("embeddings", nn.Linear(128, es))
    # else:
    #     es = 64

    if loss_type == 'triplet':
        criterion = triplet_loss
    else:
        criterion = CrossEntropyLoss()
        # print("es value for ViT ", es)
        if es == 0:  # special case for ViT
            model = base_model
        else:
            es = 64
            if base_model.fc.out_features != data_config.nc:
                model.add_module("fc", nn.Linear(es, data_config.nc))
        # model.add_module("softmax", nn.Softmax(dim=1))

    params = model.parameters()
    optimizer = get_optimizer(opt, lr, params)
    if lr_sched is not None:
        lr_sched = get_lr_scheduler(optimizer, name=lr_sched, n_epochs=200)  
    return model, optimizer, criterion, lr_sched
