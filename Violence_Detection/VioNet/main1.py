import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean, VioNet_efficientnet_3d
from dataset import VioDB
from config import Config

from spatial_transforms import Compose, ToTensor, Normalize
from spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from temporal_transforms import CenterCrop, RandomCrop
from target_transforms import Label, Video

from utils import Log


def main(config):
    # load model
    if config.model == 'c3d':
        model, params = VioNet_C3D(config)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config)
    elif config.model == 'efficientnet_3d':
        model, params = VioNet_efficientnet_3d(config) 
    elif config.model == 'VioNet_Res3D':
        model, params = VioNet_efficientnet_3d(config) 
    elif config.model == 'VioNet_Res3D1':
        model, params = VioNet_efficientnet_3d(config)
        # default densenet
    else:
        model, params = VioNet_densenet_lean(config)

    # dataset
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv

    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    temporal_transform = RandomCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    train_batch = config.train_batch

    train_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                       '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                       spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    val_batch = config.val_batch

    val_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                     '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                     spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    # make dir
    if not os.path.exists('./pth'):
        os.mkdir('./pth')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # log
    batch_log = Log(
        './log/{}_fps{}_{}_batch{}.log'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
        ), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        './log/{}_fps{}_{}_epoch{}.log'.format(config.model, sample_duration,
                                               dataset, cv),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        './log/{}_fps{}_{}_val{}.log'.format(config.model, sample_duration,
                                             dataset, cv),
        ['epoch', 'loss', 'acc'])

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)

    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    #optimizer=torch.optim.RMSprop(params, 
     #                             lr=0.01,
      #                            alpha=0.99,
       #                           eps=1e-08,
        #                          weight_decay=0,
         #                         momentum=0,
          #                        centered=False)
    optimizer = torch.optim.Adam(params=params,
                                  lr=0.01,
                                  betas=(0.9, 0.999),
                                  eps=1e-8,
                                  weight_decay=weight_decay,
                                  amsgrad=True)

    #optimizer = torch.optim.SGD(params=params,
         #                       lr=learning_rate,
        #                        momentum=momentum,
       #                         weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    acc_baseline = config.acc_baseline
    loss_baseline = 1

    for i in range(config.num_epoch):
        train(i, train_loader, model, criterion, optimizer, device, batch_log,
              epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, device,
                                val_log)
        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and
                                      val_loss < loss_baseline):
            torch.save(
                model.state_dict(),
                './pth/{}_fps{}_{}{}_{}_{:.4f}_{:.6f}.pth'.format(
                    config.model, sample_duration, dataset, cv, i, val_acc,
                    val_loss))
            acc_baseline = val_acc
            loss_baseline = val_loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        'efficientnet_3d',#'efficientnet_3d',  #efficientnet_3d,  c3d, convlstm, densenet, densenet_lean
        'drift',
        device='cuda',
        num_epoch=200,
        acc_baseline=0.50,
        ft_begin_idx=0,
        sample_duration=100,
        sample_size=(112, 112)
    )

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-4,
            'batch_size': 20
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'saudi_fight':{
            'lr': 1e-2,
            'batch_size': 20
        },
        'mix':{
            'lr': 1e-2,
            'batch_size': 1
        },
        'drift':{
            'lr':1e-2,
            'batch_size': 1
        }
    }

    for dataset in ['drift']: #['hockey', 'movie', 'vif', 'saudi_fight','mix']
        config.dataset = dataset
        config.train_batch = configs[dataset]['batch_size']
        config.val_batch = configs[dataset]['batch_size']
        config.learning_rate = configs[dataset]['lr']
        main(config)
        # 5 fold cross validation
        #for cv in range(1, 6):
        #    config.num_cv = cv
        #    main(config)
