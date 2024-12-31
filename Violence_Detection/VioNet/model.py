import torch
import torch.nn as nn
#import n
import models.densenet as dn
from models.c3d import C3D
from models.densenet import densenet88, densenet121
from models.convlstm import ConvLSTM
from models.efficientnet_pytorch_3d.efficientnet3d import efficientnet3d

def VioNet_C3D(config):
    device = config.device
    model = C3D(num_classes=2).to(device)

    state_dict = torch.load('/home/bilel/1-demos/AnomalyDetection/AVSS2019/weights/C3D_Kinetics.pth')
    model.load_state_dict(state_dict)
    params = model.parameters()

    return model, params


def VioNet_ConvLSTM(config):
    device = config.device
    model = ConvLSTM(256, device).to(device)
    # freeze pretrained alexnet params
    for name, param in model.named_parameters():
        if 'conv_net' in name:
            param.requires_grad = False
    params = model.parameters()

    return model, params


def VioNet_densenet(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet121(num_classes=2,
                        sample_size=sample_size,
                        sample_duration=sample_duration).to(device)

    state_dict = torch.load('/home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/weights/DenseNet_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


# the model we finally adopted in DenseNet
def VioNet_densenet_lean(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet88(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    state_dict = torch.load('/home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/weights/DenseNetLean_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


# The model we adopted in ResNet Res3D
def VioNet_Res3D(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = resnet18(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    state_dict = torch.load('/home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/weights/r3d18_K_200ep.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params
#### changes
def VioNet_Res3D1(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = resnet18(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    state_dict = torch.load('/home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/weights/r3d18_K_200ep.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)
#### changes end
    return model, params
def VioNet_efficientnet_3d(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    #model = den
    #model = densenet88(num_classes=2,
    #                   sample_size=sample_size,
    #                   sample_duration=sample_duration).to(device)

    #state_dict = torch.load('/home/bilel/1-demos/AnomalyDetection/AVSS2019/weights/DenseNetLean_Kinetics.pth')
    #model.load_state_dict(state_dict)

    #params = dn.get_fine_tuning_params(model, ft_begin_idx)

    # model = efficientnet3d().to('cpu')
    model = efficientnet3d().to('cuda')
    params = model.parameters()
    return model, params
