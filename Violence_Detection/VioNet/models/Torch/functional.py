
# import pytorch
import torch
import torch.nn.functional as F

def mish(input):
    
    return input * torch.tanh(F.softplus(input))

