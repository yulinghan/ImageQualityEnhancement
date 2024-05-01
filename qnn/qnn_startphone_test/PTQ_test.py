import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import progressbar
import logging
from thop import profile
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.examples import mnist_torch_model
from aimet_torch.quantsim import QuantizationSimModel
from model.mnist_model import Net
from aimet_torch.cross_layer_equalization import equalize_model

logger = logging.getLogger('Eval')

if __name__ == '__main__':
    model = torch.load('model/model.pth').eval()
    input = torch.randn(1, 1, 28, 28)

    #自动检测模型定义中的BN层，并可以将其合并到相邻的卷积层中
    fold_all_batch_norms(model, input.shape)

    #跨层均衡(CLE)
    equalize_model(model, input.shape)
