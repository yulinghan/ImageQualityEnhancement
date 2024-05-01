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

logger = logging.getLogger('Eval')

def pass_calibration_data(sim_model, use_cuda):
    batch_size = 1000
    data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=True)

    device = torch.device('cpu')

    samples = 1000
    batch_cntr = 0
    for batch_idx, (input_data, target_data) in enumerate(data_loader):
        inputs_batch = input_data.to(device)
        sim_model(inputs_batch)

        batch_cntr += 1
        if (batch_cntr * batch_size) > samples:
            break

def evaluate_func(model, iter_num):
    batch_size = 1000
    data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=True)

    device = torch.device('cpu')

    model = model.to(device)
    model = model.eval()

    batch_cntr = 0
    correct = 0
    with progressbar.ProgressBar(max_value=iter_num) as progress_bar:
        with torch.no_grad():
            for input_data, target_data in data_loader:
                inputs_batch = input_data.to(device)
                target_batch = target_data.to(device)

                predicted_batch = model(inputs_batch)
                predicted_batch = predicted_batch.data.max(1, keepdim=True)[1]
                correct += predicted_batch.eq(target_batch.data.view_as(predicted_batch)).sum().item()

                progress_bar.update(batch_cntr)

                batch_cntr += 1
                if batch_cntr > iter_num:
                    break

    correct /= (batch_cntr * batch_size)

    return correct

if __name__ == '__main__':
    model = torch.load('model/model.pth').eval()
    input = torch.randn(1, 1, 28, 28)

    #自动检测模型定义中的BN层，并可以将其合并到相邻的卷积层中
    fold_all_batch_norms(model, (1, 1, 28, 28))

    #创建量化仿真模型
    sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28),
        default_output_bw=8, default_param_bw=8)

    #生成量化仿真操作的比例/偏移参数
    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
        forward_pass_callback_args=None)

    #仿真模型保存
    sim.export('./', filename_prefix='model/QAT_resnet', dummy_input=input.cpu())

    iter_num = 3
    quantized_accuracy1 = evaluate_func(model, iter_num)
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)
    logger.info('Avg accuracy Top: %f on validation Dataset', quantized_accuracy1)

    model = torch.load('model/QAT_resnet.pth')
    quantized_accuracy2 = evaluate_func(model, iter_num)
    logger.info('Avg accuracy Top: %f on validation Dataset', quantized_accuracy2)
    #flops, params = profile(sim.model, inputs=(input, ))
    #print(flops, params)

