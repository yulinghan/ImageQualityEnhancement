import onnxruntime
import numpy as np
import onnx_tool
import torch
import torchvision
from torch.utils.data import DataLoader

#model_path = "model/Network.onnx"
model_path = "model/QAT_resnet.onnx"

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])), batch_size=1, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

ort_input = {'input': example_data.cpu().detach().numpy()}
onnx_tool.model_profile(model_path, ort_input)
