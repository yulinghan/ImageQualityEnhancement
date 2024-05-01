import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])), batch_size=1, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
#np.array(example_data).tofile('pic/' + str(np.array(example_targets)[0]) + '.raw')

export_onnx_file = "model/Network.onnx"
session = onnxruntime.InferenceSession(export_onnx_file,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print(session.get_inputs()[0].name)

ort_input = {session.get_inputs()[0].name: example_data.cpu().detach().numpy()}
output = session.run(None, ort_input)
output = np.squeeze(output)
dst = np.where(output==np.max(output))
print('output:', output)
print('dst:', dst[0][0])
print('example_targets:', example_targets.numpy()[0])

