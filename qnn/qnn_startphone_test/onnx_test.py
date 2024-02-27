import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

export_onnx_file = "Network.onnx"
session = onnxruntime.InferenceSession(export_onnx_file,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
ort_input = {session.get_inputs()[0].name: example_data.cpu().detach().numpy()}
output = session.run(None, ort_input)
output = np.squeeze(output)
dst = np.where(output==np.max(output))
print('output:', output)
print('dst:', dst[0][0])
print('example_targets:', example_targets.numpy()[0])

