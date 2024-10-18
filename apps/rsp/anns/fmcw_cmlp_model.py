#
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class ComplexReLU(nn.Module):
    def forward(self, input):
        # 分别对实部和虚部应用ReLU
        real = F.relu(input.real)
        imag = F.relu(input.imag)
        return torch.complex(real, imag)

class FmcwCmlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 256, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(256, 64, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(64, 16, dtype=torch.complex64),
            ComplexReLU(),
            nn.Linear(16, 1, dtype=torch.complex64)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits