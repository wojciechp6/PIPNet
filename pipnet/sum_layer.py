import torch
import torch.nn as nn

class SumLayer(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.sum(input, dim=self.dim)