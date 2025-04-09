import torch
import torch.nn as nn

class ClampLayer(nn.Module):
    def __init__(self, min_value, max_value):
        super(ClampLayer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=self.min_value, max=self.max_value)