import torch
import torch.nn as nn

class CastLayer(nn.Module):
    def __init__(self, dtype, round_before_cast=False):
        super(CastLayer, self).__init__()
        self.dtype = dtype
        self.round_before_cast = round_before_cast

    def forward(self, x):
        if self.round_before_cast:
            x = torch.round(x)  # Apply rounding if flag is enabled
        return x.to(self.dtype)