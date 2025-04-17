import torch
import torch.nn as nn

class SoftOneHotEncodingLayer(nn.Module):
    def __init__(self, num_classes, temperature=0.5):
        super(SoftOneHotEncodingLayer, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, x):
        # Compute softmax over classes for probabilistic one-hot
        x = x.unsqueeze(-1)  # Add dimension for broadcasting
        indices = torch.arange(self.num_classes, device=x.device).float()
        logits = -torch.abs(x - indices - 1) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        mask = torch.clamp(x, 0, 1)
        result = probabilities * mask

        return result