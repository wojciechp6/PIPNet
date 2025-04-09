import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, tau=1.0, hard=False, eps=None, dim=None):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, eps=self.eps, dim=self.dim)