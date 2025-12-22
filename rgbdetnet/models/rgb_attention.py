import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBAttention(nn.Module):
    """
    Novel RGB channel attention: learns how much weight to give R, G, B
    while keeping computations lightweight.

    Input: (B, 3, H, W)
    Output: (B, 3, H, W)
    """

    def __init__(self, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B,3,1,1)
        self.fc1 = nn.Linear(3, 3 // reduction)
        self.fc2 = nn.Linear(3 // reduction, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        out = x * y
        return out
