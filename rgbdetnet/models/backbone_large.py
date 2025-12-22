import torch
import torch.nn as nn
from .rgb_attention import RGBAttention


# ----------------------------------------------------------
# Basic Convolution Module
# ----------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ----------------------------------------------------------
# ResNet Bottleneck Block
# ----------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        hidden = c // 2

        self.conv1 = Conv(c, hidden, k=1, p=0)
        self.conv2 = Conv(hidden, c, k=3, p=1)

        self.use_shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y if self.use_shortcut else y


# ----------------------------------------------------------
# CSP Block (split → convs → merge)
# ----------------------------------------------------------
class CSPBlock(nn.Module):
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        hidden = c_out // 2

        # Main branch
        self.conv1 = Conv(c_in, hidden, k=1, p=0)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(n)])
        self.conv2 = Conv(hidden, hidden, k=1, p=0)

        # Shortcut branch
        self.shortcut = Conv(c_in, hidden, k=1, p=0)

        # Merge
        self.merge = Conv(hidden * 2, c_out, k=1, p=0)

    def forward(self, x):
        y1 = self.conv2(self.blocks(self.conv1(x)))
        y2 = self.shortcut(x)
        return self.merge(torch.cat([y1, y2], dim=1))


# ----------------------------------------------------------
# Hybrid Backbone (CSP + ResNet)
# ----------------------------------------------------------
class RBCDetNetLargeBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 640 → 320
        self.rgb_att = RGBAttention()
        self.stem = Conv(3, 32, k=3, s=2)  

        # Stage 1 — CSP
        self.stage1 = nn.Sequential(
            Conv(32, 64, k=3, s=2),
            CSPBlock(64, 64, n=1)
        )  # Output: 160×160

        # Stage 2 — Deeper CSP
        self.stage2 = nn.Sequential(
            Conv(64, 128, k=3, s=2),
            CSPBlock(128, 128, n=3)
        )  # Output: 80×80  (P3)

        # Stage 3 — ResNet-style
        self.stage3 = nn.Sequential(
            Conv(128, 256, k=3, s=2),
            *[Bottleneck(256) for _ in range(6)]
        )  # Output: 40×40  (P4)

        # Stage 4 — Deep residual block
        self.stage4 = nn.Sequential(
            Conv(256, 512, k=3, s=2),
            *[Bottleneck(512) for _ in range(3)]
        )  # Output: 20×20  (P5)

    def forward(self, x):
        x = self.rgb_att(x)
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        P3 = x

        x = self.stage3(x)
        P4 = x

        x = self.stage4(x)
        P5 = x

        return P3, P4, P5
