import torch
import torch.nn as nn


# ----------------------------------------------------------
# Depthwise Separable Convolution
# ----------------------------------------------------------
class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ----------------------------------------------------------
# Tiny Bottleneck (depthwise)
# ----------------------------------------------------------
class NanoBottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        hidden = c // 2
        self.conv1 = DWConv(c, hidden, k=1, s=1, p=0)
        self.conv2 = DWConv(hidden, c, k=3, s=1, p=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y if self.shortcut else y


# ----------------------------------------------------------
# Nano CSP Block (lightweight)
# ----------------------------------------------------------
class NanoCSPBlock(nn.Module):
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        hidden = c_out // 2

        self.part1 = nn.Sequential(
            DWConv(c_in, hidden, k=1, s=1, p=0),
            *[NanoBottleneck(hidden) for _ in range(n)],
            DWConv(hidden, hidden, k=1, s=1, p=0)
        )

        self.part2 = DWConv(c_in, hidden, k=1, s=1, p=0)

        self.fuse = DWConv(hidden * 2, c_out, k=1, s=1, p=0)

    def forward(self, x):
        y1 = self.part1(x)
        y2 = self.part2(x)
        return self.fuse(torch.cat([y1, y2], dim=1))


# ----------------------------------------------------------
# RBCDetNet Nano Backbone (fast for LattePanda)
# ----------------------------------------------------------
class RBCDetNetNanoBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem: 640 → 320
        self.stem = DWConv(3, 16, k=3, s=2)

        # Stage 1: 320 → 160
        self.stage1 = nn.Sequential(
            DWConv(16, 32, k=3, s=2),
            NanoCSPBlock(32, 32, n=1)
        )  # 160×160

        # Stage 2: 160 → 80 (P3)
        self.stage2 = nn.Sequential(
            DWConv(32, 64, k=3, s=2),
            NanoCSPBlock(64, 64, n=3)
        )  # 80×80

        # Stage 3: 80 → 40 (P4)
        self.stage3 = nn.Sequential(
            DWConv(64, 128, k=3, s=2),
            NanoCSPBlock(128, 128, n=3)
        )  # 40×40

        # Stage 4: 40 → 20 (P5)
        self.stage4 = nn.Sequential(
            DWConv(128, 256, k=3, s=2),
            NanoCSPBlock(256, 256, n=2)
        )  # 20×20

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        x = self.stage2(x)
        P3 = x

        x = self.stage3(x)
        P4 = x

        x = self.stage4(x)
        P5 = x

        return P3, P4, P5
