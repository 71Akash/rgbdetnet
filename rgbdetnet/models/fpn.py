import torch
import torch.nn as nn


# ----------------------------------------------------------
# Standard convolution block for Large model
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
# Depthwise convolution for Nano model
# ----------------------------------------------------------
class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ----------------------------------------------------------
# Feature Pyramid + PANet Fusion
# ----------------------------------------------------------
class FPN(nn.Module):
    """
    PANet-style top-down + bottom-up feature fusion.
    Adapts automatically for Large or Nano model.

    Large model -> standard conv
    Nano model  -> depthwise conv
    """

    def __init__(self, channels=[128, 256, 512], nano=False):
        super().__init__()

        C3, C4, C5 = channels
        self.nano = nano

        # Choose fusion block
        Block = DWConv if nano else Conv

        # -----------------------------
        # TOP-DOWN PATHWAY
        # -----------------------------

        # P5 -> reduce channels
        self.reduce_p5 = Block(C5, C4, k=1, s=1, p=0)

        # Fuse P4 + upsample(P5)
        self.fuse_p4 = Block(C4 + C4, C4, k=3, s=1, p=1)

        # Reduce P4 for the next step
        self.reduce_p4 = Block(C4, C3, k=1, s=1, p=0)

        # Fuse P3 + upsample(P4)
        self.fuse_p3 = Block(C3 + C3, C3, k=3, s=1, p=1)

        # -----------------------------
        # BOTTOM-UP PATHWAY
        # -----------------------------
        self.down_p3 = Block(C3, C3, k=3, s=2, p=1)
        self.fuse_p4_pan = Block(C4 + C3, C4, k=3, s=1, p=1)

        self.down_p4 = Block(C4, C4, k=3, s=2, p=1)
        self.fuse_p5_pan = Block(C5 + C4, C5, k=3, s=1, p=1)

    def forward(self, P3, P4, P5):
        # -----------------------------
        # TOP-DOWN
        # -----------------------------
        P5_td = self.reduce_p5(P5)
        P5_up = nn.functional.interpolate(P5_td, scale_factor=2, mode="nearest")

        P4_td = self.fuse_p4(torch.cat([P4, P5_up], dim=1))
        P4_td_reduced = self.reduce_p4(P4_td)
        P4_up = nn.functional.interpolate(P4_td_reduced, scale_factor=2, mode="nearest")

        P3_td = self.fuse_p3(torch.cat([P3, P4_up], dim=1))

        # -----------------------------
        # BOTTOM-UP (PAN)
        # -----------------------------
        P3_down = self.down_p3(P3_td)
        P4_pan = self.fuse_p4_pan(torch.cat([P4_td, P3_down], dim=1))

        P4_down = self.down_p4(P4_pan)
        P5_pan = self.fuse_p5_pan(torch.cat([P5, P4_down], dim=1))

        return P3_td, P4_pan, P5_pan
