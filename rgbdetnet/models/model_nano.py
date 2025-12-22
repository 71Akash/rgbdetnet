import torch
import torch.nn as nn

from .backbone_nano import RBCDetNetNanoBackbone
from .fpn import FPN
from .head_anchorfree import AnchorFreeHead


# ----------------------------------------------------------
# RBCDetNet-Nano (for LattePanda deployment)
# ----------------------------------------------------------
class RBCDetNetNano(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Lightweight backbone
        self.backbone = RBCDetNetNanoBackbone()

        # Depthwise PANet FPN
        self.fpn = FPN(channels=[64, 128, 256], nano=True)

        # Tiny anchor-free decoupled head
        self.head = AnchorFreeHead(num_classes=num_classes,
                                   in_channels=[64, 128, 256],
                                   feat_channels=64,
                                   nano=True)

    def forward(self, x):
        # Backbone
        P3, P4, P5 = self.backbone(x)

        # FPN fusion
        P3_td, P4_pan, P5_pan = self.fpn(P3, P4, P5)

        # Nano head predictions
        outputs = self.head([P3_td, P4_pan, P5_pan])

        return {"main": outputs}
