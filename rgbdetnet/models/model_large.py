import torch
import torch.nn as nn

from .backbone_large import RBCDetNetLargeBackbone
from .fpn import FPN
from .head_anchorfree import AnchorFreeHead


# ----------------------------------------------------------
# RBCDetNet-Large (with auxiliary supervision)
# ----------------------------------------------------------
class RBCDetNetLarge(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Backbone output: P3, P4, P5
        self.backbone = RBCDetNetLargeBackbone()

        # FPN to fuse features
        self.fpn = FPN(channels=[128, 256, 512], nano=False)

        # Main detection head
        self.head = AnchorFreeHead(num_classes=num_classes,
                                   in_channels=[128, 256, 512],
                                   feat_channels=128,
                                   nano=False)

        # Auxiliary heads at earlier scales
        self.aux_head_P4 = AnchorFreeHead(num_classes=num_classes,
                                          in_channels=[256],
                                          feat_channels=128,
                                          nano=False)

        self.aux_head_P3 = AnchorFreeHead(num_classes=num_classes,
                                          in_channels=[128],
                                          feat_channels=128,
                                          nano=False)

    def forward(self, x):
        # Backbone
        P3, P4, P5 = self.backbone(x)

        # FPN
        P3_td, P4_pan, P5_pan = self.fpn(P3, P4, P5)

        # Main head predictions
        main_outputs = self.head([P3_td, P4_pan, P5_pan])

        # Auxiliary output at P4
        aux_P4 = self.aux_head_P4([P4_pan])

        # Auxiliary output at P3
        aux_P3 = self.aux_head_P3([P3_td])

        return {
            "main": main_outputs,
            "aux_P4": aux_P4,
            "aux_P3": aux_P3
        }
