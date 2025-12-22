import torch
import torch.nn as nn


# ----------------------------------------------------------
# Basic conv block
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
# Depthwise conv block (for lightweight Nano head)
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
# A+B Hybrid Decoupled Anchor-Free Head
# FCOS distances + YOLOX decoupled head structure
# ----------------------------------------------------------
class AnchorFreeHead(nn.Module):
    def __init__(self, num_classes=6, in_channels=[128, 256, 512], feat_channels=128, nano=False):
        super().__init__()

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nano = nano

        Block = DWConv if nano else Conv

        # Shared tower for bbox regression
        self.bbox_tower = nn.ModuleList([
            nn.Sequential(
                Block(ch, feat_channels),
                Block(feat_channels, feat_channels)
            ) for ch in in_channels
        ])

        # Shared tower for classification + objectness
        self.cls_tower = nn.ModuleList([
            nn.Sequential(
                Block(ch, feat_channels),
                Block(feat_channels, feat_channels)
            ) for ch in in_channels
        ])

        # Prediction layers
        self.bbox_pred = nn.ModuleList([
            nn.Conv2d(feat_channels, 4, 1) for _ in in_channels  # (l, t, r, b)
        ])

        self.obj_pred = nn.ModuleList([
            nn.Conv2d(feat_channels, 1, 1) for _ in in_channels  # objectness
        ])

        self.cls_pred = nn.ModuleList([
            nn.Conv2d(feat_channels, num_classes, 1) for _ in in_channels  # classes
        ])

        # FCOS: learnable scale for box regression
        self.scales = nn.Parameter(torch.ones(len(in_channels), dtype=torch.float))

    def forward(self, features):
        """
        features: [P3, P4, P5]
        returns: list of predictions for each scale
        """

        outputs = []

        for i, x in enumerate(features):
            # Decoupled heads
            bbox_feature = self.bbox_tower[i](x)
            cls_feature = self.cls_tower[i](x)

            # FCOS distances
            bbox = torch.exp(self.scales[i] * self.bbox_pred[i](bbox_feature))

            # class + objectness
            obj = self.obj_pred[i](cls_feature)
            cls = self.cls_pred[i](cls_feature)

            outputs.append((bbox, obj, cls))

        return outputs
