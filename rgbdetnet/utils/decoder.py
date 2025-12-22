import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# Decode FCOS distances into XYXY boxes
# ----------------------------------------------------------
class FCOSDecoder:
    """
    Converts:
      - distances (l, t, r, b)
      - objectness
      - class logits
    into absolute bounding boxes.

    Supports multi-scale decoding for P3, P4, P5.
    """

    def __init__(self, num_classes=6, strides=[8, 16, 32]):
        self.num_classes = num_classes
        self.strides = strides

    def _coords(self, h, w, stride, device):
        """
        Generate grid centers for feature map based on stride.
        """
        y = torch.arange(0, h * stride, stride, device=device)
        x = torch.arange(0, w * stride, stride, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        return xx, yy  # shape: [H, W]

    def decode_single(self, bbox, obj, cls, stride):
        """
        bbox: (B, 4, H, W)
        obj:  (B, 1, H, W)
        cls:  (B, C, H, W)
        """
        B, _, H, W = bbox.shape

        device = bbox.device

        # Anchor points (grid centers)
        cx, cy = self._coords(H, W, stride, device)

        # Expand
        cx = cx.view(1, 1, H, W)
        cy = cy.view(1, 1, H, W)

        l = bbox[:, 0:1]
        t = bbox[:, 1:2]
        r = bbox[:, 2:3]
        b = bbox[:, 3:4]

        # Decode XYXY
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b

        # Sigmoid for obj & cls
        obj = obj.sigmoid()
        cls = cls.sigmoid()

        # Multiply objectness into class probabilities
        prob = obj * cls

        # reshape to (B, H*W, C)
        prob = prob.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

        boxes = torch.cat([
            x1, y1, x2, y2
        ], dim=1).permute(0, 2, 3, 1).reshape(B, -1, 4)

        obj = obj.permute(0, 2, 3, 1).reshape(B, -1)

        return boxes, prob, obj

    def decode(self, outputs):
        """
        outputs = [(bbox_p3, obj_p3, cls_p3),
                   (bbox_p4, obj_p4, cls_p4),
                   (bbox_p5, obj_p5, cls_p5)]
        Returns concatenated boxes + probs from all scales.
        """

        decoded = []
        for i, (bbox, obj, cls) in enumerate(outputs):
            boxes, prob, obj_raw = self.decode_single(
                bbox, obj, cls, self.strides[i]
            )
            decoded.append((boxes, prob))

        # Concatenate across scales
        all_boxes = torch.cat([d[0] for d in decoded], dim=1)
        all_probs = torch.cat([d[1] for d in decoded], dim=1)

        return all_boxes, all_probs
