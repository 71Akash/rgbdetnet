import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# IoU / GIoU / CIoU for bounding box regression
# ----------------------------------------------------------
def box_iou_loss(pred, target, eps=1e-7, mode="giou"):
    """
    pred: (B, 4, H, W)  -> l, t, r, b distances
    target: same shape
    """

    px1 = -pred[:, 0]
    py1 = -pred[:, 1]
    px2 = pred[:, 2]
    py2 = pred[:, 3]

    tx1 = -target[:, 0]
    ty1 = -target[:, 1]
    tx2 = target[:, 2]
    ty2 = target[:, 3]

    # Pred box area
    inter_w = torch.min(px2, tx2) - torch.max(px1, tx1)
    inter_h = torch.min(py2, ty2) - torch.max(py1, ty1)
    inter = torch.clamp(inter_w, min=0) * torch.clamp(inter_h, min=0)

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    union = area_p + area_t - inter + eps
    iou = inter / union

    if mode == "iou":
        return 1 - iou

    # GIoU
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c_area = cw * ch + eps

    giou = iou - (c_area - union) / c_area
    return 1 - giou


# ----------------------------------------------------------
# Quality Focal Loss (used in YOLOX)
# ----------------------------------------------------------
class QualityFocalLoss(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, score):
        """
        pred: (B, C, H, W) logits
        target: (B, C, H, W) one-hot class targets
        score: IoU quality score (FCOS-style) → guides classification
        """
        pred_prob = pred.sigmoid()

        # positive pixels
        pos_mask = target > 0

        # focal modulation
        loss_pos = (score - pred_prob).abs().pow(self.beta) * F.binary_cross_entropy_with_logits(
            pred[pos_mask], target[pos_mask], reduction="sum"
        ) if pos_mask.sum() > 0 else pred_prob.sum() * 0.0

        # negative pixels
        neg_mask = ~pos_mask
        loss_neg = (pred_prob.pow(self.beta) * F.binary_cross_entropy_with_logits(
            pred[neg_mask], target[neg_mask], reduction="sum"
        )) if neg_mask.sum() > 0 else pred_prob.sum() * 0.0

        return loss_pos + loss_neg


# ----------------------------------------------------------
# Total Loss for RBCDetNet
# ----------------------------------------------------------
class DetectionLoss(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

        self.cls_loss = QualityFocalLoss()
        self.obj_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, preds, targets):
        """
        preds: 
            {
              "main": [(bbox_p3, obj_p3, cls_p3),
                       (bbox_p4, obj_p4, cls_p4),
                       (bbox_p5, obj_p5, cls_p5)],
              "aux_P4": [...],
              "aux_P3": [...]
            }

        targets: dictionary prepared by dataset (per feature map)
        """

        total_loss = 0
        total_box = 0
        total_obj = 0
        total_cls = 0

        # Handle main + auxiliary heads
        all_heads = []
        all_heads += preds["main"]
        if "aux_P4" in preds:
            all_heads += preds["aux_P4"]
        if "aux_P3" in preds:
            all_heads += preds["aux_P3"]

        for i, (bbox_pred, obj_pred, cls_pred) in enumerate(all_heads):
            tgt = targets[i]  # bounding box targets for this scale

            bbox_t = tgt["bbox"]
            obj_t = tgt["obj"]
            cls_t = tgt["cls"]
            quality = tgt["quality"]  # IoU quality score

            # Box loss
            box_loss = box_iou_loss(bbox_pred, bbox_t, mode="giou").sum()

            # Objectness loss
            obj_loss = self.obj_loss(obj_pred, obj_t)

            # Classification loss
            cls_loss = self.cls_loss(cls_pred, cls_t, score=quality)

            total_box += box_loss
            total_obj += obj_loss
            total_cls += cls_loss

        total_loss = total_box + total_obj + total_cls

        return {
            "loss": total_loss,
            "box": total_box,
            "obj": total_obj,
            "cls": total_cls
        }
