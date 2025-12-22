import torch


def box_iou(boxes1, boxes2):
    """
    IoU between two sets of boxes.
    boxes: (N, 4) as [x1,y1,x2,y2]
    """
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


# ----------------------------------------------------------
# Fast Weighted-NMS (WBF-lite)
# ----------------------------------------------------------
def weighted_nms(boxes, scores, labels, iou_thresh=0.55, top_k=300):
    """
    boxes:  (N, 4)
    scores: (N,)
    labels: (N,)
    """

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    unique_labels = labels.unique()

    for lab in unique_labels:
        # Filter by this class
        idx = torch.where(labels == lab)[0]
        b = boxes[idx]
        s = scores[idx]

        # Sort by score
        order = torch.argsort(s, descending=True)
        b = b[order]
        s = s[order]

        used = torch.zeros(len(b), dtype=torch.bool, device=b.device)

        for i in range(len(b)):
            if used[i]:
                continue

            # reference box
            ref_box = b[i]
            ref_score = s[i]

            # group for weighted merge
            group_boxes = [ref_box]
            group_scores = [ref_score]

            used[i] = True

            if i == len(b) - 1:
                keep_boxes.append(ref_box)
                keep_scores.append(ref_score)
                keep_labels.append(lab)
                continue

            # Compare with following boxes
            ious = box_iou(ref_box.unsqueeze(0), b[i+1:])[0]

            for j, val in enumerate(ious):
                if val > iou_thresh and not used[i+1+j]:
                    used[i+1+j] = True
                    group_boxes.append(b[i+1+j])
                    group_scores.append(s[i+1+j])

            # Weighted merge
            group_boxes = torch.stack(group_boxes)
            group_scores = torch.stack(group_scores)

            weights = group_scores / (group_scores.sum() + 1e-6)
            merged = (group_boxes * weights[:, None]).sum(dim=0)

            keep_boxes.append(merged)
            keep_scores.append(ref_score)
            keep_labels.append(lab)

    # Stack results
    keep_boxes = torch.stack(keep_boxes) if keep_boxes else torch.empty((0, 4))
    keep_scores = torch.stack(keep_scores) if keep_scores else torch.empty((0,))
    keep_labels = torch.stack(keep_labels) if keep_labels else torch.empty((0,))

    # Final sorting and top_k filtering
    if len(keep_scores) > top_k:
        order = torch.argsort(keep_scores, descending=True)[:top_k]
        keep_boxes = keep_boxes[order]
        keep_scores = keep_scores[order]
        keep_labels = keep_labels[order]

    return keep_boxes, keep_scores, keep_labels
