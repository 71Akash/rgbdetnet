import cv2
import torch
import time
import numpy as np

from models.model_large import RBCDetNetLarge
from models.model_nano import RBCDetNetNano

from utils.decoder import FCOSDecoder
from utils.nms import weighted_nms


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
class InferConfig:
    weights = "weights/best_nano.pth"     # change to large if needed
    model_type = "nano"                   # "large" or "nano"
    num_classes = 6
    class_names = ["cls0", "cls1", "cls2", "cls3", "cls4", "cls5"]
    img_size = 512                        # inference resolution
    device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# PREPROCESS
# -------------------------------------------------------
def preprocess(frame, img_size=512):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1]                   # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


# -------------------------------------------------------
# DRAW BOXES
# -------------------------------------------------------
def draw_boxes(frame, boxes, scores, labels, cfg):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().cpu().numpy()
        cls_id = int(labels[i].cpu().numpy())
        score = float(scores[i].cpu().numpy())

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{cfg.class_names[cls_id]} {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# -------------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------------
def run_inference():
    cfg = InferConfig()

    # -------------------------
    # Load model
    # -------------------------
    if cfg.model_type == "large":
        model = RBCDetNetLarge(num_classes=cfg.num_classes)
    else:
        model = RBCDetNetNano(num_classes=cfg.num_classes)

    ckpt = torch.load(cfg.weights, map_location=cfg.device)
    model.load_state_dict(ckpt)
    model.to(cfg.device)
    model.eval()

    # Decoder
    decoder = FCOSDecoder(num_classes=cfg.num_classes, strides=[8, 16, 32])

    # -------------------------
    # Camera
    # -------------------------
    cap = cv2.VideoCapture(0)

    print("Starting inference... Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]
        img = preprocess(frame, cfg.img_size).unsqueeze(0).to(cfg.device)

        # -------------------------
        # Forward pass
        # -------------------------
        with torch.no_grad():
            out = model(img)["main"]       # list of 3 scales (bbox, obj, cls)
            boxes, probs = decoder.decode(out)

        # -------------------------
        # Select best class per box
        # -------------------------
        scores, labels = probs.max(dim=-1)   # (B, N)
        scores = scores[0]
        labels = labels[0]
        boxes  = boxes[0]

        # Score threshold
        mask = scores > 0.35
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # -------------------------
        # Weighted NMS (Fast WBF)
        # -------------------------
        if len(boxes) > 0:
            boxes_nms, scores_nms, labels_nms = weighted_nms(
                boxes, scores, labels,
                iou_thresh=0.55,
                top_k=100
            )
        else:
            boxes_nms = torch.zeros((0,4))
            scores_nms = torch.zeros((0,))
            labels_nms = torch.zeros((0,))

        # -------------------------
        # Draw final boxes
        # -------------------------
        frame_out = draw_boxes(frame.copy(), boxes_nms, scores_nms, labels_nms, cfg)

        # FPS counter
        cv2.imshow("RBCDetNet Inference", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
