import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models.model_large import RBCDetNetLarge
from models.model_nano import RBCDetNetNano
from utils.loss import DetectionLoss
from utils.decoder import FCOSDecoder
from utils.nms import weighted_nms
from dataset import YOLOFCOSDataset

# -----------------------------------------------------------
# TRAINING CONFIG
# -----------------------------------------------------------
class TrainConfig:
    img_size = 640
    batch = 8
    workers = 4
    num_epochs = 100
    lr = 0.0005
    weight_decay = 0.0001
    warmup_epochs = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "weights/"
    model_type = "large"   # "large" or "nano"


# -----------------------------------------------------------
# TRAIN FUNCTION
# -----------------------------------------------------------
def train():
    cfg = TrainConfig()

    # -------------------------------------------------------
    # CREATE MODEL
    # -------------------------------------------------------
    if cfg.model_type == "large":
        model = RBCDetNetLarge(num_classes=6)
    else:
        model = RBCDetNetNano(num_classes=6)

    model = model.to(cfg.device)

    # -------------------------------------------------------
    # OPTIMIZER & LOSS
    # -------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    criterion = DetectionLoss(num_classes=6)

    # -------------------------------------------------------
    # DATASET & LOADER
    # -------------------------------------------------------
    train_dataset = YOLOFCOSDataset(
        img_dir="dataset/images/train",
        label_dir="dataset/labels/train",
        img_size=cfg.img_size,
        mosaic=True
    )

    val_dataset = YOLOFCOSDataset(
        img_dir="dataset/images/val",
        label_dir="dataset/labels/val",
        img_size=cfg.img_size,
        mosaic=False
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch,
                              shuffle=True, num_workers=cfg.workers)

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch,
                            shuffle=False, num_workers=cfg.workers)

    # -------------------------------------------------------
    # AMP MIXED PRECISION
    # -------------------------------------------------------
    scaler = GradScaler()

    # -------------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------------
    best_loss = 9e9
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("Training started...\n")

    for epoch in range(cfg.num_epochs):

        model.train()
        epoch_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(cfg.device)

            # move all target maps to device
            for t_idx in range(len(targets)):
                for key in targets[t_idx]:
                    targets[t_idx][key] = targets[t_idx][key].to(cfg.device)

            optimizer.zero_grad()

            # forward pass with AMP
            with autocast():
                preds = model(imgs)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ---------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------
        val_loss = validate(model, val_loader, criterion, cfg)

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}   Val Loss: {val_loss:.4f}")

        # Save best weights
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg.save_dir, f"best_{cfg.model_type}.pth"))
            print(" >> Best model saved.")

    print("\nTraining completed.")


# -----------------------------------------------------------
# VALIDATION LOOP
# -----------------------------------------------------------
def validate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(cfg.device)

            for t_idx in range(len(targets)):
                for key in targets[t_idx]:
                    targets[t_idx][key] = targets[t_idx][key].to(cfg.device)

            preds = model(imgs)
            loss_dict = criterion(preds, targets)
            total_loss += loss_dict["loss"].item()

    return total_loss / len(loader)


if __name__ == "__main__":
    train()
