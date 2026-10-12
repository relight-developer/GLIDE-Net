"""
Training entry point for GLIDE-Net.

Example:
    python train.py --data_path /path/to/dataset \
        --sam2_checkpoint /path/to/sam2_hiera_large.pt \
        --save_dir ./checkpoints
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import RailDefectDataset, TestDataset
from eval import MetricAccumulator, format_results
from glide_net.configs.glide_net_config import GLIDE_NetConfig
from glide_net.models.glide_net import GLIDE_Net
from glide_net.checkpoint import load_model_state, prediction_mask, read_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLIDE-Net")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset root")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="Path to the SAM2 Hiera-L checkpoint")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--val_split", type=str, default="test",
                        help="Split used for validation")
    parser.add_argument("--epochs", type=int, default=GLIDE_NetConfig.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=GLIDE_NetConfig.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=GLIDE_NetConfig.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=GLIDE_NetConfig.WEIGHT_DECAY)
    parser.add_argument("--image_size", type=int, default=GLIDE_NetConfig.INPUT_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loss_mode", choices=("legacy", "weighted"), default="legacy",
                        help="legacy reproduces the BCE reduction used for the paper checkpoint")
    return parser.parse_args()


def structure_loss(pred, mask, mode="legacy"):
    """Historical BCE + weighted IoU, or corrected pixel-weighted BCE."""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction="mean" if mode == "legacy" else "none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def trainable_state_dict(model):
    """State dict limited to trainable parameters and all buffers.

    The SAM2 Hiera trunk is frozen, so its weights are re-loaded from
    ``--sam2_checkpoint`` instead of being duplicated in every checkpoint.
    """
    keep = {name for name, p in model.named_parameters() if p.requires_grad}
    keep |= {name for name, _ in model.named_buffers()}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def train_one_epoch(model, loader, optimizer, device, loss_mode="legacy"):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        targets = batch["label"].to(device)
        inputs = torch.cat([images, depths], dim=1)

        outputs = model(inputs)
        loss = sum(structure_loss(out, targets, loss_mode) for out in outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{len(loader)}  loss {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    accumulator = MetricAccumulator()
    for images, depths, targets, _ in loader:
        outputs = model(torch.cat([images, depths], dim=1).to(device))[0]
        pred = prediction_mask(outputs, targets.shape[2:])
        gt = targets[0, 0].numpy().astype(np.uint8)
        accumulator.update(pred, gt)
    return accumulator.results()


def main():
    args = parse_args()
    if args.image_size != 352:
        raise ValueError("The released paper configuration requires --image_size 352.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = GLIDE_NetConfig.DEVICE
    os.makedirs(args.save_dir, exist_ok=True)

    model = GLIDE_Net(checkpoint_path=args.sam2_checkpoint, device=device).to(device)

    train_set = RailDefectDataset(args.data_path, split="train",
                                  image_size=args.image_size)
    val_set = TestDataset(args.data_path, split=args.val_split, image_size=args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=GLIDE_NetConfig.MIN_LEARNING_RATE)

    start_epoch, best_score = 0, float("-inf")
    if args.resume:
        checkpoint, state_dict = read_checkpoint(args.resume)
        if not all(k in checkpoint for k in ("optimizer_state_dict", "scheduler_state_dict", "epoch")):
            raise ValueError("Resume requires last.pth with optimizer, scheduler and epoch state.")
        load_model_state(model, state_dict, allow_frozen_missing=True)
        if checkpoint.get("args", {}).get("loss_mode", args.loss_mode) != args.loss_mode:
            raise ValueError("Resume loss_mode must match the saved training configuration.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint.get("best_score", best_score)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        del checkpoint, state_dict

    print(f"Training on {device} | {len(train_set)} train / {len(val_set)} val samples")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.loss_mode)
        metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"  train loss {train_loss:.4f}")
        print("  " + format_results(metrics).replace("\n", "\n  "))

        score = float(1 - metrics["MAE"] + metrics["meanMaxF"] + metrics["S"])
        improved = score > best_score
        best_score = max(best_score, score)
        state = {
            "epoch": epoch + 1,
            "model_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "args": vars(args),
        }
        torch.save(state, os.path.join(args.save_dir, "last.pth"))

        if improved:
            torch.save({"epoch": epoch + 1,
                        "model_state_dict": state["model_state_dict"],
                        "best_score": best_score, "args": vars(args)},
                       os.path.join(args.save_dir, "best.pth"))
            print(f"  new best score {best_score:.4f} -> best.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
