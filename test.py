"""
Inference entry point for GLIDE-Net.

Writes one 8-bit PNG prediction per test image to ``--save_path``; the masks
can then be scored with ``eval.py``.

Example:
    python test.py --data_path /path/to/dataset \
        --checkpoint ./checkpoints/best.pth \
        --sam2_checkpoint /path/to/sam2_hiera_large.pt \
        --save_path ./results/pred
"""
import argparse
import os
import time

import cv2
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from glide_net.configs.glide_net_config import GLIDE_NetConfig
from glide_net.models.glide_net import GLIDE_Net
from glide_net.checkpoint import load_model_state, read_checkpoint, prediction_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run GLIDE-Net inference")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset root")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained GLIDE-Net checkpoint")
    parser.add_argument("--sam2_checkpoint", type=str, default=None,
                        help="Required only for checkpoints that omit the frozen backbone")
    parser.add_argument("--save_path", type=str, default="./results/pred")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=GLIDE_NetConfig.INPUT_SIZE)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Infer only the first N samples")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Continue an interrupted run with the same checkpoint and settings")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size != 352 or args.cpu_threads < 1:
        raise ValueError("Use --image_size 352 and a positive --cpu_threads value.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    device = torch.device(args.device) if args.device else GLIDE_NetConfig.DEVICE
    torch.set_num_threads(args.cpu_threads)
    os.makedirs(args.save_path, exist_ok=True)

    model = GLIDE_Net(checkpoint_path=args.sam2_checkpoint, device=device).to(device)

    checkpoint, state_dict = read_checkpoint(args.checkpoint)
    load_model_state(model, state_dict, allow_frozen_missing=args.sam2_checkpoint is not None)
    del checkpoint, state_dict
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    dataset = TestDataset(args.data_path, split=args.split, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)
    print(f"Running inference on {len(dataset)} images")
    started = time.monotonic()

    with torch.inference_mode():
        for i, (image, depth, gt, name) in enumerate(loader):
            if args.limit is not None and i >= args.limit:
                break
            output_path = os.path.join(args.save_path, f"{name[0]}.png")
            if args.skip_existing and os.path.isfile(output_path):
                continue
            inputs = torch.cat([image, depth], dim=1).to(device)
            pred = prediction_mask(model(inputs)[0], gt.shape[2:])

            if not cv2.imwrite(output_path, pred):
                raise OSError(f"Failed to write prediction: {name[0]}")
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  saved {i + 1}/{len(dataset)} | {time.monotonic() - started:.1f}s", flush=True)

    print(f"Predictions saved to {args.save_path}")


if __name__ == "__main__":
    main()
