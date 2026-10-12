"""
Batch evaluation of prediction folders against a ground-truth folder.

Each sub-directory of ``--pred_root`` is treated as one method. Prediction
files are matched to ground-truth files by name; ground truth is read as an
8-bit mask (binarised at 128).

Example:
    python tools/evaluate_predictions.py \
        --pred_root predictions \
        --gt_dir predictions/GT \
        --output predictions/metrics.csv
"""
import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import MetricAccumulator

FIELDS = ["Model", "S", "MAE", "maxE", "maxF", "meanE", "adpF", "wF"]


def evaluate_folder(gt_dir, pred_dir, gt_names):
    accumulator = MetricAccumulator()
    for name in gt_names:
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            raise ValueError(f"Cannot read GT/prediction pair: {name}")
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        accumulator.update(pred, gt)
    return accumulator.results()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a tree of predictions")
    parser.add_argument("--pred_root", type=str, required=True,
                        help="Directory containing one sub-folder per method")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground-truth mask directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write the metrics CSV")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="Sub-folder names to skip")
    return parser.parse_args()


def main():
    args = parse_args()
    gt_names = sorted(n for n in os.listdir(args.gt_dir) if n.lower().endswith(".png"))
    if not gt_names:
        raise ValueError("No ground-truth PNG masks found.")

    rows = []
    for model in sorted(os.listdir(args.pred_root)):
        pred_dir = os.path.join(args.pred_root, model)
        if not os.path.isdir(pred_dir) or model in args.exclude:
            continue
        results = evaluate_folder(args.gt_dir, pred_dir, gt_names)
        row = {"Model": model}
        row.update({k: f"{results[k]:.4f}" for k in FIELDS if k != "Model"})
        rows.append(row)
        print(f"{model:20s} " + " ".join(f"{k}={row[k]}" for k in FIELDS if k != "Model"))

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
