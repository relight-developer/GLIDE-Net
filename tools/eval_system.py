"""System-level evaluation of the deployed inspection pipeline.

The paper metrics (``predictions/metrics.csv``) score *masks*; this script scores
the *system*: it walks a labelled test split through ``InspectionPipeline`` and
reports

* mask agreement with the ground truth (IoU / Dice / precision / recall),
* the severity distribution produced by ``app/engines/severity.py``,
* latency statistics of the full detect + segment + grade path.

Outputs land in ``docs/assets/`` so every number in the report is regenerable
with one command.

Example:
    python tools/eval_system.py \
        --images ../datasets/NEU-RSDDS-AUG/Image_test \
        --depths ../datasets/NEU-RSDDS-AUG/Depth_test \
        --gt ../datasets/NEU-RSDDS-AUG/GT_test \
        --checkpoint ckpt/glide_net_6440647.pth \
        --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt \
        --out docs/assets/system-eval.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.pipeline import build_pipeline  # noqa: E402

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}
DEPTH_EXTS = (".tiff", ".tif", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the full inspection pipeline")
    p.add_argument("--images", required=True)
    p.add_argument("--depths", default=None)
    p.add_argument("--gt", default=None, help="folder of binary masks (PNG/TIFF)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--detector", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="docs/assets/system-eval.json")
    p.add_argument("--csv", default="docs/assets/system-eval.csv")
    return p.parse_args()


def _find(folder: Path, stem: str):
    for ext in DEPTH_EXTS:
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main():
    args = parse_args()
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]
    depths = Path(args.depths) if args.depths else None
    gts = Path(args.gt) if args.gt else None

    pipeline = build_pipeline(segmenter_ckpt=args.checkpoint, onnx_path=args.onnx,
                              detector_weights=args.detector, device=args.device)

    rows, ious, dices, precs, recs = [], [], [], [], []
    for index, path in enumerate(images, start=1):
        depth = _find(depths, path.stem) if depths else None
        try:
            report = pipeline.run(path, depth)
            payload = report.as_dict()
            summary = payload["summary"]
            row = {
                "index": index,
                "image": path.name,
                "verdict": summary["verdict"],
                "defect_count": summary["defect_count"],
                "detection_count": summary["detection_count"],
                "severe": summary["severity_counts"]["severe"],
                "medium": summary["severity_counts"]["medium"],
                "light": summary["severity_counts"]["light"],
                "classes": "|".join(summary["classes"]),
                "area_ratio": round(summary["defect_area_ratio"], 6),
                "worst_score": round(summary["worst_score"], 3),
                "segment_ms": round(payload["timings"]["segment_ms"], 2),
                "detect_ms": round(payload["timings"]["detect_ms"], 2),
                "total_ms": round(payload["timings"]["total_ms"], 2),
            }
            if gts is not None:
                gt_path = _find(gts, path.stem)
                if gt_path is not None:
                    gt = np.array(Image.open(gt_path).convert("L")) > 127
                    pred = report.mask
                    if pred.shape != gt.shape:
                        pred = np.array(Image.fromarray(pred.astype(np.uint8) * 255)
                                        .resize((gt.shape[1], gt.shape[0]),
                                                Image.NEAREST)) > 127
                    inter = np.logical_and(pred, gt).sum()
                    union = np.logical_or(pred, gt).sum()
                    iou = float(inter / union) if union else 1.0
                    dice = float(2 * inter / (pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) else 1.0
                    prec = float(inter / pred.sum()) if pred.sum() else 1.0
                    rec = float(inter / gt.sum()) if gt.sum() else 1.0
                    ious.append(iou)
                    dices.append(dice)
                    precs.append(prec)
                    recs.append(rec)
                    row.update({"iou": round(iou, 4), "dice": round(dice, 4),
                                "precision": round(prec, 4), "recall": round(rec, 4)})
            rows.append(row)
        except Exception as exc:  # never abort a batch on one bad file
            rows.append({"index": index, "image": path.name, "verdict": "error",
                         "error": repr(exc)})
        if index % 50 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    totals = [r["total_ms"] for r in rows if "total_ms" in r]
    segments = [r["segment_ms"] for r in rows if "segment_ms" in r]
    verdicts = {}
    for r in rows:
        verdicts[r.get("verdict", "error")] = verdicts.get(r.get("verdict", "error"), 0) + 1

    summary = {
        "images": len(rows),
        "errors": sum(1 for r in rows if r.get("verdict") == "error"),
        "verdicts": dict(sorted(verdicts.items())),
        "mask": {
            "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "iou_p10": round(float(np.percentile(ious, 10)), 4) if ious else None,
            "dice_mean": round(float(np.mean(dices)), 4) if dices else None,
            "precision_mean": round(float(np.mean(precs)), 4) if precs else None,
            "recall_mean": round(float(np.mean(recs)), 4) if recs else None,
        },
        "latency_ms": {
            "total_mean": round(statistics.mean(totals), 2) if totals else None,
            "total_p50": round(statistics.median(totals), 2) if totals else None,
            "total_p95": round(float(np.percentile(totals, 95)), 2) if totals else None,
            "segment_mean": round(statistics.mean(segments), 2) if segments else None,
            "segment_p95": round(float(np.percentile(segments, 95)), 2) if segments else None,
        },
        "defect_count_mean": round(statistics.mean([r.get("defect_count", 0) for r in rows]), 3),
        "config": {"threshold": args.threshold, "checkpoint": args.checkpoint,
                   "onnx": args.onnx, "detector": args.detector,
                   "images": str(args.images), "gt": str(args.gt)},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
