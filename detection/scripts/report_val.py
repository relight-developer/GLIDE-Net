"""Validate a trained YOLOv8 checkpoint and dump machine-readable metrics.

``docs/evaluation.md`` and the UI's metrics tab read the JSON written here, so the
detection numbers in the report are never typed by hand.

Example:
    python detection/scripts/report_val.py \
        --weights detection/runs/neudet_yolov8m/weights/best.pt \
        --data ../datasets/NEU-DET/data.yaml \
        --split test --out docs/assets/detection-metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_yolo import resolve_data_yaml  # noqa: E402  (shared abs-path fix)


def parse_args():
    p = argparse.ArgumentParser(description="Report YOLOv8 validation metrics")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="../datasets/NEU-DET/data.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="docs/assets/detection-metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=resolve_data_yaml(args.data), split=args.split,
                        imgsz=args.imgsz, batch=args.batch, device=args.device,
                        project="detection/runs", name="val", exist_ok=True,
                        verbose=False)

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    ap_class = getattr(metrics.box, "ap_class_index", [])
    per_class = {}
    for i, class_index in enumerate(ap_class):
        try:
            per_class[names[int(class_index)]] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50-95": round(float(metrics.box.ap[i]), 4),
            }
        except (IndexError, TypeError):  # pragma: no cover - ultralytics internals
            continue

    payload = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "overall": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
        },
        "per_class": per_class,
        "class_names": list(names.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
