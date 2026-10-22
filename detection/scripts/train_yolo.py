"""Train YOLOv8 on NEU-DET (6-class hot-rolled steel surface defects).

Example:
    python detection/scripts/train_yolo.py --data ../datasets/NEU-DET/data.yaml \
        --model yolov8m.pt --epochs 120 --imgsz 768 --batch 8
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(data_path):
    """Return a data.yaml whose ``path`` is absolute.

    Ultralytics resolves a relative ``path`` against its global ``datasets_dir``
    setting rather than against the yaml file, so we normalise it here.
    """
    yaml_path = Path(data_path).expanduser().resolve()
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(spec.get("path") or yaml_path.parent)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    spec["path"] = str(root)
    for split in ("train", "val", "test"):
        if split in spec and not Path(spec[split]).is_absolute():
            spec[split] = str((root / spec[split]).resolve())
    tmp = Path(tempfile.gettempdir()) / f"{yaml_path.stem}_abs.yaml"
    tmp.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return str(tmp)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET")
    p.add_argument("--data", default="../datasets/NEU-DET/data.yaml")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="detection/runs")
    p.add_argument("--name", default="neudet")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    a = parse_args()
    data = resolve_data_yaml(a.data)
    model = YOLO(a.model)
    model.train(
        data=data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device=a.device,
        project=a.project,
        name=a.name,
        exist_ok=True,
        resume=a.resume,
        patience=a.patience,
        seed=a.seed,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        # small-sample friendly augmentation for elongated surface defects
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        plots=True,
        val=True,
    )
    metrics = model.val(data=data, split="test")
    out = Path(a.project) / a.name / "test_metrics.txt"
    out.write_text(
        f"mAP50={metrics.box.map50:.4f}\nmAP50-95={metrics.box.map:.4f}\n"
        f"precision={metrics.box.mp:.4f}\nrecall={metrics.box.mr:.4f}\n",
        encoding="utf-8",
    )
    print(out.read_text())


if __name__ == "__main__":
    main()
