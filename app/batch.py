"""Batch inspection CLI: run the full pipeline over a folder of images.

This is the script behind the acceptance test "100 张图像批量检测无崩溃": it
walks a directory, inspects every image, writes per-image overlays + a CSV and
never aborts on a single bad file (failures are recorded in the CSV and logged).

Example:
    python -m app.batch --images ../datasets/NEU-DET/images/test \
        --out results/batch_neudet --limit 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.pipeline import build_pipeline  # noqa: E402
from app.viz import render_overlay  # noqa: E402

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

CSV_FIELDS = ["index", "image", "status", "verdict", "defect_count",
              "detection_count", "severe", "medium", "light", "area_ratio",
              "worst_score", "classes", "segment_ms", "detect_ms", "total_ms", "error"]


def parse_args():
    p = argparse.ArgumentParser(description="Batch defect inspection")
    p.add_argument("--images", required=True, help="folder with RGB images")
    p.add_argument("--depths", default=None, help="folder with matching depth maps")
    p.add_argument("--out", default="results/batch", help="output folder")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint", default=None, help="GLIDE-Net checkpoint (.pth)")
    p.add_argument("--onnx", default=None, help="GLIDE-Net ONNX graph (alternative)")
    p.add_argument("--detector", default=None, help="YOLOv8 weights (.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no_detector", action="store_true")
    p.add_argument("--no_overlay", action="store_true", help="skip writing PNG masks")
    p.add_argument("--workers", type=int, default=1,
                   help="kept for parity with the UI; inference itself is sequential")
    return p.parse_args()


def _find_depth(depths: Path | None, stem: str):
    if depths is None:
        return None
    for ext in (".tiff", ".tif", ".png", ".bmp"):
        cand = depths / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main():
    args = parse_args()
    if not args.checkpoint and not args.onnx:
        raise SystemExit("pass --checkpoint or --onnx")

    images_dir = Path(args.images)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images found in {images_dir}")

    out_dir = Path(args.out)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    depths = Path(args.depths) if args.depths else None

    pipeline = build_pipeline(segmenter_ckpt=args.checkpoint, onnx_path=args.onnx,
                              detector_weights=None if args.no_detector else args.detector,
                              device=args.device)

    rows, started = [], time.perf_counter()
    for i, path in enumerate(files, 1):
        row = {k: "" for k in CSV_FIELDS}
        row.update({"index": i, "image": path.name, "status": "ok"})
        try:
            report = pipeline.run(path, _find_depth(depths, path.stem),
                                  use_detector=not args.no_detector)
            s = report.summary
            row.update({
                "verdict": s["verdict"], "defect_count": s["defect_count"],
                "detection_count": s["detection_count"],
                "severe": s["severity_counts"]["severe"],
                "medium": s["severity_counts"]["medium"],
                "light": s["severity_counts"]["light"],
                "area_ratio": s["defect_area_ratio"], "worst_score": s["worst_score"],
                "classes": ";".join(f"{k}:{v}" for k, v in s["classes"].items()),
                "segment_ms": round(report.timings["segment_ms"], 2),
                "detect_ms": round(report.timings["detect_ms"], 2),
                "total_ms": round(report.timings["total_ms"], 2),
            })
            if not args.no_overlay:
                overlay = render_overlay(path, report)
                overlay.save(out_dir / "overlays" / f"{path.stem}.png")
            (out_dir / "reports").mkdir(exist_ok=True)
            (out_dir / "reports" / f"{path.stem}.json").write_text(
                json.dumps(report.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # keep going: one bad image must not kill the batch
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        rows.append(row)
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {path.name} -> {row['verdict'] or row['status']}",
                  flush=True)

    wall = time.perf_counter() - started
    with open(out_dir / "batch.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    lat = [float(r["total_ms"]) for r in rows if r["status"] == "ok"]
    summary = {
        "images": len(rows), "succeeded": ok, "failed": len(rows) - ok,
        "wall_time_s": round(wall, 2),
        "mean_total_ms": round(float(np.mean(lat)), 2) if lat else None,
        "p95_total_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
        "throughput_img_s": round(len(rows) / wall, 2),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("severe", "medium", "light", "none")},
        "segment_backend": pipeline.segmenter.name,
        "detector": pipeline.detector.backend if pipeline.detector else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
