"""Render the figures that showcase the *app's own* visualisation output.

Two products, both straight from ``InspectionPipeline`` + ``app/viz.py`` so the
report shows what the workstation really draws:

``docs/assets/example-overlay.png``  severity-coloured mask overlay + banner
``docs/assets/example-panel.png``    1x4 strip: RGB / depth / probability / result

The example is chosen deterministically (first test image whose verdict is
``severe`` and that has a labelled region) so the committed PNGs stay stable.

Example:
    python tools/make_examples.py --checkpoint ckpt/glide_net_6440647.pth \
        --detector runs/detect/detection/runs/neudet_yolov8m/weights/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.pipeline import build_pipeline  # noqa: E402
from app.engines.preprocess import load_gray, load_rgb  # noqa: E402
from app.viz import render_overlay, render_panel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Render showcase overlays from the pipeline")
    p.add_argument("--images", default="../datasets/NEU-RSDDS-AUG/Image_test")
    p.add_argument("--depths", default="../datasets/NEU-RSDDS-AUG/Depth_test")
    p.add_argument("--checkpoint", default="ckpt/glide_net_6440647.pth")
    p.add_argument("--onnx", default=None)
    p.add_argument("--detector", default=None)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--out_dir", default="docs/assets")
    p.add_argument("--scale", type=float, default=3.0,
                   help="upscale factor for the overlay (test images are small)")
    return p.parse_args()


def main():
    args = parse_args()
    images = sorted(p for p in Path(args.images).iterdir()
                    if p.suffix.lower() in {".bmp", ".jpg", ".png"})[: args.limit]
    depths = Path(args.depths) if args.depths else None
    pipeline = build_pipeline(segmenter_ckpt=args.checkpoint, onnx_path=args.onnx,
                              detector_weights=args.detector, device="cuda")

    chosen = None
    for path in images:
        depth = None
        if depths:
            for ext in (".tiff", ".tif", ".png"):
                cand = depths / f"{path.stem}{ext}"
                if cand.exists():
                    depth = cand
                    break
        report = pipeline.run(path, depth)
        s = report.summary
        if s["verdict"] == "severe" and s["classes"]:
            chosen = (path, depth, report)
            break

    if chosen is None:
        raise SystemExit("no sample with a labelled severe defect found; widen --limit")

    path, depth, report = chosen
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = load_rgb(path)
    depth_img = load_gray(depth) if depth else None

    overlay = render_overlay(image, report, scale=args.scale)
    overlay.save(out / "example-overlay.png")
    # render_panel controls its own width via target_width
    panel = render_panel(image, depth_img, report)
    panel.save(out / "example-panel.png")

    s = report.summary
    print(f"sample: {path.name}  verdict={s['verdict']}  regions={s['defect_count']}  "
          f"area={s['defect_area_ratio']:.4f}  score={s['worst_score']}")
    print(f"regions: {[(r.label, r.severity, r.score) for r in report.regions][:4]}")
    print(f"wrote {out / 'example-overlay.png'} and {out / 'example-panel.png'}")


if __name__ == "__main__":
    main()
