"""Latency / fidelity benchmark across inference backends.

Produces the numbers quoted in ``docs/deployment.md`` and
``docs/experiment-log.md``: per-backend latency, throughput and the agreement
with the PyTorch fp32 reference, so a speed-up is never reported without its
cost.

Example:
    python deploy/benchmark.py --checkpoint ckpt/glide_net_6440647.pth \
        --onnx_fp32 deploy/onnx/glide_net_352_fp32.onnx \
        --onnx_fp16 deploy/onnx/glide_net_352_fp16.onnx \
        --images ../datasets/NEU-RSDDS-AUG/Image_test \
        --depths ../datasets/NEU-RSDDS-AUG/Depth_test \
        --out docs/assets/latency.csv
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.preprocess import load_gray, load_rgb, to_network_input  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark GLIDE-Net backends")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sam2_checkpoint", default=None)
    p.add_argument("--onnx_fp32", default=None)
    p.add_argument("--onnx_fp16", default=None)
    p.add_argument("--images", default=None, help="folder of RGB test images")
    p.add_argument("--depths", default=None, help="folder of depth maps")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--warmup", type=int, default=3,
                   help="untimed runs before measuring (CUDA context + cuDNN autotune)")
    p.add_argument("--size", type=int, default=352)
    p.add_argument("--out", default="docs/assets/latency.csv")
    p.add_argument("--json_out", default=None)
    return p.parse_args()


def _load_samples(args):
    if not args.images:
        x = np.random.randn(1, 4, args.size, args.size).astype(np.float32)
        return [("random", x, (args.size, args.size))]
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in exts)
    files = files[: args.limit]
    samples = []
    for path in files:
        depth_path = None
        if args.depths:
            for ext in (".tiff", ".tif", ".png", ".bmp"):
                cand = Path(args.depths) / f"{path.stem}{ext}"
                if cand.exists():
                    depth_path = cand
                    break
        rgb = load_rgb(path)
        depth = load_gray(depth_path) if depth_path else None
        x, size = to_network_input(rgb, depth, args.size)
        samples.append((path.stem, x, size))
    return samples


def bench_torch(args, samples):
    """End-to-end wall time per image for the PyTorch backend.

    The first call pays for the CUDA context, cuDNN kernel autotuning and the
    fp16 weight cast; without a warm-up that one-off can dominate a short run
    (it is what made an earlier 12-image table look ~4x slower than reality).
    """
    from app.engines.segmentation import SegmentationEngine

    rows = []
    for fp16 in (False, True):
        try:
            engine = SegmentationEngine(checkpoint=args.checkpoint,
                                        sam2_checkpoint=args.sam2_checkpoint,
                                        device="cuda", fp16=fp16, size=args.size)
        except Exception as exc:  # pragma: no cover - backend availability
            print(f"skip torch fp16={fp16}: {exc}")
            continue
        for i in range(max(0, args.warmup)):
            engine.backend.run(samples[i % len(samples)][1])
        sync = getattr(engine.backend, "synchronise", None)
        if sync:
            sync()
        started = time.perf_counter()
        for _, x, _ in samples:
            engine.backend.run(x)
        if sync:
            sync()
        wall = (time.perf_counter() - started) / len(samples) * 1000
        rows.append({"backend": f"torch-{'fp16' if fp16 else 'fp32'}-cuda",
                     "end_to_end_ms": round(wall, 2),
                     "fps": round(1000.0 / wall, 2), "device": "cuda"})
        del engine
    return rows


def bench_onnx(path, label, samples, warmup=3):
    from app.engines.segmentation import SegmentationEngine

    try:
        engine = SegmentationEngine(onnx_path=path, size=samples[0][1].shape[-1])
    except Exception as exc:  # pragma: no cover
        print(f"skip {label}: {exc}")
        return None
    for i in range(max(0, warmup)):
        engine.backend.run(samples[i % len(samples)][1])
    started = time.perf_counter()
    for _, x, _ in samples:
        engine.backend.run(x)
    wall = (time.perf_counter() - started) / len(samples) * 1000
    provider = engine.backend.name
    return {"backend": f"{label}[{provider}]", "end_to_end_ms": round(wall, 2),
            "fps": round(1000.0 / wall, 2),
            "device": "gpu" if "cuda" in provider else "cpu"}


def fidelity(args, samples):
    """Mask agreement between the fp32 reference and every faster backend."""
    from app.engines.segmentation import SegmentationEngine

    outputs = {}
    ref = SegmentationEngine(checkpoint=args.checkpoint, sam2_checkpoint=args.sam2_checkpoint,
                             device="cuda", fp16=False, size=args.size)
    outputs["torch-fp32"] = [ref.backend.run(x) for _, x, _ in samples]
    del ref

    half = SegmentationEngine(checkpoint=args.checkpoint, sam2_checkpoint=args.sam2_checkpoint,
                              device="cuda", fp16=True, size=args.size)
    outputs["torch-fp16"] = [half.backend.run(x) for _, x, _ in samples]
    del half

    for path, label in ((args.onnx_fp32, "onnx-fp32"), (args.onnx_fp16, "onnx-fp16")):
        if not path:
            continue
        try:
            eng = SegmentationEngine(onnx_path=path, size=args.size)
            outputs[label] = [eng.backend.run(x) for _, x, _ in samples]
            del eng
        except Exception as exc:  # pragma: no cover
            print(f"skip fidelity {label}: {exc}")

    base = np.stack(outputs["torch-fp32"])
    report = {}
    for name, outs in outputs.items():
        arr = np.stack(outs)
        diff = np.abs(arr - base)
        ious = []
        for a, b in zip(arr, base):
            pa, pb = (a > 0.5), (b > 0.5)
            union = np.logical_or(pa, pb).sum()
            ious.append(float(np.logical_and(pa, pb).sum() / union) if union else 1.0)
        report[name] = {"mae_vs_fp32": round(float(diff.mean()), 6),
                        "max_abs_vs_fp32": round(float(diff.max()), 6),
                        "mask_iou_vs_fp32_min": round(float(np.min(ious)), 6),
                        "mask_iou_vs_fp32_mean": round(float(np.mean(ious)), 6)}
    return report


def main():
    args = parse_args()
    samples = _load_samples(args)
    rows = []
    if args.checkpoint:
        rows += bench_torch(args, samples)
    for path, label in ((args.onnx_fp32, "onnx-fp32"), (args.onnx_fp16, "onnx-fp16")):
        if path:
            row = bench_onnx(path, label, samples, args.warmup)
            if row:
                rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("backend,device,end_to_end_ms,fps\n")
        for row in rows:
            f.write(f"{row['backend']},{row['device']},"
                    f"{row['end_to_end_ms']},{row['fps']}\n")
    print(f"wrote {args.out}")
    for row in rows:
        print(f"  {row['backend']:<40} {row['end_to_end_ms']:>8.2f} ms"
              f"  {row['fps']:>7.2f} img/s")

    payload = {"latency": rows, "fidelity": {},
               "hardware": {"gpu": _gpu_name(), "python": platform.python_version(),
                            "torch": _torch_version(), "samples": len(samples),
                            "warmup": args.warmup,
                            "input_size": [args.size, args.size]}}
    if args.checkpoint:
        payload["fidelity"] = fidelity(args, samples[: min(8, len(samples))])
        for name, stats in payload["fidelity"].items():
            print(f"  fidelity {name:<14} MAE={stats['mae_vs_fp32']:.6f} "
                  f"IoU>={stats['mask_iou_vs_fp32_min']:.4f}")

    out_json = args.json_out or str(Path(args.out).with_suffix(".json"))
    Path(out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"wrote {out_json}")


def _gpu_name():
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
