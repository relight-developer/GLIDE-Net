"""Recompute precision-recall curves from the published prediction masks.

The paper's PR figure was rendered from the same masks; this script regenerates it
so the report contains no copied image data. Runs on CPU in a couple of minutes.

Example:
    python tools/make_pr_curve.py --pred_root predictions --gt predictions/GT \
        --methods Ours MENet-KD TriTransNet SPNet --out docs/assets/pr-curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import py_sod_metrics  # noqa: E402

BG = "#0B0F14"
PANEL = "#131A22"
INK = "#E8EEF4"
MUTED = "#93A2B1"
COLOURS = ["#F5C542", "#4C9AFF", "#30A46C", "#E5484D", "#A855F7", "#22D3EE",
           "#F97316", "#64748B", "#EC4899", "#84CC16"]


def pr_curve(gt_dir: Path, pred_dir: Path, names):
    """Return (recall, precision, max_f, auc_pr) averaged over the dataset."""
    fmeasure = py_sod_metrics.Fmeasure()
    for name in names:
        gt = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_dir / name), cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        # keep gt in 0/255: py_sod_metrics binarises at 128 internally
        fmeasure.step(pred=pred, gt=gt)
    results = fmeasure.get_results()
    # py_sod_metrics returns parallel arrays: fm/curve is the F-measure, pr/{p,r}
    # are the precision/recall sweeps over the same 256 thresholds.
    pr = results["pr"]
    precision = np.asarray(pr["p"], dtype=np.float64)
    recall = np.asarray(pr["r"], dtype=np.float64)
    # Threshold 0 can produce an all-background prediction; those points carry no
    # ranking information, so drop them before integrating.
    keep = recall > 0.0
    recall, precision = recall[keep], precision[keep]
    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]
    # Monotone precision envelope: precision must not increase with recall.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * envelope))
    return recall, precision, float(np.asarray(results["fm"]["curve"]).max()), ap


def main():
    ap = argparse.ArgumentParser(description="Recompute PR curves from prediction masks")
    ap.add_argument("--pred_root", default="predictions")
    ap.add_argument("--gt", default="predictions/GT")
    ap.add_argument("--methods", nargs="+",
                    default=["Ours", "MENet-KD", "TriTransNet", "SPNet", "CIR-Net"])
    ap.add_argument("--out", default="docs/assets/pr-curves.png")
    ap.add_argument("--csv", default="docs/assets/pr-metrics.csv")
    args = ap.parse_args()

    root = Path(args.pred_root)
    gt_dir = Path(args.gt)
    names = sorted(p.name for p in gt_dir.glob("*.png"))
    if not names:
        raise SystemExit(f"no ground-truth masks in {gt_dir}")

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.edgecolor": "#2A3644", "grid.color": "#22303D",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    table = []
    for colour, method in zip(COLOURS, args.methods):
        pred_dir = root / method
        if not pred_dir.is_dir():
            print(f"skip {method}: {pred_dir} missing")
            continue
        recall, precision, max_f, auc = pr_curve(gt_dir, pred_dir, names)
        weight = 2.6 if method == "Ours" else 1.6
        ax.plot(recall, precision, color=colour, lw=weight,
                label=f"{method}  maxF={max_f:.4f}  AP={auc:.4f}")
        print(f"{method:<14} maxF={max_f:.4f}  AP={auc:.4f}")
        table.append({"method": method, "maxF": round(max_f, 4), "AP": round(auc, 4)})

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title("Precision-Recall（NEU RSDDS-AUG 测试集 362 张，由掩膜重算）")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")
    if table:
        csv_path = Path(args.csv)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("method,maxF,AP\n")
            for row in table:
                f.write(f"{row['method']},{row['maxF']},{row['AP']}\n")
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
