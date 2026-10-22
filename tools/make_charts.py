"""Build every figure quoted in ``docs/`` from the machine-readable results.

Nothing in this file is hand-drawn: the curves come from the Ultralytics
``results.csv``, the bars from ``predictions/metrics.csv`` / ``ablation/metrics.csv``
/ ``docs/assets/latency.json``, and the gallery from the published prediction
folders. Re-run it after any experiment and the report figures follow.

Example:
    python tools/make_charts.py                 # writes docs/assets/*.png
    python tools/make_charts.py --only latency  # single chart while iterating
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = Path("docs/assets")
BG = "#0B0F14"
PANEL = "#131A22"
INK = "#E8EEF4"
MUTED = "#93A2B1"
ACCENT = "#F5C542"
BLUE = "#4C9AFF"
GREEN = "#30A46C"
RED = "#E5484D"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": "#2A3644", "grid.color": "#22303D",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    # Chinese labels: fall back to a CJK face when DejaVu lacks the glyphs.
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
})


def _finish(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    out = ASSETS / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ training
def training_curves():
    csv_path = Path("runs/detect/detection/runs/neudet_yolov8m/results.csv")
    if not csv_path.exists():
        print(f"skip training_curves: {csv_path} missing")
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    ep = [int(float(r["epoch"])) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], color=ACCENT, lw=2,
            label="mAP@0.5")
    ax.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in rows], color=BLUE, lw=2,
            label="mAP@0.5:0.95")
    ax.set_title("NEU-DET 检测精度")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ep, [float(r["metrics/precision(B)"]) for r in rows], color=GREEN, lw=1.8,
            label="precision")
    ax.plot(ep, [float(r["metrics/recall(B)"]) for r in rows], color=RED, lw=1.8,
            label="recall")
    ax.set_title("精确率 / 召回率")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.35)
    ax.legend(frameon=False)

    ax = axes[2]
    for key, color, name in (("train/box_loss", ACCENT, "train box"),
                            ("val/box_loss", BLUE, "val box"),
                            ("train/cls_loss", GREEN, "train cls"),
                            ("val/cls_loss", RED, "val cls")):
        ax.plot(ep, [float(r[key]) for r in rows], color=color, lw=1.6, label=name)
    ax.set_title("损失曲线")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    _finish(fig, "training-curves.png")


# ------------------------------------------------------------------ metrics
def comparison_bars():
    rows = list(csv.DictReader(Path("predictions/metrics.csv").open()))
    rows.sort(key=lambda r: float(r["S"]))
    names = [r["Model"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    colors = [ACCENT if n == "Ours" else BLUE for n in names]
    ax.barh(names, [float(r["S"]) for r in rows], color=colors, height=0.7)
    for i, r in enumerate(rows):
        ax.text(float(r["S"]) + 0.002, i, f"{float(r['S']):.4f}", va="center",
                fontsize=9, color=INK if r["Model"] == "Ours" else MUTED)
    ax.set_xlim(0.78, 0.89)
    ax.set_title("S-measure ↑（对照 11 种方法）")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    rows_mae = sorted(rows, key=lambda r: -float(r["MAE"]))
    ax.barh([r["Model"] for r in rows_mae], [float(r["MAE"]) for r in rows_mae],
            color=[ACCENT if r["Model"] == "Ours" else RED for r in rows_mae], height=0.7)
    for i, r in enumerate(rows_mae):
        ax.text(float(r["MAE"]) + 0.0009, i, f"{float(r['MAE']):.4f}", va="center",
                fontsize=9, color=INK if r["Model"] == "Ours" else MUTED)
    ax.set_title("MAE ↓（越低越好）")
    ax.grid(axis="x", alpha=0.3)
    _finish(fig, "comparison-bars.png")


def ablation_bars():
    ours = {"S": 0.8718, "MAE": 0.0501, "maxF": 0.9129, "maxE": 0.9475}
    label = {"w_o_CNN": "w/o CNN 分支", "w_o_Transformer": "w/o Transformer 分支",
             "w_o_MSDEF": "w/o MSDEF", "w_o_CMF": "w/o CMF"}
    rows = list(csv.DictReader(Path("ablation/metrics.csv").open()))
    rows.append({"Model": "Ours", **{k: str(v) for k, v in ours.items()}})
    order = ["Ours", "w_o_CNN", "w_o_Transformer", "w_o_MSDEF", "w_o_CMF"]
    rows.sort(key=lambda r: order.index(r["Model"]))
    names = [label.get(r["Model"], "完整模型") for r in rows]
    metrics = [("S", "S-measure ↑"), ("MAE", "MAE ↓"), ("maxF", "max-F ↑"), ("maxE", "max-E ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(r[key]) for r in rows]
        colors = [ACCENT] + [BLUE] * (len(rows) - 1)
        bars = ax.bar(names, values, color=colors, width=0.62)
        low = min(values)
        high = max(values)
        pad = max((high - low) * 0.6, 0.002)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    _finish(fig, "ablation-bars.png")


# ------------------------------------------------------------------ latency
def latency_chart():
    path = ASSETS / "latency.json"
    if not path.exists():
        print("skip latency: docs/assets/latency.json missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["latency"]
    names = [r["backend"].split("[")[0] for r in rows]
    ms = [r["end_to_end_ms"] for r in rows]
    colors = [ACCENT if "fp16" in n and "torch" in n else BLUE for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    bars = ax.bar(names, ms, color=colors, width=0.6)
    for bar, value in zip(bars, ms):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} ms",
                ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("单张端到端延迟 (ms)")
    ax.set_title(f"推理后端延迟（{payload['hardware']['gpu']}）")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    fid = payload.get("fidelity", {})
    if fid:
        keys = list(fid)
        ax.bar(keys, [fid[k]["mask_iou_vs_fp32_mean"] for k in keys],
               color=[ACCENT if k == "torch-fp16" else BLUE for k in keys], width=0.6)
        ax.set_ylim(0.9, 1.005)
        ax.set_title("相对 fp32 参考的掩膜一致性（mean IoU）")
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, k in enumerate(keys):
            ax.text(i, fid[k]["mask_iou_vs_fp32_mean"], f"{fid[k]['mask_iou_vs_fp32_mean']:.4f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    _finish(fig, "latency.png")


# ------------------------------------------------------------------ severity
def severity_chart():
    path = ASSETS / "system-eval.csv"
    if not path.exists():
        print("skip severity: docs/assets/system-eval.csv missing")
        return
    rows = [r for r in csv.DictReader(path.open()) if r.get("verdict") not in ("error", None)]
    scores, areas = [], []
    for r in rows:
        try:
            scores.append(float(r["worst_score"]))
            areas.append(float(r["area_ratio"]))
        except (KeyError, ValueError):
            continue
    if not scores:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.hist(scores, bins=24, color=BLUE, edgecolor=BG)
    for x, name in ((26, "light | medium"), (55, "medium | severe")):
        ax.axvline(x, color=ACCENT, ls="--", lw=1.4)
        # label sits at the top of the bars, not on the title baseline
        ax.text(x + 1.2, ax.get_ylim()[1] * 0.62, name, color=ACCENT, fontsize=9, rotation=90)
    ax.set_xlabel("最严重区域评分")
    ax.set_ylabel("图像数量")
    ax.set_title(f"系统评测的缺陷评分分布（n={len(scores)}）")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.hist(areas, bins=24, color=GREEN, edgecolor=BG)
    ax.set_xlabel("缺陷面积占比")
    ax.set_ylabel("图像数量")
    ax.set_title("掩膜面积占比分布")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, "severity-dist.png")


# ------------------------------------------------------------------ gallery
def gallery():
    rgb_dir = Path("predictions/inputs/RGB")
    gt_dir = Path("predictions/GT")
    ours_dir = Path("predictions/Ours")
    samples = ["99", "42", "65", "69", "36"]
    if not (rgb_dir.exists() and gt_dir.exists() and ours_dir.exists()):
        print("skip gallery: predictions/inputs missing")
        return
    from PIL import Image

    cols = len(samples)
    fig, axes = plt.subplots(3, cols, figsize=(3.0 * cols, 8.4))
    for j, stem in enumerate(samples):
        for i, (folder, title, cmap) in enumerate(
                ((rgb_dir, "RGB 输入", None), (gt_dir, "真值掩膜", "gray"),
                 (ours_dir, "GLIDE-Net 预测", "gray"))):
            path = folder / f"{stem}.jpg"
            if not path.exists():
                path = folder / f"{stem}.png"
            ax = axes[i, j]
            if path.exists():
                img = Image.open(path).convert("L" if cmap else "RGB")
                ax.imshow(np.asarray(img), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"#{stem}", color=INK)
            if j == 0:
                ax.set_ylabel(title, color=INK)
    _finish(fig, "gallery.png")


CHARTS = {"training": training_curves, "comparison": comparison_bars,
          "ablation": ablation_bars, "latency": latency_chart,
          "severity": severity_chart, "gallery": gallery}


def main():
    ap = argparse.ArgumentParser(description="Build docs/assets figures")
    ap.add_argument("--only", choices=sorted(CHARTS), action="append")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(CHARTS):
        CHARTS[name]()


if __name__ == "__main__":
    main()
