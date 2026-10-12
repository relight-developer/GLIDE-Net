"""
Build qualitative comparison / ablation grids from the published predictions.

Example:
    python tools/make_figures.py --out_dir figures
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

# Sample ids shown in the paper's qualitative figures.
SAMPLES = ["99", "42", "65", "69", "36"]

COMPARISON_COLUMNS = ["Ours", "SPNet", "TriTransNet", "CIR-Net", "RFNet", "FHENet",
                      "OpenRSS", "MENet-KD", "SA2F", "SAINet", "SAM2-Adapter", "RDNet-KD"]
ABLATION_COLUMNS = ["w_o_CNN", "w_o_Transformer", "w_o_MSDEF", "w_o_CMF"]


def _load(path):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "times.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_grid(rows, out_path, col_width=220, padding=8, label_height=28):
    """rows: list of (list of PIL images or None, list of column labels)."""
    num_cols = len(rows[0][0])
    font = _font(int(label_height * 0.7))

    grid = []
    for imgs, labels in rows:
        resized = []
        row_h = 1
        for img in imgs:
            if img is None:
                resized.append(None)
                continue
            h = max(1, round(img.height * col_width / img.width))
            img = img.resize((col_width, h), Image.LANCZOS)
            resized.append(img)
            row_h = max(row_h, h)
        grid.append((resized, row_h, labels))

    total_w = padding + num_cols * (col_width + padding)
    total_h = padding + sum(h + label_height + padding for _, h, _ in grid)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = padding
    for resized, row_h, labels in grid:
        for idx, img in enumerate(resized):
            x = padding + idx * (col_width + padding)
            if img is None:
                draw.rectangle([x, y, x + col_width, y + row_h], fill=(210, 210, 210))
            else:
                canvas.paste(img, (x, y + (row_h - img.height) // 2))
        for idx, label in enumerate(labels):
            x = padding + idx * (col_width + padding)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (col_width - tw) / 2, y + row_h + 6), label,
                      fill=(0, 0, 0), font=font)
        y += row_h + label_height + padding

    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")


def collect(pred_root, gt_dir, rgb_dir, depth_dir, columns):
    rows = []
    labels = ["Image", "Depth", "GT"] + columns
    for sample in SAMPLES:
        imgs = [
            _load(os.path.join(rgb_dir, sample + ".jpg")),
            _load(os.path.join(depth_dir, sample + ".png")),
            _load(os.path.join(gt_dir, sample + ".png")),
        ]
        for col in columns:
            imgs.append(_load(os.path.join(pred_root, col, sample + ".png")))
        rows.append((imgs, labels))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build qualitative figures")
    parser.add_argument("--predictions", type=str, default="predictions")
    parser.add_argument("--ablation", type=str, default="ablation")
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt_dir = os.path.join(args.predictions, "GT")
    rgb_dir = os.path.join(args.predictions, "inputs", "RGB")
    depth_dir = os.path.join(args.predictions, "inputs", "Depth")

    build_grid(
        collect(args.predictions, gt_dir, rgb_dir, depth_dir, COMPARISON_COLUMNS),
        os.path.join(args.out_dir, "qualitative-comparison.png"),
    )
    build_grid(
        collect(args.ablation, gt_dir, rgb_dir, depth_dir, ABLATION_COLUMNS),
        os.path.join(args.out_dir, "ablation.png"),
    )


if __name__ == "__main__":
    main()
