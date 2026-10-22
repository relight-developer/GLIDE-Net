"""Overlay rendering shared by the UI, the batch CLI and the docs figures.

Everything is drawn with Pillow so the module works without a browser, a
matplotlib figure factory or a GPU. Two products come out of the same layout
code:

``render_overlay``      RGB + prediction blend + defect boxes + severity markers
``render_panel``        a 1x4 strip (RGB / depth / probability / overlay) used in
                        the report and the slide deck
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEVERITY_COLOURS = {
    "severe": (220, 38, 38),
    "medium": (234, 138, 16),
    "light": (37, 99, 235),
    "none": (22, 163, 74),
}
SEVERITY_ZH = {"severe": "严重", "medium": "中等", "light": "轻微", "none": "无缺陷"}
CLASS_ZH = {
    "crazing": "龟裂", "inclusion": "夹杂", "patches": "斑块",
    "pitted_surface": "麻点", "rolled-in_scale": "氧化皮压入", "scratches": "划伤",
}


def _font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _heat(prob: np.ndarray) -> Image.Image:
    """Blue -> yellow -> red heat map of the defect probability."""
    p = np.clip(prob, 0.0, 1.0)
    r = np.clip(1.6 * p - 0.25, 0, 1)
    g = np.clip(1.5 - 2.6 * np.abs(p - 0.55), 0, 1)
    b = np.clip(1.15 - 2.0 * p, 0, 1)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return Image.fromarray(rgb.astype(np.uint8))


def render_overlay(image, report, alpha: float = 0.45, draw_boxes: bool = True,
                   draw_labels: bool = True, scale: float = 1.0) -> Image.Image:
    """Blend the predicted mask over the RGB image and annotate the defects."""
    from app.engines.preprocess import load_rgb

    base = load_rgb(image).convert("RGB")
    w, h = base.size
    if scale != 1.0:
        base = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = base.size

    mask = Image.fromarray((report.mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    heat = _heat(report.prob).resize((w, h), Image.BILINEAR)
    tinted = Image.new("RGB", (w, h), SEVERITY_COLOURS["severe"])
    overlay = base.copy()
    overlay = Image.composite(Image.blend(base, heat, alpha), overlay,
                              mask.point(lambda v: 255 if v > 127 else 0))

    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _font(max(11, int(min(w, h) * 0.030)))
    small = _font(max(9, int(min(w, h) * 0.024)))

    if draw_boxes:
        for region in report.regions:
            colour = SEVERITY_COLOURS.get(region.severity, SEVERITY_COLOURS["light"])
            box = [region.x * w / report.image_size[0], region.y * h / report.image_size[1],
                   (region.x + region.w) * w / report.image_size[0],
                   (region.y + region.h) * h / report.image_size[1]]
            draw.rectangle(box, outline=colour + (255,), width=max(2, int(min(w, h) / 220)))
            if draw_labels:
                name = CLASS_ZH.get(region.label, region.label) if region.label else \
                    SEVERITY_ZH[region.severity]
                tag = f"{name} {region.score:.0f}"
                tw = draw.textlength(tag, font=small)
                ty = max(0, box[1] - small.size - 6)
                draw.rounded_rectangle([box[0], ty, box[0] + tw + 8, ty + small.size + 5],
                                       radius=3, fill=colour + (215,))
                draw.text((box[0] + 4, ty + 2), tag, font=small, fill=(255, 255, 255, 255))

    summary = report.summary
    banner = (f"{SEVERITY_ZH[summary['verdict']]}  ·  缺陷 {summary['defect_count']} 处  ·  "
              f"面积 {summary['defect_area_ratio'] * 100:.2f}%")
    bw = draw.textlength(banner, font=font)
    draw.rounded_rectangle([8, 8, 8 + bw + 16, 8 + font.size + 12], radius=6,
                           fill=(15, 23, 42, 205))
    draw.text((16, 13), banner, font=font, fill=(248, 250, 252, 255))
    return overlay


def render_panel(image, depth, report, labels: Optional[Iterable[str]] = None,
                 target_width: int = 1400) -> Image.Image:
    """RGB | depth | probability heat map | overlay, side by side with captions."""
    from app.engines.preprocess import load_gray, load_rgb

    rgb = load_rgb(image).convert("RGB")
    w, h = rgb.size
    cell = int(target_width / 4)
    cell_h = max(1, int(cell * h / w))

    panels = []
    panels.append(rgb.resize((cell, cell_h), Image.LANCZOS))
    if depth is None:
        panels.append(Image.new("RGB", (cell, cell_h), (30, 41, 59)))
    else:
        panels.append(load_gray(depth).convert("RGB").resize((cell, cell_h), Image.LANCZOS))
    prob = np.clip(report.prob, 0, 1)
    panels.append(_heat(prob).resize((cell, cell_h), Image.BILINEAR))
    panels.append(render_overlay(image, report).resize((cell, cell_h), Image.LANCZOS))

    caption_h = max(22, int(cell * 0.07))
    strip = Image.new("RGB", (cell * 4, cell_h + caption_h), (248, 250, 252))
    draw = ImageDraw.Draw(strip)
    font = _font(max(12, int(cell * 0.045)))
    captions = list(labels or ["输入 RGB", "深度图", "缺陷概率", "检测结果"])
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * cell, caption_h))
        tw = draw.textlength(captions[i], font=font)
        draw.text((i * cell + (cell - tw) / 2, caption_h * 0.22), captions[i],
                  font=font, fill=(30, 41, 59))
        if i:
            draw.line([(i * cell, 0), (i * cell, cell_h + caption_h)],
                      fill=(203, 213, 225), width=1)
    return strip


def overlay_bytes(image, report, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    render_overlay(image, report).save(buf, format=fmt)
    return buf.getvalue()


def save_panel(path, image, depth, report, **kwargs) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_panel(image, depth, report, **kwargs).save(out)
    return out
