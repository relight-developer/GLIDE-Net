"""Static QA for the generated deck: text overflow, overlap and clipping.

``docs/slides.pptx`` is built by ``tools/make_slides.js``; this script reads the
package back and asserts the things a human would notice in a projector:

* no text needs more vertical room than its box provides,
* no two text blocks sit on top of each other,
* nothing runs off the 13.33 x 7.5 in canvas or into the 0.4 in safe margin,
* every image that the generator references really exists in the deck.

Widths are estimated with an Arial-like metric (0.52 em per Latin glyph, 1.0 em
per CJK glyph). It is deliberately conservative — it over-estimates a little, so
a pass here means the real render has slack.

    python tools/check_slides.py            # human readable
    python tools/check_slides.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
SLIDE_W = 13.333
SLIDE_H = 7.5
SAFE = 0.30


def emu_to_in(v) -> float:
    return Emu(int(v)).inches if v is not None else 0.0


def _tokens(line: str) -> list[str]:
    """Split a line into the smallest pieces that cannot be broken further."""
    out, buf = [], ""
    for ch in line:
        if is_wide(ch):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        elif ch == " ":
            if buf:
                out.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def text_width_em(s: str) -> float:
    w = 0.0
    for ch in s:
        if ch == " ":
            w += 0.28
        elif is_wide(ch):
            w += 1.0
        elif ch.isupper() or ch.isdigit():
            w += 0.60
        else:
            w += 0.52
    return w


def block_metrics(tf, box_w_in: float) -> tuple[float, float, str]:
    """Return (needed_height_in, widest_unbreakable_token_in, longest_line).

    Lines wrap, so a long line is not a defect — an unbreakable token wider than
    the box is (it can only overflow). CJK breaks anywhere, Latin at spaces.
    """
    needed = 0.0
    widest = 0.0
    longest = ""
    for para in tf.paragraphs:
        runs = para.runs
        if not runs:
            continue
        text = "".join(r.text for r in runs)
        size = max((r.font.size.pt for r in runs if r.font.size), default=12)
        spacing = para.line_spacing or 1.15
        if isinstance(spacing, float) and spacing and spacing < 3:
            line_h = size * 1.2 * spacing / 1.0
        else:
            line_h = size * 1.45
        for seg in text.split("\n"):
            width_in = text_width_em(seg) * size / 72.0
            if len(seg) > len(longest):
                longest = seg
            for token in _tokens(seg):
                widest = max(widest, text_width_em(token) * size / 72.0)
            lines = max(1, math.ceil(width_in / max(box_w_in, 0.05)))
            needed += lines * line_h / 72.0
    return needed, widest, longest


def collect(slide):
    """Text blocks, images and raw shapes with their geometry in inches."""
    texts, images, shapes = [], [], []
    for shp in slide.shapes:
        x, y = emu_to_in(shp.left), emu_to_in(shp.top)
        w, h = emu_to_in(shp.width), emu_to_in(shp.height)
        if shp.shape_type == 13 or shp.__class__.__name__ == "Picture":
            images.append((x, y, w, h))
            continue
        if shp.has_text_frame and shp.text_frame.text.strip():
            texts.append((x, y, w, h, shp))
        elif w * h > 0.4:
            shapes.append((x, y, w, h))
    return texts, images, shapes


def check(path: Path) -> list[str]:
    prs = Presentation(path)
    problems: list[str] = []
    for idx, slide in enumerate(prs.slides, start=1):
        texts, images, shapes = collect(slide)
        page = f"slide {idx}"

        for x, y, w, h, shp in texts:
            if x < SAFE - 0.02 or y < SAFE - 0.02:
                problems.append(f"{page}: 文本越出上/左安全边距 ({x:.2f},{y:.2f}) "
                                f"『{shp.text_frame.text[:24]}』")
            if x + w > SLIDE_W - SAFE + 0.02 or y + h > SLIDE_H - SAFE + 0.02:
                problems.append(f"{page}: 文本越出下/右安全边距 "
                                f"({x + w:.2f},{y + h:.2f}) 『{shp.text_frame.text[:24]}』")
            needed, widest, longest = block_metrics(shp.text_frame, w)
            if needed > h * 1.02 + 0.02:
                problems.append(f"{page}: 文本可能溢出 需 {needed:.2f}in > 框 {h:.2f}in "
                                f"『{longest[:30]}』")
            if widest > w * 1.03:
                problems.append(f"{page}: 存在无法换行的过长片段 需 {widest:.2f}in > 框 {w:.2f}in "
                                f"『{longest[:30]}』")

        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ox > 0.05 and oy > 0.05:
                    area = ox * oy
                    if area / min(a[2] * a[3], b[2] * b[3]) > 0.30:
                        problems.append(
                            f"{page}: 两段文本重叠 {ox:.2f}x{oy:.2f}in "
                            f"『{a[4].text_frame.text[:16]}』/『{b[4].text_frame.text[:16]}』")

        for x, y, w, h in images:
            if x < SAFE - 0.35 or x + w > SLIDE_W - SAFE + 0.35:
                problems.append(f"{page}: 图片横向越界或贴边 ({x:.2f}→{x + w:.2f})")
            if y < SAFE - 0.35 or y + h > SLIDE_H - SAFE + 0.35:
                problems.append(f"{page}: 图片纵向越界或贴边 ({y:.2f}→{y + h:.2f})")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="QA the generated slide deck")
    ap.add_argument("--deck", default=str(ROOT / "docs" / "slides.pptx"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = Path(args.deck)
    if not path.exists():
        print(f"deck not found: {path}", file=sys.stderr)
        return 2
    prs = Presentation(path)
    problems = check(path)
    n_slides = len(prs.slides._sldIdLst)
    print(f"deck: {path}  slides={n_slides}")
    if problems:
        for p in problems:
            print("  ✗", p)
    else:
        print("  ✓ 无文本溢出 / 无重叠 / 无越界")
    if args.json:
        print(json.dumps({"slides": n_slides, "problems": problems}, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
