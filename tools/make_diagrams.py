"""Generate the project's architecture diagrams as draw.io XML.

Reproducible diagrams beat checked-in binaries: every box here is generated
from data (see ``MODEL_DIAGRAM`` / ``SYSTEM_DIAGRAM``) so the figure and the
text in docs/ can never drift apart silently.

Example:
    python tools/make_diagrams.py            # writes docs/assets/*.drawio
    python tools/make_diagrams.py --export   # also renders PNG via the draw.io CLI
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("docs/assets")

# ---------------------------------------------------------------- style kit
PALETTE = {
    "input": ("#0E141B", "#3A4C5E", "#E8EEF4"),
    "frozen": ("#16212C", "#4C9AFF", "#CFE3FF"),
    "train": ("#16212C", "#F5C542", "#FFEFC2"),
    "fuse": ("#1B2A22", "#30A46C", "#D6F5E3"),
    "decode": ("#241A1B", "#E5484D", "#FFD9DB"),
    "output": ("#101820", "#8695A4", "#E8EEF4"),
    "owner": ("#0B0F14", "#F5C542", "#F5C542"),
}


_NEWLINE = chr(0)


def esc(text: str) -> str:
    """Escape a cell value so the file stays well-formed XML.

    draw.io labels are HTML (``html=1``), so ``<b>``/``<br/>`` are legitimate
    *content* but illegal as raw XML text. They must be escaped: draw.io decodes
    the entities and then renders the markup. ``&#xa;`` is kept as a real
    newline entity instead of becoming literal text.
    """
    text = text.replace("&#xa;", _NEWLINE)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace(_NEWLINE, "&#xa;")


def _style(kind: str, *, dashed: bool = False, rounded: bool = True) -> str:
    fill, stroke, font = PALETTE[kind]
    dash = "dashed=1;" if dashed else ""
    return (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};{dash}"
            f"fontFamily=Helvetica;fontSize=17;verticalAlign=middle;arcSize=6;")


def box(cells, cid, x, y, w, h, label, kind="train", dashed=False, parent="1",
        font_size=None, align="center", valign="middle"):
    style = _style(kind, dashed=dashed)
    if font_size:
        style += f"fontSize={font_size};"
    if align != "center":
        style += f"align={align};"
    if valign != "middle":
        style += f"verticalAlign={valign};"
    cells.append(
        f'<mxCell id="{cid}" value="{esc(label)}" style="{style}" vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')


def label(cells, cid, x, y, w, h, text, color="#93A2B1", size=15, align="left"):
    cells.append(
        f'<mxCell id="{cid}" value="{esc(text)}" style="text;html=1;align={align};'
        f'verticalAlign=middle;fontColor={color};fontSize={size};fontFamily=Helvetica;'
        f'fontStyle=1;letterSpacing=1;" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')


def edge(cells, cid, src, dst, text="", *, dashed=False, color="#4C6A85",
         width=1.5, exit_=None, entry=None, waypoints=None, font_color="#93A2B1"):
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
             f"html=1;strokeColor={color};strokeWidth={width};endArrow=block;"
             f"endFill=1;fontColor={font_color};fontSize=10;labelBackgroundColor=none;")
    if dashed:
        style += "dashed=1;"
    if exit_:
        style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    pts = ""
    if waypoints:
        inner = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in waypoints)
        pts = f'<Array as="points">{inner}</Array>'
    cells.append(
        f'<mxCell id="{cid}" value="{esc(text)}" style="{style}" edge="1" parent="1" '
        f'source="{src}" target="{dst}">'
        f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')


def wrap(name, cells, width, height):
    body = "\n        ".join(cells)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" version="26.0.0">
  <diagram name="{name}" id="{name}">
    <mxGraphModel dx="{width}" dy="{height}" grid="0" gridSize="10" guides="1"
                  tooltips="1" connect="1" arrows="1" fold="1" page="0"
                  pageScale="1" pageWidth="{width}" pageHeight="{height}"
                  math="0" shadow="0" background="#0B0F14">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# ---------------------------------------------------------------- diagram 1
def model_diagram():
    """GLIDE-Net itself: hybrid encoder -> CMF -> MSDEF decoder."""
    c, W, H = [], 1680, 940

    label(c, "t", 40, 22, 700, 22, "GLIDE-NET · 分割分支 ARCHITECTURE")

    # inputs
    box(c, "rgb", 40, 150, 150, 54, "RGB 图像&#xa;3 × 352 × 352", "input")
    box(c, "dep", 40, 240, 150, 54, "深度图&#xa;1 × 352 × 352", "input")

    # transformer branch
    box(c, "vit", 250, 108, 210, 250,
        "<b>Transformer 分支</b><br/><br/>SAM2 Hiera-L<br/>（主干冻结）<br/><br/>"
        "Adapter ×44<br/><span style='font-size:15px'>dim→32→dim, GELU</span>", "frozen")
    for i, (yy, ch) in enumerate([(118, 144), (180, 288), (242, 576), (304, 1152)], start=1):
        box(c, f"xt{i}", 480, yy, 76, 46, f"x{i}<sup>t</sup><br/><span style='font-size:13px'>{ch}</span>",
            "frozen", font_size=10)

    # cnn branch
    box(c, "cnnr", 250, 420, 150, 96, "<b>CNN 分支 · RGB</b><br/>ConvNeXt-style<br/>2 blocks / stage", "train")
    box(c, "cnnd", 250, 552, 150, 96, "<b>CNN 分支 · Depth</b><br/>ConvNeXt-style<br/>2 blocks / stage", "train")
    for i, (yy, ch) in enumerate([(422, 144), (466, 288), (510, 576)], start=1):
        box(c, f"xr{i}", 430, yy, 70, 38, f"x{i}<sup>r</sup><br/><span style='font-size:12px'>{ch}</span>",
            "train", font_size=9)
    for i, (yy, ch) in enumerate([(554, 144), (598, 288), (642, 576)], start=1):
        box(c, f"xd{i}", 430, yy, 70, 38, f"x{i}<sup>d</sup><br/><span style='font-size:12px'>{ch}</span>",
            "train", font_size=9)

    # CMF stack
    for i, yy in enumerate([108, 420, 552], start=1):
        box(c, f"cmf{i}", 600, yy, 150, 54,
            f"<b>CMF{i}</b><br/><span style='font-size:14px'>CE → SE → ⊕ RGB</span>", "fuse")
    box(c, "cmfnote", 560, 700, 240, 70,
        "<span style='font-size:14px'>CE: sigmoid(MLP(GMP[Fr,Fd]))⊙Fd<br/>"
        "SE: sigmoid(Conv7×(Max_c))⊙CE</span>", "fuse", dashed=True, font_size=10)

    # fusion adders
    for i, yy in enumerate([108, 420, 552], start=1):
        c.append(
            f'<mxCell id="add{i}" value="⊕" style="ellipse;whiteSpace=wrap;html=1;'
            f'fillColor=#1B2A22;strokeColor=#30A46C;fontColor=#D6F5E3;fontSize=16;" '
            f'vertex="1" parent="1"><mxGeometry x="790" y="{yy + 8}" width="38" height="38" '
            f'as="geometry" /></mxCell>')
        box(c, f"xf{i}", 860, yy + 6, 66, 42, f"x{i}<br/><span style='font-size:13px'>融合</span>",
            "fuse", font_size=10)
    box(c, "xf4", 860, 296, 66, 42, "x4<br/><span style='font-size:13px'>1152</span>", "fuse", font_size=10)

    # decoder
    box(c, "msdef", 990, 108, 250, 300,
        "<b>MSDEF 解码器</b><br/><br/>"
        "<span style='font-size:15px'>stage1: x3 ← WD(x2,x3,x4)<br/>"
        "stage2: x2 ← WD(x1,x2,x3)<br/>stage3: x1 ← WD(x1,x2)</span><br/><br/>"
        "<span style='font-size:15px'>DEU = SA ⊙ CA ⊙ x + LFP<br/>"
        "WD = Softmax(MLP(GAP(concat)))</span>", "decode")
    for i, yy in enumerate([130, 226, 322], start=1):
        box(c, f"side{i}", 1300, yy, 96, 30, f"side{i} 输出", "output", font_size=10)
    box(c, "out", 1300, 386, 96, 40, "<b>掩膜</b><br/>1×352×352", "output", font_size=10)
    box(c, "loss", 1300, 470, 200, 96,
        "<b>深监督</b><br/><span style='font-size:15px'>L = Σ structure_loss<br/>"
        "(BCE + weighted IoU)<br/>三项等权</span>", "output", font_size=10)

    # ---- edges
    edge(c, "e1", "rgb", "vit", "RGB", exit_=(1, 0.5), entry=(0, 0.2))
    edge(c, "e2", "rgb", "cnnr", "RGB", exit_=(1, 0.85), entry=(0, 0.2))
    edge(c, "e3", "dep", "cnnd", "深度", exit_=(1, 0.5), entry=(0, 0.2))
    edge(c, "e4", "vit", "xt1", "x1..x4", exit_=(1, 0.1), entry=(0, 0.5))
    edge(c, "e5", "cnnr", "xr1", "", exit_=(1, 0.2), entry=(0, 0.5))
    edge(c, "e6", "cnnd", "xd1", "", exit_=(1, 0.2), entry=(0, 0.5))
    for i in (1, 2, 3):
        edge(c, f"e7{i}", f"xr{i}", f"cmf{i}", "", exit_=(1, 0.5), entry=(0, 0.25))
        edge(c, f"e8{i}", f"xd{i}", f"cmf{i}", "", exit_=(1, 0.5), entry=(0, 0.75))
        edge(c, f"e9{i}", f"cmf{i}", f"add{i}", "", exit_=(1, 0.5), entry=(0, 0.5))
        edge(c, f"e10{i}", f"xt{i}", f"add{i}", "", exit_=(1, 0.5), entry=(1, 0.5),
             waypoints=[(770, 131 + (i - 1) * 92)])
        edge(c, f"e11{i}", f"add{i}", f"xf{i}", "", exit_=(1, 0.5), entry=(0, 0.5))
        edge(c, f"e12{i}", f"xf{i}", "msdef", "", exit_=(1, 0.5), entry=(0, 0.12 + 0.14 * (i - 1)))
    edge(c, "e13", "xt4", "xf4", "", exit_=(1, 0.5), entry=(0, 0.5), dashed=True)
    edge(c, "e14", "xf4", "msdef", "bottleneck", exit_=(1, 0.5), entry=(0, 0.07))
    for i in (1, 2, 3):
        edge(c, f"e15{i}", "msdef", f"side{i}", "", exit_=(1, 0.16 + 0.16 * (i - 1)),
             entry=(0, 0.5))
    edge(c, "e16", "msdef", "out", "head", exit_=(1, 0.62), entry=(0, 0.5))
    edge(c, "e17", "out", "loss", "", exit_=(0.5, 1), entry=(0.3, 0))
    edge(c, "e18", "side1", "loss", "", exit_=(0.5, 1), entry=(0.5, 0), dashed=True)
    return wrap("glide-net-architecture", c, W, H)


# ---------------------------------------------------------------- diagram 2
def system_diagram():
    """The course-practice system around the model: data, models, service, UI."""
    c, W, H = [], 1680, 650
    label(c, "t", 40, 22, 900, 22, "缺陷检测系统 · SYSTEM ARCHITECTURE")

    # data layer
    box(c, "d1", 40, 110, 210, 74, "<b>NEU RSDDS-AUG</b><br/>1 862 张 RGB-D<br/>训练 1 500 / 测试 362", "input")
    box(c, "d2", 40, 208, 210, 74, "<b>NEU-DET</b><br/>1 797 张 · 6 类<br/>边界框标注", "input")
    box(c, "d3", 40, 306, 210, 74, "<b>Deep PCB / 现场图</b><br/>（可选扩展）", "input")
    box(c, "d4", 40, 404, 210, 60, "dataset.py / data.yaml<br/>配对 · 归一化 · 增强", "input", font_size=11)

    # model layer
    box(c, "m1", 320, 110, 230, 130,
        "<b>分割 · GLIDE-Net</b><br/><span style='font-size:15px'>SAM2-Hiera + ConvNeXt<br/>"
        "+ CMF + MSDEF<br/>352×352 · 4 通道</span>", "frozen")
    box(c, "m2", 320, 264, 230, 120,
        "<b>检测 · YOLOv8m</b><br/><span style='font-size:15px'>NEU-DET 六类<br/>"
        "768×768 · mAP@0.5 ≥ 0.85 目标</span>", "train")
    box(c, "m3", 320, 408, 230, 74,
        "<b>分级 · 规则引擎</b><br/>面积 × 细长 × 置信", "decode")

    # service layer
    box(c, "s1", 620, 110, 230, 74, "<b>app/engines/</b><br/>preprocess · severity<br/>pipeline 融合", "fuse")
    box(c, "s2", 620, 208, 230, 74, "<b>SegmentationEngine</b><br/>PyTorch / ONNX Runtime", "fuse")
    box(c, "s3", 620, 306, 230, 74, "<b>DetectionEngine</b><br/>Ultralytics YOLO", "fuse")
    box(c, "s4", 620, 404, 230, 74, "<b>app/batch.py</b><br/>文件夹级流水线 · CSV", "fuse")

    # deployment
    box(c, "p1", 920, 110, 200, 56, "ckpt/*.pth", "output")
    box(c, "p2", 920, 190, 200, 56, "deploy/onnx/*.onnx", "output")
    box(c, "p3", 920, 270, 200, 56, "TensorRT / CPU EP", "output")
    box(c, "p4", 920, 350, 200, 56, "deploy/benchmark.py", "output")
    box(c, "p5", 920, 430, 200, 56, "延迟 / 保真度对照", "output")

    # presentation
    box(c, "v1", 1200, 110, 220, 110,
        "<b>Gradio 工作站</b><br/><span style='font-size:15px'>单图检测<br/>批量检测<br/>指标看板</span>", "train")
    box(c, "v2", 1200, 250, 220, 74, "<b>叠加图 / 四联图</b><br/>Pillow 渲染", "train")
    box(c, "v3", 1200, 348, 220, 74, "<b>报告 JSON / CSV</b><br/>逐缺陷明细", "train")
    box(c, "v4", 1200, 446, 220, 74, "<b>docs/</b><br/>架构 · 实验 · 分工", "input")

    # offline eval strip
    box(c, "e1", 320, 520, 500, 92,
        "<b>离线评测链路</b>　train.py → test.py → eval.py → predictions/metrics.csv<br/>"
        "<span style='font-size:15px'>S-measure · max-F · max-E · MAE（py_sod_metrics，362 张）</span>",
        "output", font_size=11, align="left")
    # edges: offline (dashed) vs online (solid)
    edge(c, "a1", "d1", "m1", "", dashed=True, exit_=(1, 0.5), entry=(0, 0.3))
    edge(c, "a2", "d2", "m2", "", dashed=True, exit_=(1, 0.5), entry=(0, 0.5))
    edge(c, "a3", "d3", "m2", "", dashed=True, exit_=(1, 0.3), entry=(0, 0.85))
    edge(c, "a4", "d4", "d1", "", dashed=True, exit_=(0.5, 0), entry=(0.5, 1))
    edge(c, "a5", "d4", "d2", "", dashed=True, exit_=(1, 0.5), entry=(1, 0.5))
    edge(c, "a6", "m1", "s2", "", exit_=(1, 0.4), entry=(0, 0.5))
    edge(c, "a7", "m2", "s3", "", exit_=(1, 0.5), entry=(0, 0.5))
    edge(c, "a8", "m3", "s1", "", exit_=(1, 0.5), entry=(0, 1))
    edge(c, "a9", "s2", "s1", "", exit_=(0.5, 0), entry=(0.5, 1))
    edge(c, "a10", "s3", "s1", "", exit_=(0.5, 0), entry=(1, 0.5))
    edge(c, "a11", "s1", "v1", "", exit_=(1, 0.5), entry=(0, 0.5),
         waypoints=[(1160, 147), (1160, 165)])
    edge(c, "a12", "s2", "v3", "", dashed=True, exit_=(1, 0.9), entry=(0, 0.5),
         waypoints=[(1160, 315), (1160, 383)])
    edge(c, "a13", "p1", "s2", "", exit_=(0, 0.5), entry=(1, 0.2))
    edge(c, "a14", "p2", "s2", "", exit_=(0, 0.5), entry=(1, 0.8))
    edge(c, "a15", "p3", "p2", "", dashed=True, exit_=(0.5, 0), entry=(0.5, 1))
    edge(c, "a16", "p4", "p5", "", exit_=(0.5, 1), entry=(0.5, 0))
    edge(c, "a17", "p2", "p4", "", dashed=True, exit_=(0.9, 1), entry=(0.1, 0))
    edge(c, "a18", "s1", "e1", "", dashed=True, exit_=(0.2, 1), entry=(0.1, 0),
         waypoints=[(666, 530), (370, 530)])
    edge(c, "a21", "s2", "s4", "", exit_=(0.5, 1), entry=(0.5, 0))
    return wrap("system-architecture", c, W, H)


# ---------------------------------------------------------------- diagram 3
def pipeline_diagram():
    """Single-image inspection flow with latency budget annotations."""
    c, W, H = [], 1680, 470
    label(c, "t", 40, 22, 900, 22, "单图检测流水线 · INSPECTION PIPELINE")

    steps = [
        ("s1", 40, "上传输入", "RGB + 深度&#xa;EXIF 校正"),
        ("s2", 236, "预处理", "352×352&#xa;ImageNet 归一化"),
        ("s3", 432, "分割推理", "GLIDE-Net&#xa;✓ 用户选择"),
        ("s4", 628, "检测推理", "YOLOv8m&#xa;✓ 用户选择"),
        ("s5", 824, "掩膜后处理", "min-max 拉伸&#xa;阈值二值化"),
        ("s6", 1020, "连通域", "8 邻域标记&#xa;面积过滤"),
        ("s7", 1216, "融合", "IoU 继承类别&#xa;置信度对齐"),
        ("s8", 1412, "分级与报告", "评分 → 三级&#xa;JSON / CSV"),
    ]
    for cid, x, title, sub in steps:
        kind = "train" if cid in ("s3", "s4") else "fuse" if cid in ("s7", "s8") else "input"
        box(c, cid, x, 130, 150, 86, f"<b>{title}</b><br/><span style='font-size:14px'>{sub}</span>",
            kind, font_size=12)
        if cid != "s8":
            nxt = f"s{int(cid[1:]) + 1}"
            edge(c, f"p{cid}", cid, nxt, "", exit_=(1, 0.5), entry=(0, 0.5))

    box(c, "lat1", 432, 260, 150, 44, "≈ 40 ms<br/><span style='font-size:13px'>fp16 / cuda</span>",
        "output", font_size=11)
    box(c, "lat2", 628, 260, 150, 44, "≈ 11 ms<br/><span style='font-size:13px'>768×768</span>",
        "output", font_size=11)
    edge(c, "l1", "s3", "lat1", "", dashed=True, exit_=(0.5, 1), entry=(0.5, 0))
    edge(c, "l2", "s4", "lat2", "", dashed=True, exit_=(0.5, 1), entry=(0.5, 0))

    box(c, "o1", 1412, 340, 150, 66, "<b>叠加图</b><br/>分级配色", "decode", font_size=11)
    box(c, "o2", 1216, 340, 150, 66, "<b>四联视图</b><br/>RGB/深度/概率/结果", "decode", font_size=11)
    box(c, "o3", 1020, 340, 150, 66, "<b>REST / Gradio</b><br/>JSON 报告", "decode", font_size=11)
    edge(c, "q1", "s8", "o1", "", exit_=(0.5, 1), entry=(0.5, 0))
    edge(c, "q2", "s7", "o2", "", exit_=(0.5, 1), entry=(0.5, 0))
    edge(c, "q3", "s6", "o3", "", exit_=(0.5, 1), entry=(0.5, 0))

    return wrap("inspection-pipeline", c, W, H)


# ---------------------------------------------------------------- validation
def _vertices(xml: str):
    """Yield (id, x, y, w, h, is_edge, is_text) for every cell of a diagram."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)          # also asserts the file is well-formed
    for cell in root.iter("mxCell"):
        style = cell.get("style") or ""
        geo = cell.find("mxGeometry")
        if cell.get("vertex") != "1" or geo is None:
            continue
        if not style.startswith("text;"):        # plain labels overlap freely
            yield (cell.get("id"), float(geo.get("x", 0)), float(geo.get("y", 0)),
                   float(geo.get("width", 0)), float(geo.get("height", 0)),
                   style.startswith("text;"))


def check_layout(xml: str, name: str, page: tuple[int, int]) -> list[str]:
    """Reject diagrams that silently break: bad XML, boxes off-page or stacked.

    The PNG is what reviewers see, so an overlap here is a real defect — two
    boxes drawn on top of each other look like one garbled box. This catches it
    at generation time instead of in the review.
    """
    problems = []
    boxes = list(_vertices(xml))
    page_w, page_h = page
    for cid, x, y, w, h, _ in boxes:
        if x < 0 or y < 0 or x + w > page_w or y + h > page_h:
            problems.append(f"{cid} 超出画布 ({x},{y},{w},{h}) vs {page_w}x{page_h}")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ai, ax, ay, aw, ah, _ = a
            bi, bx, by, bw, bh, _ = b
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox > 2 and oy > 2:                # 2 px tolerance for shared edges
                inter = ox * oy
                if inter / min(aw * ah, bw * bh) > 0.35:
                    problems.append(f"{ai} 与 {bi} 重叠 {ox:.0f}×{oy:.0f}")
    if problems:
        print(f"[layout] {name}: {len(problems)} 处问题", file=sys.stderr)
        for msg in problems:
            print("   ", msg, file=sys.stderr)
    else:
        print(f"[layout] {name}: ok ({len(boxes)} 个盒子)")
    return problems


# ---------------------------------------------------------------- export
def find_drawio():
    for name in ("drawio", "draw.io"):
        path = shutil.which(name)
        if path:
            return path
    for cand in ("C:/Users/Jerry/AppData/Local/Programs/draw.io/draw.io.exe",
                 "/Applications/draw.io.app/Contents/MacOS/draw.io",
                 "/usr/bin/drawio"):
        if Path(cand).exists():
            return cand
    return None


def export(drawio: str, files: list[Path], width: int = 2000):
    outs = []
    for src in files:
        png = src.with_suffix(".png")
        subprocess.run([drawio, "-x", "-f", "png", "--width", str(width),
                        "-o", str(png), str(src)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs.append(png)
    return outs


def main():
    ap = argparse.ArgumentParser(description="Generate docs/assets/*.drawio")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export", action="store_true", help="render PNG via draw.io CLI")
    ap.add_argument("--width", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "glide-net-architecture.drawio": model_diagram(),
        "system-architecture.drawio": system_diagram(),
        "inspection-pipeline.drawio": pipeline_diagram(),
    }
    pages = {"glide-net-architecture.drawio": (1680, 940),
             "system-architecture.drawio": (1680, 900),
             "inspection-pipeline.drawio": (1680, 470)}
    written, problems = [], []
    for name, xml in diagrams.items():
        path = out / name
        path.write_text(xml, encoding="utf-8")
        written.append(path)
        print(f"wrote {path}")
        problems += check_layout(xml, name, pages[name])
    if problems:
        print(f"abort: {len(problems)} 处布局问题，先修再导出", file=sys.stderr)
        return 2

    if args.export:
        binary = find_drawio()
        if not binary:
            print("draw.io CLI not found; install it or open the .drawio files manually",
                  file=sys.stderr)
            return 1
        for png in export(binary, written, args.width):
            print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
