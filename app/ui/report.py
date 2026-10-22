"""HTML fragments for the inspection workstation.

Kept separate from the Gradio wiring so the layout can be unit-tested as
strings, and so the report sheet has exactly one implementation shared by the
single-image tab and the batch tab.
"""
from __future__ import annotations

import html
from typing import Iterable, Optional

SEVERITY_ZH = {"severe": "严重", "medium": "中等", "light": "轻微", "none": "合格"}
VERDICT_SENTENCE = {
    "severe": "存在严重缺陷，建议停机复检",
    "medium": "存在中等缺陷，建议安排复查",
    "light": "存在轻微缺陷，建议持续观察",
    "none": "未检出缺陷，判定合格",
}
CLASS_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "氧化皮压入",
    "scratches": "划伤",
}


def title_block(repo: str = "GLIDE-Net · NEU RSDDS-AUG", build: str = "课堂实践",
                backend: str = "—", device: str = "cuda") -> str:
    return f"""
<div id="titleblock">
  <div class="cell">
    <div class="k">Rail Surface Inspection · RGB-D Segmentation</div>
    <h1>钢轨表面缺陷<em>检测工作站</em></h1>
    <div class="spec">{html.escape(repo)}　·　{html.escape(build)}</div>
  </div>
  <div class="cell">
    <div class="k">推理后端</div>
    <div class="v small">{html.escape(backend)}</div>
    <div class="spec">{html.escape(device)}</div>
  </div>
  <div class="cell">
    <div class="k">输入规格</div>
    <div class="v small">352 × 352</div>
    <div class="spec">RGB + DEPTH</div>
  </div>
  <div class="cell">
    <div class="k">判定等级</div>
    <div class="v small">三级</div>
    <div class="spec">LIGHT / MEDIUM / SEVERE</div>
  </div>
</div>
"""


def status_line(text: str, kind: str = "ok") -> str:
    cls = "err" if kind == "err" else ""
    return f'<div id="status" class="{cls}"><b>{html.escape(kind.upper())}</b>　{html.escape(text)}</div>'


def report_sheet(report_dict: dict, elapsed_ms: float = 0.0,
                 note: Optional[str] = None) -> str:
    """The inspection sheet: verdict banner, gauges, per-defect rows."""
    summary = report_dict["summary"]
    verdict = summary.get("verdict", "none")
    counts = summary.get("severity_counts", {})
    regions = report_dict.get("regions", [])
    classes = summary.get("classes") or {}

    gauges = [
        ("缺陷区域", f"{summary.get('defect_count', 0)}", "处"),
        ("缺陷面积", f"{summary.get('defect_area_ratio', 0) * 100:.2f}", "%"),
        ("最高评分", f"{summary.get('worst_score', 0):.1f}", "/100"),
        ("检测框", f"{summary.get('detection_count', 0)}", "个"),
        ("严重", f"{counts.get('severe', 0)}", "处"),
        ("耗时", f"{elapsed_ms:.0f}", "ms"),
    ]
    gauge_html = "".join(
        f'<div class="gauge"><div class="n">{v}<small>{unit}</small></div>'
        f'<div class="k">{html.escape(k)}</div></div>' for k, v, unit in gauges
    )

    class_html = "".join(
        f'<div class="row"><div class="tag">·</div><div>{html.escape(name)}</div>'
        f'<div class="score">{num} 处</div><div class="sev light">识别</div></div>'
        for name, num in classes.items())

    rows = []
    for i, region in enumerate(regions[:12], 1):
        label = region.get("label")
        name = CLASS_ZH.get(label, label) if label else "未分类缺陷"
        conf = region.get("label_conf")
        conf_txt = f" · {conf * 100:.0f}%" if conf else ""
        rows.append(
            f'<div class="row"><div class="tag">{i:02d}</div>'
            f'<div>{html.escape(str(name))}{conf_txt}'
            f'<span style="color:#8695A4;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:10.5px"> · {region["w"]}×{region["h"]}px · '
            f'面积 {region["area_ratio"] * 100:.2f}%</span></div>'
            f'<div class="score">{region["score"]:.1f}</div>'
            f'<div class="sev {region["severity"]}">{SEVERITY_ZH[region["severity"]]}</div></div>')
    row_html = "".join(rows) or '<div class="empty">未检出连通缺陷区域。</div>'

    note = note or ("评分 = 面积项 × 细长项 × 置信项，阈值 26 / 55 分对应中等 / 严重。"
                    "面积项以 12% 画面占比为满分参考；细长项奖励划伤类长条缺陷。")
    return f"""
<div id="report">
  <div class="head">
    <div class="verdict {verdict}">{SEVERITY_ZH.get(verdict, verdict)}</div>
    <div class="sub">{html.escape(VERDICT_SENTENCE.get(verdict, ""))}</div>
  </div>
  <div class="gauges">{gauge_html}</div>
  {class_html}
  {row_html}
  <div class="note">{html.escape(note)}</div>
</div>
"""


def empty_report() -> str:
    return """
<div id="report">
  <div class="head">
    <div class="verdict none" style="color:#8695A4">待检测</div>
    <div class="sub">上传待检图像，点击「开始检测」</div>
  </div>
  <div class="gauges">
    <div class="gauge"><div class="n">—</div><div class="k">缺陷区域</div></div>
    <div class="gauge"><div class="n">—</div><div class="k">缺陷面积</div></div>
    <div class="gauge"><div class="n">—</div><div class="k">最高评分</div></div>
    <div class="gauge"><div class="n">—</div><div class="k">检测框</div></div>
    <div class="gauge"><div class="n">—</div><div class="k">严重</div></div>
    <div class="gauge"><div class="n">—</div><div class="k">耗时</div></div>
  </div>
  <div class="empty">RGB 与深度图上传后，系统将输出缺陷掩膜、类别与严重程度分级。</div>
</div>
"""


def metric_tiles(items: Iterable[dict]) -> str:
    """Small stack of metric tables rendered as tiles."""
    out = []
    for tile in items:
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in tile["columns"])
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in tile["rows"])
        out.append(
            f'<div class="tile"><h3>{html.escape(tile["title"])}'
            f'<span>{html.escape(tile.get("note", ""))}</span></h3>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    return f'<div id="metrics">{"".join(out)}</div>'
