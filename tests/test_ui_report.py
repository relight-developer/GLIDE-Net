"""Tests for the report HTML fragments and the overlay renderer.

Both are pure-Python (strings + Pillow), so they run on CPU without a model.
The point is to lock the *contract*: the UI must never crash on an empty report,
and the severity wording must stay in sync with ``app/engines/severity.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engines.pipeline import InspectionReport  # noqa: E402
from app.engines.severity import GRADE_THRESHOLDS, extract_regions  # noqa: E402
from app.ui import report as R  # noqa: E402


def _fake_report(severity: str = "severe", defects: int = 1) -> InspectionReport:
    mask = np.zeros((80, 120), dtype=np.uint8)
    prob = np.zeros((80, 120), dtype=np.float32)
    if defects:
        mask[10:40, 10:40] = 1
        prob[10:40, 10:40] = 0.93
    regions = extract_regions(mask, prob)
    summary = {
        "verdict": severity if regions else "none",
        "defect_count": len(regions),
        "detection_count": 1 if regions else 0,
        "severity_counts": {"light": 0, "medium": 0, "severe": len(regions)},
        "classes": {"斑块": 1} if regions else {},
        "defect_area_ratio": float(mask.mean()),
        "worst_score": regions[0].score if regions else 0.0,
    }
    return InspectionReport(image_size=(120, 80), prob=prob, mask=mask,
                            detections=[], regions=regions, summary=summary,
                            timings={"segment_ms": 12.0, "detect_ms": 3.0, "total_ms": 20.0})


def test_empty_report_renders_without_defects():
    html = R.empty_report()
    assert 'id="report"' in html
    assert "待检测" in html
    # every gauge must show a placeholder rather than "None"
    assert "None" not in html


def test_report_sheet_shows_verdict_and_count():
    html = R.report_sheet(_fake_report("severe"))
    assert 'id="report"' in html
    assert R.SEVERITY_ZH["severe"] in html
    assert R.VERDICT_SENTENCE["severe"] in html
    assert "严重" in html


def test_report_sheet_on_empty_mask_says_pass():
    report = _fake_report(severity="none", defects=0)
    html = R.report_sheet(report)
    assert R.SEVERITY_ZH["none"] in html or "合格" in html


def test_severity_labels_cover_every_grade():
    for grade in GRADE_THRESHOLDS:
        assert grade in R.SEVERITY_ZH
        assert grade in R.VERDICT_SENTENCE
    assert "none" in R.SEVERITY_ZH


def test_status_line_marks_state():
    ok = R.status_line("完成", "ok")
    err = R.status_line("失败", "err")
    assert 'id="status"' in ok and "完成" in ok
    assert 'class="err"' in err


def test_metric_tiles_render_rows():
    html = R.metric_tiles([{"title": "对比", "note": "362 张",
                            "columns": ["方法", "S"], "rows": [["Ours", "0.8718"]]}])
    assert "对比" in html and "Ours" in html and "0.8718" in html


def test_overlay_and_panel_render():
    from PIL import Image

    from app.viz import render_overlay, render_panel

    image = Image.new("RGB", (120, 80), (40, 60, 80))
    depth = Image.new("L", (120, 80), 128)
    report = _fake_report("severe")
    overlay = render_overlay(image, report)
    assert overlay.size == image.size
    panel = render_panel(image, depth, report)
    # four cells side by side, each padded -> wider than a single frame
    assert panel.width > image.width * 3
    assert panel.height >= image.height


def test_overlay_handles_no_defects():
    from PIL import Image

    from app.viz import render_overlay

    image = Image.new("RGB", (64, 48), (0, 0, 0))
    overlay = render_overlay(image, _fake_report("none", defects=0))
    assert overlay.size == (64, 48)


def test_class_zh_matches_detector_vocabulary():
    from app.engines.detection import CLASS_ZH as ENGINE_ZH

    assert R.CLASS_ZH == ENGINE_ZH
