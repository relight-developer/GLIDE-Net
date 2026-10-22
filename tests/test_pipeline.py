"""Unit tests for the inspection pipeline (no checkpoint, no GPU required).

Run with:
    python -m pytest tests -q

Everything here is CPU-only and uses synthetic masks, so the suite stays fast
and can run on a reviewer's laptop: the GPU paths are exercised by
``tools/eval_system.py`` and ``deploy/benchmark.py`` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engines.preprocess import (load_gray, load_rgb, normalise_prob,
                                    prob_to_mask, to_network_input)
from app.engines.severity import (GRADE_THRESHOLDS, _grade, attach_labels,
                                  extract_regions, inspect_image, score_region)


# ------------------------------------------------------------------ preprocessing
def test_network_input_shape_and_normalisation():
    from PIL import Image

    rgb = Image.new("RGB", (200, 100), (255, 255, 255))
    x, size = to_network_input(rgb, None, 352)
    assert x.shape == (1, 4, 352, 352)
    assert x.dtype == np.float32
    assert size == (200, 100)
    # white pixel, ImageNet-normalised: (1 - mean) / std
    assert x[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
    # no depth -> zero channel
    assert x[0, 3].max() == 0.0


def test_depth_is_stretched_to_uint8_range(tmp_path):
    from PIL import Image

    raw = (np.arange(100 * 100).reshape(100, 100) + 50_000).astype(np.uint16)
    path = tmp_path / "depth.png"
    Image.fromarray(raw).save(path)  # uint16 2-D -> mode "I;16"
    gray = load_gray(path)
    assert gray is not None
    values = np.asarray(gray)
    assert values.dtype == np.uint8
    assert values.min() == 0 and values.max() == 255


def test_normalise_and_threshold():
    prob = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    stretched = normalise_prob(prob)
    assert stretched.min() == pytest.approx(0.0, abs=1e-6)
    assert stretched.max() == pytest.approx(1.0, abs=1e-6)
    _, mask = prob_to_mask(stretched, (2, 2), threshold=0.5)
    assert mask.shape == (2, 2)
    assert mask.sum() == 2  # the two upper values survive


def test_load_rgb_accepts_paths_bytes_and_arrays(tmp_path):
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
    for src in (path, str(path), path.read_bytes(),
                np.zeros((24, 32, 3), dtype=np.uint8)):
        img = load_rgb(src)
        assert img.mode == "RGB"
        assert img.size == (32, 24)


# ------------------------------------------------------------------ severity
def test_grade_bands_are_monotonic():
    assert _grade(0.0) == "light"
    assert _grade(GRADE_THRESHOLDS["medium"] - 0.1) == "light"
    assert _grade(GRADE_THRESHOLDS["medium"]) == "medium"
    assert _grade(GRADE_THRESHOLDS["severe"] - 0.1) == "medium"
    assert _grade(GRADE_THRESHOLDS["severe"]) == "severe"


def test_score_is_bounded_and_monotonic_in_area():
    small = score_region(0.01, 1.0, 0.9)
    big = score_region(0.6, 1.0, 0.9)
    assert 0 < small < big <= 100.0
    assert score_region(1.0, 1.0, 1.0) == pytest.approx(100.0, abs=1e-6)


def test_score_rewards_elongation_and_penalises_low_confidence():
    base = score_region(0.2, 1.0, 0.95)
    assert score_region(0.2, 8.0, 0.95) > base
    assert score_region(0.2, 1.0, 0.55) < base


def _two_blob_mask(h=120, w=160):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:40, 10:40] = 1          # compact 30x30 square
    mask[70:74, 20:120] = 1         # long thin scratch
    return mask


def test_extract_regions_finds_both_blobs_and_sorts_by_score():
    mask = _two_blob_mask()
    regions = extract_regions(mask, prob=np.full(mask.shape, 0.9, dtype=np.float32))
    assert len(regions) == 2
    assert regions[0].score >= regions[1].score
    assert {r.area_px for r in regions} == {900, 400}
    assert all(r.severity in GRADE_THRESHOLDS for r in regions)


def test_extract_regions_ignores_specks():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[5, 5] = 1  # area ratio 2.5e-5, below MIN_AREA_RATIO
    assert extract_regions(mask) == []


def test_extract_regions_rejects_non_2d():
    with pytest.raises(ValueError):
        extract_regions(np.zeros((2, 2, 2), dtype=np.uint8))


def test_attach_labels_uses_best_overlapping_box():
    mask = _two_blob_mask()
    regions = extract_regions(mask)
    detections = [
        {"label": "patches", "confidence": 0.9, "x1": 5, "y1": 5, "x2": 45, "y2": 45},
        {"label": "scratches", "confidence": 0.8, "x1": 15, "y1": 65, "x2": 125, "y2": 80},
    ]
    labelled = attach_labels(regions, detections)
    labels = {r.label for r in labelled}
    assert labels == {"patches", "scratches"}
    assert all(r.label_conf is not None for r in labelled)


def test_attach_labels_leaves_unmatched_regions_unlabelled():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    regions = attach_labels(extract_regions(mask),
                           [{"label": "patches", "confidence": 0.9,
                             "x1": 80, "y1": 80, "x2": 95, "y2": 95}])
    assert regions[0].label is None


def test_inspect_image_reports_counts_and_worst_grade():
    mask = _two_blob_mask()
    report = inspect_image(mask, prob=np.full(mask.shape, 0.95, dtype=np.float32))
    assert report["defect_count"] == 2
    assert sum(report["severity_counts"].values()) == 2
    assert report["overall"] == _grade(report["worst_score"])
    assert 0 < report["defect_area_ratio"] < 1


def test_inspect_image_on_empty_mask():
    report = inspect_image(np.zeros((64, 64), dtype=np.uint8))
    assert report["defect_count"] == 0
    assert report["overall"] == "none"
    assert report["regions"] == []
