"""Defect grading: from binary mask to per-defect severity levels.

The grading rule is intentionally rule based so that an inspector can read the
score sheet and disagree with it. Three cues are combined:

``area``       share of the inspected surface covered by the blob
``elongation`` long side / short side of the bounding box (thin defects such as
               scratches and rolled-in scale are the ones that grow along the
               rolling direction)
``confidence`` mean predicted probability inside the blob

The score is on a 0-100 scale and mapped to ``light`` / ``medium`` / ``severe``
at 26 and 55. Thresholds and weights live in :data:`GRADE_THRESHOLDS` so the
report can quote exactly one source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np

GRADE_THRESHOLDS = {"light": 0.0, "medium": 26.0, "severe": 55.0}
# Calibrated on the 362 ground-truth masks of NEU RSDDS-AUG, whose defect area
# ratio has p10 = 0.10, p25 = 0.15, p50 = 0.25, p75 = 0.39, p90 = 0.52. Saturation
# at 0.55 means the light/medium/severe bands sit near p25 and p75, so the three
# grades stay populated instead of collapsing into one bucket.
AREA_REFERENCE = 0.55
ELONGATION_WEIGHT = 0.18
ELONGATION_REFERENCE = 6.0
MIN_AREA_RATIO = 0.0002    # ignore specks below 0.02% of the frame


@dataclass
class DefectRegion:
    """One connected defect blob."""

    x: int
    y: int
    w: int
    h: int
    area_px: int
    area_ratio: float
    elongation: float
    mean_conf: float
    score: float
    severity: str
    label: Optional[str] = None
    label_conf: Optional[float] = None
    components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _grade(score: float) -> str:
    if score >= GRADE_THRESHOLDS["severe"]:
        return "severe"
    if score >= GRADE_THRESHOLDS["medium"]:
        return "medium"
    return "light"


def score_region(area_ratio: float, elongation: float, mean_conf: float = 1.0) -> float:
    """Combine area, elongation and confidence into a 0-100 severity score."""
    area_term = min(1.0, area_ratio / AREA_REFERENCE)
    elong_term = 1.0 + ELONGATION_WEIGHT * min(1.0, max(0.0, elongation - 1.0) / ELONGATION_REFERENCE)
    conf_term = 0.75 + 0.25 * float(np.clip(mean_conf, 0.0, 1.0))
    return float(min(100.0, 100.0 * area_term * elong_term * conf_term))


def extract_regions(binary: np.ndarray, prob: Optional[np.ndarray] = None,
                    min_area_ratio: float = MIN_AREA_RATIO) -> list[DefectRegion]:
    """Label the binary mask and return one :class:`DefectRegion` per blob.

    ``binary`` is a {0, 1} uint8 array; ``prob`` the matching probability map.
    """
    from scipy import ndimage

    if binary.ndim != 2:
        raise ValueError("binary mask must be 2-D")
    labelled, count = ndimage.label(binary.astype(np.uint8), structure=np.ones((3, 3)))
    if count == 0:
        return []

    total = float(binary.size)
    regions: list[DefectRegion] = []
    for idx in range(1, count + 1):
        ys, xs = np.where(labelled == idx)
        if ys.size == 0:
            continue
        area_ratio = ys.size / total
        if area_ratio < min_area_ratio:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        long_side, short_side = max(w, h), max(1, min(w, h))
        elongation = long_side / short_side
        mean_conf = float(prob[ys, xs].mean()) if prob is not None else 1.0
        score = score_region(area_ratio, elongation, mean_conf)
        regions.append(DefectRegion(x=x0, y=y0, w=w, h=h, area_px=int(ys.size),
                                    area_ratio=round(area_ratio, 5),
                                    elongation=round(float(elongation), 2),
                                    mean_conf=round(mean_conf, 4),
                                    score=round(score, 1), severity=_grade(score),
                                    components={"sqrt_area": round(float(np.sqrt(ys.size)), 2),
                                                "linearity": round(1.0 - short_side / long_side, 3)}))
    regions.sort(key=lambda r: (r.score, r.area_px), reverse=True)
    return regions


def attach_labels(regions: Iterable[DefectRegion], detections: Iterable[dict],
                  iou_threshold: float = 0.2) -> list[DefectRegion]:
    """Give each mask blob the class of the YOLO box that covers it best.

    Boxes come from the detector branch, blobs from the segmentation branch; a
    blob takes the highest-scoring box whose intersection-over-union with the
    blob's bounding box exceeds ``iou_threshold``.
    """
    dets = list(detections)

    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    out: list[DefectRegion] = []
    for region in regions:
        box = (region.x, region.y, region.x + region.w, region.y + region.h)
        best, best_iou = None, iou_threshold
        for det in dets:
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
            value = iou(box, det_box)
            if value >= best_iou:
                best, best_iou = det, value
        if best is not None:
            region.label = best.get("label")
            region.label_conf = best.get("confidence")
        out.append(region)
    return out


def inspect_image(binary: np.ndarray, prob: Optional[np.ndarray] = None,
                  detections: Optional[Iterable[dict]] = None) -> dict:
    """Full grading report for one image."""
    regions = extract_regions(binary, prob)
    if detections:
        regions = attach_labels(regions, detections)
    counts = {"light": 0, "medium": 0, "severe": 0}
    for region in regions:
        counts[region.severity] += 1
    worst = max((r.score for r in regions), default=0.0)
    return {
        "defect_count": len(regions),
        "severity_counts": counts,
        "overall": _grade(worst) if regions else "none",
        "worst_score": round(worst, 1),
        "defect_area_ratio": round(float(binary.mean()) if binary.size else 0.0, 5),
        "regions": [r.as_dict() for r in regions[:50]],
        "thresholds": GRADE_THRESHOLDS,
    }
