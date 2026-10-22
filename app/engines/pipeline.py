"""End-to-end inspection pipeline: detect -> segment -> fuse -> grade.

One image in, one report out. The report is a plain JSON-serialisable dict so
that the Gradio UI, the batch CLI and the evaluation scripts all consume the
exact same structure.

Fusion rules
------------
The two branches are complementary and neither is trusted blindly:

* a segmentation blob that overlaps a detection box inherits that box's class
  label (the detector is the only branch that knows *what* the defect is);
* a detection without any blob keeps its box but gets ``mask_iou = 0`` and the
  region is sized from the box, flagged ``severity="light"`` unless its score
  says otherwise -- this keeps pure-detector defects visible in the report;
* the report's ``overall`` grade is the worst graded region, and
  ``defect_area_ratio`` comes from the mask (pixel accurate).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .detection import CLASS_ZH, Detection, DetectionEngine
from .preprocess import load_gray, load_rgb, prob_to_mask
from .segmentation import SegmentationEngine
from .severity import DefectRegion, _grade, attach_labels, extract_regions


@dataclass
class InspectionReport:
    image_size: tuple[int, int]
    prob: np.ndarray
    mask: np.ndarray
    detections: list[Detection]
    regions: list[DefectRegion]
    summary: dict
    timings: dict

    def as_dict(self) -> dict:
        return {
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "detections": [d.as_dict() for d in self.detections],
            "regions": [r.as_dict() for r in self.regions],
            "summary": self.summary,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }


class InspectionPipeline:
    """Detector + segmenter + grader, sharing one preprocessing contract."""

    def __init__(self, segmenter: SegmentationEngine,
                 detector: Optional[DetectionEngine] = None,
                 mask_threshold: float = 0.5):
        self.segmenter = segmenter
        self.detector = detector
        self.mask_threshold = mask_threshold

    def run(self, rgb, depth=None, *, use_detector: bool = True,
            threshold: Optional[float] = None) -> InspectionReport:
        """Inspect one image. ``threshold`` overrides the instance default so the
        UI slider changes the mask that is graded, not just a label."""
        t0 = time.perf_counter()
        image = load_rgb(rgb)
        depth_img = load_gray(depth) if depth is not None else None

        thr = self.mask_threshold if threshold is None else float(threshold)
        seg = self.segmenter.predict(image, depth_img, threshold=thr)
        t_seg = time.perf_counter() - t0

        dets: list[Detection] = []
        t_det = 0.0
        if self.detector is not None and use_detector:
            t1 = time.perf_counter()
            # the detector sees RGB only: NEU-DET is trained on 3-channel images
            det = self.detector.predict(np.asarray(image, dtype=np.uint8))
            dets = det.detections
            t_det = time.perf_counter() - t1

        regions = extract_regions(seg.mask, seg.prob)
        if dets:
            regions = attach_labels(regions, [d.as_dict() for d in dets])
        summary = _summarise(regions, seg.mask, dets)

        timings = {"segment_ms": seg.latency_ms, "detect_ms": t_det * 1000.0,
                   "total_ms": (time.perf_counter() - t0) * 1000.0}
        return InspectionReport(image_size=seg.size, prob=seg.prob, mask=seg.mask,
                                detections=dets, regions=regions, summary=summary,
                                timings=timings)


def _summarise(regions: list[DefectRegion], mask: np.ndarray,
               detections: list[Detection]) -> dict:
    counts = {"light": 0, "medium": 0, "severe": 0}
    for region in regions:
        counts[region.severity] += 1
    labelled: dict[str, int] = {}
    for region in regions:
        if region.label:
            labelled[region.label] = labelled.get(region.label, 0) + 1
    worst = max((r.score for r in regions), default=0.0)
    # one source of truth for the grade bands (app/engines/severity.py)
    overall = _grade(worst) if regions else "none"
    return {
        "verdict": overall,
        "defect_count": len(regions),
        "detection_count": len(detections),
        "severity_counts": counts,
        "classes": {CLASS_ZH.get(k, k): v for k, v in labelled.items()},
        "defect_area_ratio": round(float(mask.mean()) if mask.size else 0.0, 5),
        "worst_score": round(worst, 1),
    }


def build_pipeline(segmenter_ckpt: Optional[str] = None,
                   onnx_path: Optional[str] = None,
                   detector_weights: Optional[str] = None,
                   device: str = "cuda", fp16: bool = True) -> InspectionPipeline:
    """Convenience factory used by the app, the CLI and the tests."""
    segmenter = SegmentationEngine(checkpoint=segmenter_ckpt, onnx_path=onnx_path,
                                  device=device, fp16=fp16)
    detector = DetectionEngine(detector_weights) if detector_weights else None
    return InspectionPipeline(segmenter=segmenter, detector=detector)
