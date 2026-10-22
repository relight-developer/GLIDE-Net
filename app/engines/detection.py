"""YOLOv8 defect-detection engine (6-class NEU-DET model).

The detector answers "what kind of defect is where"; the segmentation engine
answers "exactly which pixels are defective". Both are exposed through the same
:class:`DetectionResult` container so the pipeline can fuse them without
caring which runtime produced them.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

DEFAULT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
                   "rolled-in_scale", "scratches"]

CLASS_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "氧化皮压入",
    "scratches": "划伤",
}


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_dict(self) -> dict:
        return {"label": self.label, "label_zh": CLASS_ZH.get(self.label, self.label),
                "confidence": round(float(self.confidence), 4),
                "x1": int(self.x1), "y1": int(self.y1),
                "x2": int(self.x2), "y2": int(self.y2)}


@dataclass
class DetectionResult:
    detections: list[Detection]
    latency_ms: float
    backend: str
    class_counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"detections": [d.as_dict() for d in self.detections],
                "latency_ms": round(self.latency_ms, 2),
                "backend": self.backend,
                "class_counts": self.class_counts}


class DetectionEngine:
    """Thin, warm-up-able wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(self, weights: str, device: str = "0", imgsz: int = 768,
                 conf: float = 0.25, iou: float = 0.45, half: bool = True,
                 warmup: int = 1):
        from ultralytics import YOLO

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        self.backend = f"yolo-{os.path.basename(weights)}"
        if warmup:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            for _ in range(warmup):
                self._infer(dummy)

    def _infer(self, image: np.ndarray):
        return self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                  device=self.device, half=self.half, verbose=False,
                                  max_det=100)[0]

    def predict(self, image: np.ndarray, conf: Optional[float] = None) -> DetectionResult:
        """``image`` is an RGB uint8 array (H, W, 3)."""
        started = time.perf_counter()
        if conf is not None and conf != self.conf:
            self.conf = conf
        result = self._infer(image)
        latency = (time.perf_counter() - started) * 1000.0

        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, classes):
                detections.append(Detection(label=self.names[cls_id], confidence=float(score),
                                            x1=int(round(x1)), y1=int(round(y1)),
                                            x2=int(round(x2)), y2=int(round(y2))))
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return DetectionResult(detections=detections, latency_ms=latency,
                               backend=self.backend, class_counts=counts)

    def predict_batch(self, images: Iterable[np.ndarray]) -> list[DetectionResult]:
        return [self.predict(img) for img in images]
