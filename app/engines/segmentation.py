"""GLIDE-Net segmentation engine (PyTorch or ONNX Runtime backend).

The engine hides three differences that matter to the service layer:
input padding/normalisation, output resizing back to the original resolution
and the choice of runtime. Every backend returns the same payload, so the UI
and the evaluation scripts never branch on runtime.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .preprocess import (INPUT_SIZE, load_gray, load_rgb, normalise_prob,
                         prob_to_mask, to_network_input)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class SegmentationResult:
    prob: np.ndarray          # float32 [H, W] in [0, 1], original resolution
    mask: np.ndarray          # uint8 {0, 1}, original resolution
    latency_ms: float
    backend: str
    size: tuple[int, int]     # (w, h) of the input image


class _TorchBackend:
    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cuda", fp16: bool = True,
                 sam2_checkpoint: Optional[str] = None):
        import torch
        from glide_net.checkpoint import load_model_state, read_checkpoint
        from glide_net.models.glide_net import GLIDE_Net

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu"
                                   else "cpu")
        self.model = GLIDE_Net(checkpoint_path=sam2_checkpoint, device=self.device)
        self.model = self.model.to(self.device)
        _, state = read_checkpoint(checkpoint)
        load_model_state(self.model, state, allow_frozen_missing=sam2_checkpoint is not None)
        del state
        self.model.eval()
        self.half = bool(fp16 and self.device.type == "cuda")
        if self.half:
            self.model.half()
        self.dtype = torch.float16 if self.half else torch.float32
        self.name = f"torch-{'fp16' if self.half else 'fp32'}-{self.device.type}"

    def synchronise(self) -> None:
        """Block until queued CUDA work finishes (needed for honest timing)."""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def run(self, x: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            inp = t.from_numpy(x).to(self.device, dtype=self.dtype)
            logits = self.model(inp)[0]
            prob = logits.float().sigmoid()[0, 0].cpu().numpy()
        return prob


class _OnnxBackend:
    name = "onnx"

    def __init__(self, onnx_path: str, providers: Optional[list[str]] = None):
        import onnxruntime as ort

        # The CUDA provider links against the cuDNN/cuBLAS DLLs that ship with
        # torch; registering torch's lib directory keeps the install portable.
        try:
            import torch

            dll_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
            if hasattr(os, "add_dll_directory") and os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)
        except Exception:  # pragma: no cover - torch is optional for ONNX only use
            pass

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in available] or available
        self.session = ort.InferenceSession(onnx_path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.dtype = np.float16 if "float16" in str(self.session.get_inputs()[0].type) else np.float32
        active = self.session.get_providers()[0]
        self.name = f"onnx-{'fp16' if self.dtype == np.float16 else 'fp32'}-{active.replace('ExecutionProvider', '').lower()}"

    def run(self, x: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: x.astype(self.dtype)})[0]
        prob = np.asarray(out, dtype=np.float32)
        while prob.ndim > 2:
            prob = prob[0]
        return prob


class SegmentationEngine:
    """Deterministic GLIDE-Net inference wrapper with a warm-up pass."""

    def __init__(self, checkpoint: Optional[str] = None, onnx_path: Optional[str] = None,
                 device: str = "cuda", fp16: bool = True, size: int = INPUT_SIZE,
                 sam2_checkpoint: Optional[str] = None, warmup: int = 2):
        if onnx_path:
            self.backend = _OnnxBackend(onnx_path)
        elif checkpoint:
            self.backend = _TorchBackend(checkpoint, device=device, fp16=fp16,
                                         sam2_checkpoint=sam2_checkpoint)
        else:
            raise ValueError("Pass either checkpoint=... or onnx_path=...")
        self.size = size
        dummy = np.zeros((1, 4, size, size), dtype=np.float32)
        for _ in range(warmup):
            self.backend.run(dummy)

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def predict(self, rgb, depth=None, threshold: float = 0.5) -> SegmentationResult:
        image = load_rgb(rgb)
        depth_img = load_gray(depth) if depth is not None else None
        x, size = to_network_input(image, depth_img, self.size)

        started = time.perf_counter()
        prob = self.backend.run(x)
        latency_ms = (time.perf_counter() - started) * 1000.0

        prob = normalise_prob(prob)
        prob_rs, binary = prob_to_mask(prob, size, threshold)
        return SegmentationResult(prob=prob_rs, mask=binary, latency_ms=latency_ms,
                                  backend=self.name, size=size)
