"""Shared preprocessing / post-processing for the inspection service.

Kept dependency-light (numpy + Pillow) so the API, the Gradio demo and the
CLI batch tool all agree on the exact same pixel pipeline.
"""
from __future__ import annotations

import io
from typing import BinaryIO, Optional, Union

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 352

ImageLike = Union[str, bytes, BinaryIO, Image.Image, np.ndarray]


def load_rgb(src: ImageLike) -> Image.Image:
    """Load an arbitrary source into an RGB PIL image (EXIF-rotated)."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        arr = src if src.dtype == np.uint8 else np.clip(src, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr)
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_gray(src: Optional[ImageLike]) -> Optional[Image.Image]:
    """Load the depth branch input as 8-bit single channel."""
    if src is None:
        return None
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        arr = np.asarray(src)
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo) * 255.0
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("L")


def to_network_input(rgb: Image.Image, depth: Optional[Image.Image],
                     size: int = INPUT_SIZE):
    """Return (NCHW float32 array [1,4,H,W], original size (w, h))."""
    orig_w, orig_h = rgb.size
    rgb_r = np.asarray(rgb.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    rgb_n = (rgb_r - IMAGENET_MEAN) / IMAGENET_STD
    if depth is None:
        depth_n = np.zeros((size, size, 1), dtype=np.float32)
    else:
        depth_r = np.asarray(depth.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        depth_n = depth_r[..., None]
    x = np.concatenate([rgb_n, depth_n], axis=2).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), (orig_w, orig_h)


def normalise_prob(prob: np.ndarray) -> np.ndarray:
    """Per-image min-max stretch, matching ``glide_net.checkpoint.prediction_mask``."""
    lo, hi = float(prob.min()), float(prob.max())
    return (prob - lo) / (hi - lo + 1e-8)


def prob_to_mask(prob: np.ndarray, size, threshold: float = 0.5):
    """Resize probability map to ``size`` and binarise it."""
    img = Image.fromarray((prob * 255.0).clip(0, 255).astype(np.uint8))
    if (img.size[0], img.size[1]) != tuple(size):
        img = img.resize(tuple(size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    binary = (arr >= threshold).astype(np.uint8)
    return arr, binary
