"""
Datasets for RGB-D rail surface defect detection.

Expected directory layout (``split`` is ``train`` / ``val`` / ``test``)::

    <root>/
        Image_<split>/   *.bmp    RGB images
        Depth_<split>/   *.tiff   depth maps
        GT_<split>/      *.png    binary masks
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_image(path, mode):
    with open(path, "rb") as f:
        return Image.open(f).convert(mode)


def _pair_by_stem(images, depths, masks):
    """Pair RGB / depth / mask files that share the same file stem."""
    by_stem = lambda files: {os.path.splitext(os.path.basename(p))[0]: p for p in files}
    img_map, depth_map, mask_map = by_stem(images), by_stem(depths), by_stem(masks)
    if set(img_map) != set(depth_map) or set(img_map) != set(mask_map):
        raise ValueError("RGB, depth and mask file stems must match exactly.")
    stems = sorted(set(img_map) & set(depth_map) & set(mask_map))
    if not stems:
        raise RuntimeError("No matched RGB/depth/mask triplets were found.")
    return [(img_map[s], depth_map[s], mask_map[s]) for s in stems]


class RailDefectDataset(Dataset):
    """RGB-D dataset used for training and validation."""

    def __init__(self, root, split="train", image_size=352, augment=None,
                 img_ext=".bmp", depth_ext=".tiff", mask_ext=".png"):
        if augment is None:
            augment = split == "train"
        self.augment = augment
        self.image_size = image_size

        def list_files(subdir, ext):
            d = os.path.join(root, subdir)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing dataset directory: {d}")
            return [os.path.join(d, n) for n in os.listdir(d) if n.endswith(ext)]

        self.samples = _pair_by_stem(
            list_files(f"Image_{split}", img_ext),
            list_files(f"Depth_{split}", depth_ext),
            list_files(f"GT_{split}", mask_ext),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, mask_path = self.samples[idx]

        image = _load_image(img_path, "RGB")
        depth = _load_image(depth_path, "L")
        mask = _load_image(mask_path, "L")

        size = (self.image_size, self.image_size)
        image = F.resize(image, size)
        depth = F.resize(depth, size, interpolation=InterpolationMode.BICUBIC)
        mask = F.resize(mask, size, interpolation=InterpolationMode.BICUBIC)

        if self.augment:
            if random.random() < 0.5:
                image, depth, mask = F.hflip(image), F.hflip(depth), F.hflip(mask)
            if random.random() < 0.5:
                image, depth, mask = F.vflip(image), F.vflip(depth), F.vflip(mask)

        image = F.normalize(F.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        depth = F.to_tensor(depth)
        mask = F.to_tensor(mask)

        return {"image": image, "depth": depth, "label": mask}


class TestDataset(Dataset):
    """Dataset used for inference; returns the ground truth at its original size."""

    def __init__(self, root, split="test", image_size=352,
                 img_ext=".bmp", depth_ext=".tiff", mask_ext=".png"):
        self.image_size = image_size

        def list_files(subdir, ext):
            d = os.path.join(root, subdir)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing dataset directory: {d}")
            return [os.path.join(d, n) for n in os.listdir(d) if n.endswith(ext)]

        self.samples = _pair_by_stem(
            list_files(f"Image_{split}", img_ext),
            list_files(f"Depth_{split}", depth_ext),
            list_files(f"GT_{split}", mask_ext),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, mask_path = self.samples[idx]

        image = _load_image(img_path, "RGB")
        depth = _load_image(depth_path, "L")
        mask = np.asarray(_load_image(mask_path, "L"), dtype=np.uint8)

        size = (self.image_size, self.image_size)
        image = F.normalize(F.to_tensor(F.resize(image, size)), IMAGENET_MEAN, IMAGENET_STD)
        depth = F.to_tensor(F.resize(depth, size, interpolation=InterpolationMode.BILINEAR))
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)

        name = os.path.splitext(os.path.basename(img_path))[0]
        return image, depth, mask, name
