"""Checkpoint validation and prediction mask formatting."""

import torch

def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or contain model_state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint contains no valid model state dict.")
    return checkpoint, state


def load_model_state(model, state, allow_frozen_missing=False):
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed = {name for name, p in model.named_parameters()
               if not p.requires_grad} if allow_frozen_missing else set()
    wrong_shapes = [k for k in set(state) & set(expected)
                    if state[k].shape != expected[k].shape]
    wrong_dtypes = [k for k in set(state) & set(expected)
                    if state[k].dtype != expected[k].dtype]
    if missing - allowed or unexpected or wrong_shapes or wrong_dtypes:
        raise ValueError(
            f"Invalid checkpoint: missing={sorted(missing - allowed)}, "
            f"unexpected={sorted(unexpected)}, wrong_shapes={wrong_shapes}, "
            f"wrong_dtypes={wrong_dtypes}")
    model.load_state_dict(state, strict=not allow_frozen_missing)


def prediction_mask(logits, size):
    """Resize logits, apply sigmoid and min-max normalization, then uint8."""
    import numpy as np
    import torch.nn.functional as F

    pred = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    pred = pred.sigmoid().detach().cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return (pred * 255).astype(np.uint8)
