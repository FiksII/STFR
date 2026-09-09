from __future__ import annotations

import torch
import torch.nn.functional as F


def bound_lpips_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_size < 1:
        raise ValueError("LPIPS maximum size must be positive")

    height, width = prediction.shape[-2:]
    long_edge = max(height, width)
    if long_edge <= max_size:
        return prediction, target

    scale = max_size / long_edge
    output_size = (
        max(1, round(height * scale)),
        max(1, round(width * scale)),
    )
    options = {
        "size": output_size,
        "mode": "bilinear",
        "align_corners": False,
        "antialias": True,
    }
    return (
        F.interpolate(prediction, **options),
        F.interpolate(target, **options),
    )
