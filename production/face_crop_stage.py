from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceCropConfig:
    oval_scale: float = 1.15
    adjacency_rings: int = 2
    minimum_detected_frames: int = 3
    minimum_selected_faces: int = 10_000

    def validate(self) -> None:
        if self.oval_scale < 1.0:
            raise ValueError("Face oval scale must be at least 1")
        if self.adjacency_rings < 0:
            raise ValueError("Face adjacency rings cannot be negative")
        if self.minimum_detected_frames < 1:
            raise ValueError("Minimum detected frames must be positive")
        if self.minimum_selected_faces < 1:
            raise ValueError("Minimum selected faces must be positive")


def padded_face_oval_mask(
    landmarks: np.ndarray,
    image_size: tuple[int, int],
    scale: float,
) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    height, width = image_size
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Face oval landmarks must have shape [N, 2] with N >= 3")
    if not np.isfinite(points).all():
        raise ValueError("Face oval landmarks contain non-finite coordinates")
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    if scale < 1.0:
        raise ValueError("Face oval scale must be at least 1")

    center = points.mean(axis=0, keepdims=True)
    polygon = center + (points - center) * scale
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    polygon = np.floor(polygon).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask.astype(bool)


def aggregate_visible_faces(
    face_rasters: Iterable[np.ndarray],
    oval_masks: Iterable[np.ndarray],
    face_count: int,
) -> np.ndarray:
    if face_count < 1:
        raise ValueError("Face count must be positive")
    rasters = list(face_rasters)
    masks = list(oval_masks)
    if len(rasters) != len(masks):
        raise ValueError("Face rasters and oval masks must have matching counts")

    selected = np.zeros(face_count, dtype=bool)
    for raster, mask in zip(rasters, masks, strict=True):
        raster_array = np.asarray(raster)
        if raster_array.ndim == 3 and raster_array.shape[-1] == 1:
            raster_array = raster_array[..., 0]
        mask_array = np.asarray(mask, dtype=bool)
        if raster_array.shape != mask_array.shape:
            raise ValueError("Face raster and oval mask must have matching shapes")
        ids = raster_array[mask_array].astype(np.int64, copy=False)
        ids = ids[(ids >= 0) & (ids < face_count)]
        selected[np.unique(ids)] = True
    return selected


def expand_face_selection(
    selected: np.ndarray,
    adjacency: np.ndarray,
    rings: int,
) -> np.ndarray:
    result = np.asarray(selected, dtype=bool).copy()
    pairs = np.asarray(adjacency, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError("Selected face mask must be one-dimensional")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Face adjacency must have shape [N, 2]")
    if rings < 0:
        raise ValueError("Face adjacency rings cannot be negative")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(result)):
        raise ValueError("Face adjacency contains invalid indices")

    for _ in range(rings):
        touching = result[pairs[:, 0]] | result[pairs[:, 1]]
        result[pairs[touching].reshape(-1)] = True
    return result
