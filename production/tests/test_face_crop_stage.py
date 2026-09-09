from __future__ import annotations

import numpy as np
import pytest

from production.face_crop_stage import (
    FaceCropConfig,
    aggregate_visible_faces,
    expand_face_selection,
    padded_face_oval_mask,
)


def test_padded_face_oval_mask_expands_about_landmark_center() -> None:
    landmarks = np.array(
        [[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]],
        dtype=np.float64,
    )

    mask = padded_face_oval_mask(landmarks, (12, 12), scale=1.5)

    assert mask.dtype == np.bool_
    assert mask.shape == (12, 12)
    assert mask[5, 5]
    assert mask[3, 5]
    assert not mask[0, 0]


def test_padded_face_oval_mask_clips_to_image_bounds() -> None:
    landmarks = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=np.float64,
    )

    mask = padded_face_oval_mask(landmarks, (4, 4), scale=1.5)

    assert mask[0, 0]
    assert not mask[3, 3]


def test_aggregate_visible_faces_ignores_background_and_outside_pixels() -> None:
    raster = np.array([[-1, 0], [1, 2]], dtype=np.int64)
    oval = np.array([[False, True], [True, False]])

    selected = aggregate_visible_faces([raster], [oval], face_count=3)

    assert selected.tolist() == [True, True, False]


def test_aggregate_visible_faces_combines_multiple_views() -> None:
    rasters = [
        np.array([[0, 1], [-1, -1]], dtype=np.int64),
        np.array([[2, 3], [-1, -1]], dtype=np.int64),
    ]
    masks = [
        np.array([[True, False], [False, False]]),
        np.array([[False, True], [False, False]]),
    ]

    selected = aggregate_visible_faces(rasters, masks, face_count=4)

    assert selected.tolist() == [True, False, False, True]


def test_aggregate_visible_faces_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        aggregate_visible_faces(
            [np.zeros((2, 2), dtype=np.int64)],
            [np.zeros((3, 3), dtype=bool)],
            face_count=1,
        )


def test_expand_face_selection_adds_exact_adjacency_rings() -> None:
    selected = np.array([True, False, False, False])
    adjacency = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)

    one_ring = expand_face_selection(selected, adjacency, rings=1)
    two_rings = expand_face_selection(selected, adjacency, rings=2)

    assert one_ring.tolist() == [True, True, False, False]
    assert two_rings.tolist() == [True, True, True, False]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FaceCropConfig(oval_scale=0.9), "oval scale"),
        (FaceCropConfig(adjacency_rings=-1), "adjacency rings"),
        (FaceCropConfig(minimum_detected_frames=0), "detected frames"),
        (FaceCropConfig(minimum_selected_faces=0), "selected faces"),
    ],
)
def test_face_crop_config_rejects_invalid_values(
    config: FaceCropConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()
