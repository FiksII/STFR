from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from production.face_crop_stage import (
    FaceCropConfig,
    aggregate_visible_faces,
    crop_face_mesh,
    expand_face_selection,
    fill_small_face_gaps,
    padded_face_oval_mask,
)


def test_production_crop_defaults_do_not_expand_beyond_face_oval() -> None:
    config = FaceCropConfig()

    assert config.oval_scale == 1.0
    assert config.adjacency_rings == 0
    assert config.maximum_hole_faces == 1000


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


def test_fill_small_face_gaps_keeps_large_outside_component() -> None:
    selected = np.array([True, False, True, False, False, False])
    adjacency = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        dtype=np.int64,
    )

    filled, report = fill_small_face_gaps(
        selected,
        adjacency,
        maximum_hole_faces=100,
    )

    assert filled.tolist() == [True, True, True, False, False, False]
    assert report == {"components": 2, "filled_components": 1, "filled_faces": 1}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FaceCropConfig(oval_scale=0.9), "oval scale"),
        (FaceCropConfig(adjacency_rings=-1), "adjacency rings"),
        (FaceCropConfig(maximum_hole_faces=-1), "hole faces"),
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


def make_crop_inputs(tmp_path):
    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "source.ply"
    mesh.export(mesh_path)
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    frames = []
    for index in range(3):
        name = f"{index:05d}.png"
        Image.new("RGB", (8, 8), "white").save(frames_root / name)
        frames.append(
            {
                "file_path": f"/capture/{name}",
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text(
        json.dumps(
            {
                "w": 8,
                "h": 8,
                "fl_x": 8.0,
                "fl_y": 8.0,
                "cx": 4.0,
                "cy": 4.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return mesh, mesh_path, frames_root, transforms_path


def test_crop_requires_minimum_detected_frames(tmp_path) -> None:
    _, mesh_path, frames_root, transforms_path = make_crop_inputs(tmp_path)

    with pytest.raises(ValueError, match="at least 3"):
        crop_face_mesh(
            source_path=mesh_path,
            selected_frames_root=frames_root,
            transforms_path=transforms_path,
            output_path=tmp_path / "face.ply",
            config=FaceCropConfig(minimum_detected_frames=3),
            device="cuda:0",
            detector=lambda _: None,
            rasterizer=lambda frame, image_size: np.zeros(image_size, dtype=np.int64),
        )


def test_crop_exports_union_of_visible_faces(tmp_path) -> None:
    mesh, mesh_path, frames_root, transforms_path = make_crop_inputs(tmp_path)
    visible_by_name = {
        "00000.png": 0,
        "00001.png": 1,
        "00002.png": 2,
    }

    def detect(_):
        return np.array(
            [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]],
            dtype=np.float64,
        )

    def rasterize(frame, image_size):
        raster = np.full(image_size, -1, dtype=np.int64)
        raster[2:6, 2:6] = visible_by_name[Path(frame["file_path"]).name]
        return raster

    output = tmp_path / "face.ply"
    report = crop_face_mesh(
        source_path=mesh_path,
        selected_frames_root=frames_root,
        transforms_path=transforms_path,
        output_path=output,
        config=FaceCropConfig(
            adjacency_rings=0,
            maximum_hole_faces=0,
            minimum_detected_frames=3,
            minimum_selected_faces=3,
        ),
        device="cuda:0",
        detector=detect,
        rasterizer=rasterize,
    )

    result = trimesh.load_mesh(output, process=False)
    assert report["source_faces"] == len(mesh.faces)
    assert report["detected_frames"] == 3
    assert report["missed_frames"] == []
    assert report["selected_faces_before_expansion"] == 3
    assert report["selected_faces_after_expansion"] == 3
    assert report["filled_hole_faces"] == 0
    assert report["output_faces"] == len(result.faces) == 3


def test_crop_rejects_too_few_selected_faces(tmp_path) -> None:
    _, mesh_path, frames_root, transforms_path = make_crop_inputs(tmp_path)

    with pytest.raises(ValueError, match="selected 1 faces"):
        crop_face_mesh(
            source_path=mesh_path,
            selected_frames_root=frames_root,
            transforms_path=transforms_path,
            output_path=tmp_path / "face.ply",
            config=FaceCropConfig(
                adjacency_rings=0,
                minimum_detected_frames=3,
                minimum_selected_faces=2,
            ),
            device="cuda:0",
            detector=lambda _: np.array([[1, 1], [6, 1], [4, 6]]),
            rasterizer=lambda frame, image_size: np.zeros(image_size, dtype=np.int64),
        )
