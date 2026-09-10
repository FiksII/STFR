from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from production.head_crop_stage import (
    HeadCropConfig,
    aggregate_visible_faces,
    crop_head_mesh,
    fill_small_face_gaps,
    open_face_selection,
)
from production.head_segmentation import HeadMaskConfig


def test_head_crop_defaults_preserve_head_detail() -> None:
    config = HeadCropConfig()

    assert config.maximum_hole_faces == 1000
    assert config.opening_rings == 3
    assert config.minimum_detected_frames == 3
    assert config.minimum_selected_faces == 10_000
    assert config.mask.neck_height_ratio == 0.45


def test_aggregate_visible_faces_combines_masked_ids_from_views() -> None:
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


def test_fill_small_face_gaps_keeps_large_outside_component() -> None:
    selected = np.array([True, False, True, False, False, False])
    adjacency = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        dtype=np.int64,
    )

    filled, report = fill_small_face_gaps(selected, adjacency, maximum_hole_faces=100)

    assert filled.tolist() == [True, True, True, False, False, False]
    assert report == {"components": 2, "filled_components": 1, "filled_faces": 1}


def test_open_face_selection_removes_thin_tip_without_expanding() -> None:
    selected = np.array([True, True, True, True, True, False, False])
    adjacency = np.array(
        [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [3, 5], [4, 6]],
        dtype=np.int64,
    )

    opened = open_face_selection(selected, adjacency, rings=1)

    assert opened.tolist() == [True, True, True, True, False, False, False]


def make_crop_inputs(tmp_path: Path):
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


def test_crop_exports_head_union_and_separate_face_anchor(tmp_path: Path) -> None:
    mesh, mesh_path, frames_root, transforms_path = make_crop_inputs(tmp_path)
    head_ids = {"00000.png": 0, "00001.png": 1, "00002.png": 2}
    anchor_ids = {"00000.png": 3, "00001.png": 4, "00002.png": 5}

    def detect(_):
        return np.array(
            [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]],
            dtype=np.float64,
        )

    def parse(_):
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[0:2, 2:7] = 17
        labels[2:7, 2:7] = 1
        return labels

    def rasterize(frame, image_size):
        name = Path(frame["file_path"]).name
        raster = np.full(image_size, -1, dtype=np.int64)
        raster[0:2, 2:7] = head_ids[name]
        raster[2:7, 2:7] = anchor_ids[name]
        return raster

    output = tmp_path / "head.ply"
    anchor = tmp_path / "anchor.ply"
    diagnostics = tmp_path / "masks"
    report = crop_head_mesh(
        source_path=mesh_path,
        selected_frames_root=frames_root,
        transforms_path=transforms_path,
        output_path=output,
        face_anchor_path=anchor,
        diagnostics_root=diagnostics,
        model_path=tmp_path / "model.onnx",
        config=HeadCropConfig(
            mask=HeadMaskConfig(
                close_kernel_ratio=0.0,
                minimum_component_pixels=1,
            ),
            maximum_hole_faces=0,
            opening_rings=0,
            minimum_selected_faces=6,
        ),
        detector=detect,
        parser=parse,
        rasterizer=rasterize,
    )

    head = trimesh.load_mesh(output, process=False)
    face_anchor = trimesh.load_mesh(anchor, process=False)
    assert report["source_faces"] == len(mesh.faces)
    assert report["output_faces"] == len(head.faces) == 6
    assert report["face_anchor_faces"] == len(face_anchor.faces) == 3
    assert report["detected_frames"] == 3
    assert len(list(diagnostics.glob("*.png"))) == 3


def test_crop_requires_minimum_detected_frames(tmp_path: Path) -> None:
    _, mesh_path, frames_root, transforms_path = make_crop_inputs(tmp_path)

    with pytest.raises(ValueError, match="at least 3"):
        crop_head_mesh(
            source_path=mesh_path,
            selected_frames_root=frames_root,
            transforms_path=transforms_path,
            output_path=tmp_path / "head.ply",
            face_anchor_path=tmp_path / "anchor.ply",
            diagnostics_root=tmp_path / "masks",
            model_path=tmp_path / "model.onnx",
            config=HeadCropConfig(
                mask=HeadMaskConfig(minimum_component_pixels=1),
                minimum_detected_frames=3,
                minimum_selected_faces=1,
            ),
            detector=lambda _: None,
            parser=lambda _: np.zeros((8, 8), dtype=np.uint8),
            rasterizer=lambda frame, size: np.zeros(size, dtype=np.int64),
        )
