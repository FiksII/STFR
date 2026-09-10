from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

from production.head_segmentation import (
    HeadMaskConfig,
    MediaPipeFaceAnchorDetector,
    OnnxFaceParser,
    build_head_mask,
    face_oval_mask,
)
from production.unwrap_2dgs_uv import load_triangle_mesh


@dataclass(frozen=True)
class HeadCropConfig:
    mask: HeadMaskConfig = HeadMaskConfig()
    maximum_hole_faces: int = 1000
    opening_rings: int = 3
    minimum_detected_frames: int = 3
    minimum_selected_faces: int = 10_000

    def validate(self) -> None:
        self.mask.validate()
        if self.maximum_hole_faces < 0:
            raise ValueError("Maximum hole faces cannot be negative")
        if self.opening_rings < 0:
            raise ValueError("Head opening rings cannot be negative")
        if self.minimum_detected_frames < 1:
            raise ValueError("Minimum detected frames must be positive")
        if self.minimum_selected_faces < 1:
            raise ValueError("Minimum selected faces must be positive")


class Pytorch3DHeadRasterizer:
    def __init__(self, mesh: trimesh.Trimesh, metadata: dict, device: str) -> None:
        import torch

        from texture.mesh_renderer import MeshRenderer

        self._torch = torch
        self._renderer = MeshRenderer(device)
        self._vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(device).float()[
            None
        ]
        self._faces = torch.from_numpy(np.asarray(mesh.faces)).to(device).long()[None]
        self._attributes = torch.ones(
            (1, len(mesh.vertices), 1),
            dtype=torch.float32,
            device=device,
        )
        self._metadata = metadata

    def __call__(
        self,
        frame: dict,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        torch = self._torch
        height, width = image_size
        intrinsic = torch.eye(3, dtype=torch.float32, device=self._vertices.device)
        intrinsic[0, 0] = float(self._metadata["fl_x"]) / width
        intrinsic[1, 1] = float(self._metadata["fl_y"]) / height
        intrinsic[0, 2] = float(self._metadata["cx"]) / width
        intrinsic[1, 2] = float(self._metadata["cy"]) / height
        camera_to_world = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32,
            device=self._vertices.device,
        )[None]
        mesh_dict = {
            "faces": self._faces,
            "vertice": self._vertices,
            "attributes": self._attributes,
            "size": image_size,
        }
        with torch.inference_mode():
            _, pixel_to_face = self._renderer.render_mesh(
                mesh_dict,
                intrinsic[None],
                torch.inverse(camera_to_world)[:, :3],
            )
        return pixel_to_face[0, :, :, 0].detach().cpu().numpy()


def aggregate_visible_faces(
    face_rasters: Iterable[np.ndarray],
    masks: Iterable[np.ndarray],
    face_count: int,
) -> np.ndarray:
    if face_count < 1:
        raise ValueError("Face count must be positive")
    rasters = list(face_rasters)
    mask_list = list(masks)
    if len(rasters) != len(mask_list):
        raise ValueError("Face rasters and masks must have matching counts")

    selected = np.zeros(face_count, dtype=bool)
    for raster, mask in zip(rasters, mask_list, strict=True):
        raster_array = np.asarray(raster)
        if raster_array.ndim == 3 and raster_array.shape[-1] == 1:
            raster_array = raster_array[..., 0]
        mask_array = np.asarray(mask, dtype=bool)
        if raster_array.shape != mask_array.shape:
            raise ValueError("Face raster and mask must have matching shapes")
        ids = raster_array[mask_array].astype(np.int64, copy=False)
        ids = ids[(ids >= 0) & (ids < face_count)]
        selected[np.unique(ids)] = True
    return selected


def fill_small_face_gaps(
    selected: np.ndarray,
    adjacency: np.ndarray,
    maximum_hole_faces: int,
) -> tuple[np.ndarray, dict[str, int]]:
    result = np.asarray(selected, dtype=bool).copy()
    pairs = np.asarray(adjacency, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError("Selected face mask must be one-dimensional")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Face adjacency must have shape [N, 2]")
    if maximum_hole_faces < 0:
        raise ValueError("Maximum hole faces cannot be negative")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(result)):
        raise ValueError("Face adjacency contains invalid indices")

    unselected_ids = np.flatnonzero(~result)
    empty_report = {
        "components": int(len(unselected_ids)),
        "filled_components": 0,
        "filled_faces": 0,
    }
    if maximum_hole_faces == 0 or not len(unselected_ids) or not len(pairs):
        return result, empty_report

    local_ids = np.full(len(result), -1, dtype=np.int64)
    local_ids[unselected_ids] = np.arange(len(unselected_ids))
    unselected_pairs = pairs[~result[pairs].any(axis=1)]
    local_pairs = local_ids[unselected_pairs]
    rows = np.concatenate((local_pairs[:, 0], local_pairs[:, 1]))
    columns = np.concatenate((local_pairs[:, 1], local_pairs[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(unselected_ids), len(unselected_ids)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)

    border_pairs = pairs[result[pairs[:, 0]] != result[pairs[:, 1]]]
    border_unselected = np.where(
        result[border_pairs[:, 0]],
        border_pairs[:, 1],
        border_pairs[:, 0],
    )
    touches_selection = np.zeros(component_count, dtype=bool)
    touches_selection[labels[local_ids[border_unselected]]] = True
    fill_components = touches_selection & (component_sizes <= maximum_hole_faces)
    fill_components[int(np.argmax(component_sizes))] = False
    fill_local = fill_components[labels]
    result[unselected_ids[fill_local]] = True
    return result, {
        "components": int(component_count),
        "filled_components": int(fill_components.sum()),
        "filled_faces": int(fill_local.sum()),
    }


def open_face_selection(
    selected: np.ndarray,
    adjacency: np.ndarray,
    rings: int,
) -> np.ndarray:
    original = np.asarray(selected, dtype=bool).copy()
    pairs = np.asarray(adjacency, dtype=np.int64)
    if original.ndim != 1:
        raise ValueError("Selected face mask must be one-dimensional")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Face adjacency must have shape [N, 2]")
    if rings < 0:
        raise ValueError("Head opening rings cannot be negative")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(original)):
        raise ValueError("Face adjacency contains invalid indices")

    eroded = original.copy()
    for _ in range(rings):
        border = eroded[pairs[:, 0]] != eroded[pairs[:, 1]]
        border_pairs = pairs[border]
        eroded[border_pairs[eroded[border_pairs]]] = False

    opened = eroded
    for _ in range(rings):
        touching = opened[pairs[:, 0]] | opened[pairs[:, 1]]
        opened[pairs[touching].reshape(-1)] = True
        opened &= original
    return opened


Detector = Callable[[Path], np.ndarray | None]
Parser = Callable[[Path], np.ndarray]
Rasterizer = Callable[[dict, tuple[int, int]], np.ndarray]


def crop_head_mesh(
    source_path: Path,
    selected_frames_root: Path,
    transforms_path: Path,
    output_path: Path,
    face_anchor_path: Path,
    diagnostics_root: Path,
    model_path: Path,
    config: HeadCropConfig = HeadCropConfig(),
    device: str = "cuda:0",
    detector: Detector | None = None,
    parser: Parser | None = None,
    rasterizer: Rasterizer | None = None,
) -> dict:
    config.validate()
    source_path = Path(source_path)
    selected_frames_root = Path(selected_frames_root)
    transforms_path = Path(transforms_path)
    output_path = Path(output_path)
    face_anchor_path = Path(face_anchor_path)
    diagnostics_root = Path(diagnostics_root)
    source = load_triangle_mesh(source_path)
    selected_images = sorted(selected_frames_root.glob("*.png"))
    if not selected_images:
        raise FileNotFoundError(f"No selected PNG frames under {selected_frames_root}")
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)

    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames_by_name = {
        Path(frame["file_path"]).name: frame for frame in metadata.get("frames", [])
    }
    missing_cameras = [
        image.name for image in selected_images if image.name not in frames_by_name
    ]
    if missing_cameras:
        raise ValueError(
            "Selected frames are missing camera transforms: "
            + ", ".join(missing_cameras)
        )

    owned_detector = detector is None
    active_detector = MediaPipeFaceAnchorDetector() if detector is None else detector
    active_parser = OnnxFaceParser(model_path) if parser is None else parser
    active_rasterizer = (
        Pytorch3DHeadRasterizer(source, metadata, device)
        if rasterizer is None
        else rasterizer
    )
    head_rasters: list[np.ndarray] = []
    head_masks: list[np.ndarray] = []
    anchor_masks: list[np.ndarray] = []
    mask_reports: list[dict] = []
    detected_names: list[str] = []
    missed_names: list[str] = []
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    try:
        for image_path in selected_images:
            landmarks = active_detector(image_path)
            if landmarks is None:
                missed_names.append(image_path.name)
                continue
            with Image.open(image_path) as image:
                image_size = (image.height, image.width)
            labels = np.asarray(active_parser(image_path))
            if labels.shape != image_size:
                raise ValueError(
                    f"Parser output for {image_path.name} has shape {labels.shape}, "
                    f"expected {image_size}"
                )
            head_mask, mask_report = build_head_mask(labels, landmarks, config.mask)
            raster = active_rasterizer(frames_by_name[image_path.name], image_size)
            head_rasters.append(raster)
            head_masks.append(head_mask)
            anchor_masks.append(face_oval_mask(landmarks, image_size))
            mask_reports.append({"frame": image_path.name, **mask_report})
            detected_names.append(image_path.name)
            cv2.imwrite(
                str(diagnostics_root / image_path.name),
                head_mask.astype(np.uint8) * 255,
            )
    finally:
        if owned_detector:
            active_detector.close()

    if len(detected_names) < config.minimum_detected_frames:
        raise ValueError(
            f"Head crop requires at least {config.minimum_detected_frames} detected "
            f"frames, got {len(detected_names)}"
        )

    selected = aggregate_visible_faces(
        head_rasters,
        head_masks,
        face_count=len(source.faces),
    )
    selected_before_hole_fill = int(selected.sum())
    adjacency = np.asarray(source.face_adjacency)
    selected, hole_report = fill_small_face_gaps(
        selected,
        adjacency,
        config.maximum_hole_faces,
    )
    selected_after_hole_fill = int(selected.sum())
    selected = open_face_selection(selected, adjacency, config.opening_rings)
    selected_after_opening = int(selected.sum())
    if selected_after_opening < config.minimum_selected_faces:
        raise ValueError(
            f"Head crop retained {selected_after_opening} faces, fewer than the "
            f"required {config.minimum_selected_faces}"
        )

    anchor_selected = aggregate_visible_faces(
        head_rasters,
        anchor_masks,
        face_count=len(source.faces),
    )
    if not anchor_selected.any():
        raise ValueError("Face anchor did not select any mesh faces")

    cropped = source.submesh([np.flatnonzero(selected)], append=True, repair=False)
    anchor = source.submesh(
        [np.flatnonzero(anchor_selected)],
        append=True,
        repair=False,
    )
    if not isinstance(cropped, trimesh.Trimesh) or not len(cropped.faces):
        raise ValueError("Head crop did not produce a triangle mesh")
    if not isinstance(anchor, trimesh.Trimesh) or not len(anchor.faces):
        raise ValueError("Face anchor did not produce a triangle mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_anchor_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.export(output_path)
    anchor.export(face_anchor_path)

    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "face_anchor": str(face_anchor_path.resolve()),
        "diagnostics": str(diagnostics_root.resolve()),
        "model": str(Path(model_path).resolve()),
        "model_sha256": getattr(active_parser, "model_sha256", "injected"),
        "device": device,
        "config": asdict(config),
        "selected_frames": len(selected_images),
        "detected_frames": len(detected_names),
        "detected_frame_names": detected_names,
        "missed_frames": missed_names,
        "mask_reports": mask_reports,
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "selected_faces_before_hole_fill": selected_before_hole_fill,
        "selected_faces_after_hole_fill": selected_after_hole_fill,
        "selected_faces_after_opening": selected_after_opening,
        "unselected_components": hole_report["components"],
        "filled_hole_components": hole_report["filled_components"],
        "filled_hole_faces": hole_report["filled_faces"],
        "face_anchor_faces": int(len(anchor.faces)),
        "output_vertices": int(len(cropped.vertices)),
        "output_faces": int(len(cropped.faces)),
        "bounds": np.asarray(cropped.bounds, dtype=float).tolist(),
    }
