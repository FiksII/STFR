from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import trimesh

from production.unwrap_2dgs_uv import load_triangle_mesh


FACE_OVAL_INDICES = np.array(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    dtype=np.int64,
)


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


class MediaPipeFaceOvalDetector:
    def __init__(self) -> None:
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def __call__(self, image_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read selected frame: {image_path}")
        height, width = image.shape[:2]
        result = self._face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        return np.array(
            [
                [landmarks[index].x * width, landmarks[index].y * height]
                for index in FACE_OVAL_INDICES
            ],
            dtype=np.float64,
        )

    def close(self) -> None:
        self._face_mesh.close()


class Pytorch3DFaceRasterizer:
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
            (1, len(mesh.vertices), 1), dtype=torch.float32, device=device
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
        camera_extrinsic = torch.inverse(camera_to_world)[:, :3]
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
                camera_extrinsic,
            )
        return pixel_to_face[0, :, :, 0].detach().cpu().numpy()


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


Detector = Callable[[Path], np.ndarray | None]
Rasterizer = Callable[[dict, tuple[int, int]], np.ndarray]


def crop_face_mesh(
    source_path: Path,
    selected_frames_root: Path,
    transforms_path: Path,
    output_path: Path,
    config: FaceCropConfig = FaceCropConfig(),
    device: str = "cuda:0",
    detector: Detector | None = None,
    rasterizer: Rasterizer | None = None,
) -> dict:
    config.validate()
    source_path = Path(source_path)
    selected_frames_root = Path(selected_frames_root)
    transforms_path = Path(transforms_path)
    output_path = Path(output_path)
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
    missing_cameras = [image.name for image in selected_images if image.name not in frames_by_name]
    if missing_cameras:
        raise ValueError(
            "Selected frames are missing camera transforms: "
            + ", ".join(missing_cameras)
        )

    owned_detector = detector is None
    active_detector = MediaPipeFaceOvalDetector() if detector is None else detector
    active_rasterizer = (
        Pytorch3DFaceRasterizer(source, metadata, device)
        if rasterizer is None
        else rasterizer
    )
    face_rasters: list[np.ndarray] = []
    oval_masks: list[np.ndarray] = []
    detected_names: list[str] = []
    missed_names: list[str] = []
    try:
        for image_path in selected_images:
            oval = active_detector(image_path)
            if oval is None:
                missed_names.append(image_path.name)
                continue
            with Image.open(image_path) as image:
                image_size = (image.height, image.width)
            mask = padded_face_oval_mask(oval, image_size, config.oval_scale)
            raster = active_rasterizer(frames_by_name[image_path.name], image_size)
            face_rasters.append(raster)
            oval_masks.append(mask)
            detected_names.append(image_path.name)
    finally:
        if owned_detector:
            active_detector.close()

    if len(detected_names) < config.minimum_detected_frames:
        raise ValueError(
            f"Face crop requires at least {config.minimum_detected_frames} detected "
            f"frames, got {len(detected_names)}"
        )

    selected = aggregate_visible_faces(
        face_rasters,
        oval_masks,
        face_count=len(source.faces),
    )
    selected_before_expansion = int(selected.sum())
    selected = expand_face_selection(
        selected,
        np.asarray(source.face_adjacency),
        config.adjacency_rings,
    )
    selected_after_expansion = int(selected.sum())
    if selected_after_expansion < config.minimum_selected_faces:
        raise ValueError(
            f"Face crop selected {selected_after_expansion} faces, fewer than the "
            f"required {config.minimum_selected_faces}"
        )

    cropped = source.submesh([np.flatnonzero(selected)], append=True, repair=False)
    if not isinstance(cropped, trimesh.Trimesh) or not len(cropped.faces):
        raise ValueError("Face crop did not produce a triangle mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.export(output_path)
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "device": device,
        "config": {
            "oval_scale": config.oval_scale,
            "adjacency_rings": config.adjacency_rings,
            "minimum_detected_frames": config.minimum_detected_frames,
            "minimum_selected_faces": config.minimum_selected_faces,
        },
        "selected_frames": len(selected_images),
        "detected_frames": len(detected_names),
        "detected_frame_names": detected_names,
        "missed_frames": missed_names,
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "selected_faces_before_expansion": selected_before_expansion,
        "selected_faces_after_expansion": selected_after_expansion,
        "output_vertices": int(len(cropped.vertices)),
        "output_faces": int(len(cropped.faces)),
        "bounds": np.asarray(cropped.bounds, dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop visible facial triangles from an STFR 2DGS mesh."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oval-scale", type=float, default=1.15)
    parser.add_argument("--adjacency-rings", type=int, default=2)
    parser.add_argument("--minimum-detected-frames", type=int, default=3)
    parser.add_argument("--minimum-selected-faces", type=int, default=10_000)
    args = parser.parse_args()
    report = crop_face_mesh(
        source_path=args.source,
        selected_frames_root=args.frames,
        transforms_path=args.transforms,
        output_path=args.output,
        config=FaceCropConfig(
            oval_scale=args.oval_scale,
            adjacency_rings=args.adjacency_rings,
            minimum_detected_frames=args.minimum_detected_frames,
            minimum_selected_faces=args.minimum_selected_faces,
        ),
        device=args.device,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
