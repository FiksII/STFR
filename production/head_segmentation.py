from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from production.model_assets import verify_model


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

HEAD_LABELS = frozenset({1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
NECK_LABEL = 17
EXCLUDED_LABELS = frozenset({3, 14, 15, 16, 18})


@dataclass(frozen=True)
class HeadMaskConfig:
    neck_height_ratio: float = 0.45
    close_kernel_ratio: float = 0.015
    minimum_component_pixels: int = 64

    def validate(self) -> None:
        if self.neck_height_ratio < 0:
            raise ValueError("Neck height ratio cannot be negative")
        if self.close_kernel_ratio < 0:
            raise ValueError("Close kernel ratio cannot be negative")
        if self.minimum_component_pixels < 1:
            raise ValueError("Minimum component pixels must be positive")


def _validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Face landmarks must have shape [N, 2] with N >= 3")
    if not np.isfinite(points).all():
        raise ValueError("Face landmarks contain non-finite coordinates")
    if np.ptp(points[:, 0]) <= 0 or np.ptp(points[:, 1]) <= 0:
        raise ValueError("Face landmarks must span a measurable area")
    return points


def face_oval_mask(
    landmarks: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    points = _validate_landmarks(landmarks).copy()
    height, width = image_size
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    polygon = np.floor(points).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask.astype(bool)


def _close_mask(mask: np.ndarray, face_height: float, ratio: float) -> np.ndarray:
    size = max(1, int(round(face_height * ratio)))
    if size % 2 == 0:
        size += 1
    if size == 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    ).astype(bool)


def _anchored_component(
    candidate: np.ndarray,
    anchor: np.ndarray,
    minimum_pixels: int,
) -> tuple[np.ndarray, int]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise ValueError("Head mask has no component intersecting the face anchor")
    component_ids = np.arange(1, component_count)
    valid = component_ids[stats[1:, cv2.CC_STAT_AREA] >= minimum_pixels]
    if not len(valid):
        raise ValueError("Head mask has no component intersecting the face anchor")
    overlaps = np.array([np.count_nonzero(anchor & (labels == item)) for item in valid])
    best_offset = int(np.argmax(overlaps))
    if overlaps[best_offset] == 0:
        raise ValueError("Head mask has no component intersecting the face anchor")
    return labels == int(valid[best_offset]), int(len(valid))


def build_head_mask(
    labels: np.ndarray,
    landmarks: np.ndarray,
    config: HeadMaskConfig = HeadMaskConfig(),
) -> tuple[np.ndarray, dict]:
    config.validate()
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("Face parsing labels must be a two-dimensional array")
    points = _validate_landmarks(landmarks)
    height, width = labels.shape
    face_height = float(np.ptp(points[:, 1]))
    face_width = float(np.ptp(points[:, 0]))
    center_x = float(points[:, 0].mean())
    chin_y = float(points[:, 1].max())
    yy, xx = np.indices(labels.shape)

    head = np.isin(labels, tuple(HEAD_LABELS))
    neck = (
        (labels == NECK_LABEL)
        & (yy <= chin_y + config.neck_height_ratio * face_height)
        & (np.abs(xx - center_x) <= 0.55 * face_width)
    )
    candidate = _close_mask(head | neck, face_height, config.close_kernel_ratio)
    candidate[np.isin(labels, tuple(EXCLUDED_LABELS))] = False
    candidate[(labels == NECK_LABEL) & ~neck] = False
    anchor = face_oval_mask(points, (height, width))
    result, candidate_components = _anchored_component(
        candidate,
        anchor,
        config.minimum_component_pixels,
    )
    return result, {
        **asdict(config),
        "image_size": [height, width],
        "head_class_pixels": int(head.sum()),
        "neck_class_pixels": int(np.count_nonzero(labels == NECK_LABEL)),
        "retained_neck_pixels": int(neck.sum()),
        "candidate_components": candidate_components,
        "output_pixels": int(result.sum()),
    }


class MediaPipeFaceAnchorDetector:
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


SessionFactory = Callable[[str, list], object]


def _create_onnx_session(path: str, providers: list) -> object:
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=providers)


class OnnxFaceParser:
    def __init__(
        self,
        model_path: Path,
        session_factory: SessionFactory = _create_onnx_session,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_sha256 = verify_model(self.model_path)
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        self._session = session_factory(str(self.model_path), providers)
        self._input_name = self._session.get_inputs()[0].name

    def __call__(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        batch = resized.astype(np.float32) / 255.0
        batch = (batch - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        batch = np.transpose(batch, (2, 0, 1))[None]
        logits = np.asarray(self._session.run(None, {self._input_name: batch})[0])
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 19:
            raise ValueError(f"Unexpected face parser output shape: {logits.shape}")
        labels = np.argmax(logits[0], axis=0).astype(np.uint8)
        return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
