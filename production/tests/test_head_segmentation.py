from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import production.head_segmentation as segmentation
from production.head_segmentation import (
    HeadMaskConfig,
    OnnxFaceParser,
    build_head_mask,
    face_oval_mask,
)


def square_landmarks() -> np.ndarray:
    return np.array(
        [[6.0, 8.0], [14.0, 8.0], [14.0, 14.0], [6.0, 14.0]],
        dtype=np.float64,
    )


def test_head_mask_keeps_head_and_short_neck_but_excludes_accessories() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[3:8, 7:13] = 17
    labels[8:14, 6:14] = 1
    labels[9:12, 5:6] = 7
    labels[8:10, 8:12] = 6
    labels[14:20, 8:12] = 14
    labels[18:20, :] = 16

    mask, report = build_head_mask(
        labels,
        square_landmarks(),
        HeadMaskConfig(close_kernel_ratio=0.0, minimum_component_pixels=1),
    )

    assert mask[4, 10]
    assert mask[10, 5]
    assert not mask[9, 10]
    assert mask[15, 10]
    assert not mask[17, 10]
    assert not mask[18, 10]
    assert report["neck_height_ratio"] == 0.45
    assert report["output_pixels"] == int(mask.sum())


def test_head_mask_limits_neck_to_jaw_width() -> None:
    labels = np.zeros((24, 30), dtype=np.uint8)
    labels[8:15, 6:15] = 1
    labels[14:18, :] = 14

    mask, _ = build_head_mask(
        labels,
        square_landmarks(),
        HeadMaskConfig(close_kernel_ratio=0.0, minimum_component_pixels=1),
    )

    assert mask[15, 10]
    assert not mask[15, 2]
    assert not mask[15, 20]


def test_head_mask_keeps_component_with_largest_face_anchor_overlap() -> None:
    labels = np.zeros((24, 30), dtype=np.uint8)
    labels[8:15, 6:15] = 1
    labels[2:7, 22:28] = 17

    mask, report = build_head_mask(
        labels,
        square_landmarks(),
        HeadMaskConfig(close_kernel_ratio=0.0, minimum_component_pixels=1),
    )

    assert mask[10, 10]
    assert not mask[4, 24]
    assert report["candidate_components"] == 2


def test_head_mask_rejects_component_without_face_anchor() -> None:
    labels = np.zeros((24, 30), dtype=np.uint8)
    labels[2:7, 22:28] = 17

    with pytest.raises(ValueError, match="face anchor"):
        build_head_mask(
            labels,
            square_landmarks(),
            HeadMaskConfig(close_kernel_ratio=0.0, minimum_component_pixels=1),
        )


def test_face_oval_mask_matches_image_size_and_contains_center() -> None:
    mask = face_oval_mask(square_landmarks(), (24, 30))

    assert mask.shape == (24, 30)
    assert mask.dtype == np.bool_
    assert mask[11, 10]
    assert not mask[0, 0]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (HeadMaskConfig(neck_height_ratio=-0.1), "Neck height"),
        (HeadMaskConfig(close_kernel_ratio=-0.1), "Close kernel"),
        (HeadMaskConfig(minimum_component_pixels=0), "component pixels"),
    ],
)
def test_head_mask_config_rejects_invalid_values(
    config: HeadMaskConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


class FakeInput:
    name = "input"


class FakeSession:
    def __init__(self) -> None:
        self.batch: np.ndarray | None = None

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, output_names, inputs):
        self.batch = inputs["input"]
        logits = np.zeros((1, 19, 512, 512), dtype=np.float32)
        logits[:, 17, :, 256:] = 1.0
        return [logits]


def test_onnx_face_parser_restores_source_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (20, 10), (128, 64, 32)).save(image_path)
    fake_session = FakeSession()
    providers = []

    def make_session(path, configured_providers):
        providers.extend(configured_providers)
        return fake_session

    monkeypatch.setattr(segmentation, "verify_model", lambda path: "model-sha")
    parser = OnnxFaceParser(
        tmp_path / "model.onnx",
        session_factory=make_session,
    )

    labels = parser(image_path)

    assert labels.shape == (10, 20)
    assert np.all(labels[:, :10] == 0)
    assert np.all(labels[:, 10:] == 17)
    assert fake_session.batch is not None
    assert fake_session.batch.shape == (1, 3, 512, 512)
    assert providers == ["CPUExecutionProvider"]
