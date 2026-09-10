from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

import production.download_models as download_cli
from production.model_assets import (
    FACE_PARSING_MODEL,
    ModelAsset,
    download_model,
    verify_model,
)


def asset_for(content: bytes) -> ModelAsset:
    return ModelAsset(
        filename="test.onnx",
        url="https://example.invalid/test.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_face_parser_asset_is_pinned() -> None:
    assert FACE_PARSING_MODEL.url == (
        "https://github.com/yakhyo/face-parsing/releases/download/weights/"
        "resnet18.onnx"
    )
    assert FACE_PARSING_MODEL.sha256 == (
        "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
    )


def test_verify_model_accepts_matching_content(tmp_path: Path) -> None:
    content = b"valid-model"
    path = tmp_path / "test.onnx"
    path.write_bytes(content)

    assert verify_model(path, asset_for(content)) == hashlib.sha256(content).hexdigest()


def test_verify_model_rejects_wrong_content(tmp_path: Path) -> None:
    path = tmp_path / "test.onnx"
    path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_model(path, asset_for(b"expected"))


def test_download_model_replaces_destination_only_after_verification(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"old")
    content = b"new-model"

    report = download_model(
        destination,
        asset_for(content),
        opener=lambda _: io.BytesIO(content),
    )

    assert destination.read_bytes() == content
    assert not destination.with_name(f".{destination.name}.tmp").exists()
    assert report["bytes"] == len(content)
    assert report["sha256"] == hashlib.sha256(content).hexdigest()


def test_download_model_keeps_existing_file_when_download_is_invalid(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"old")

    with pytest.raises(ValueError, match="SHA-256"):
        download_model(
            destination,
            asset_for(b"expected"),
            opener=lambda _: io.BytesIO(b"corrupt"),
        )

    assert destination.read_bytes() == b"old"
    assert not destination.with_name(f".{destination.name}.tmp").exists()


def test_download_models_cli_prints_install_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "face-parser.onnx"
    expected = {"output": str(output), "bytes": 123, "sha256": "abc"}
    monkeypatch.setattr(download_cli, "download_model", lambda path: expected)

    assert download_cli.main(["--output", str(output)]) == 0
    assert '"bytes": 123' in capsys.readouterr().out
