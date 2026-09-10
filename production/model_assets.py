from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import BinaryIO, Callable
from urllib.request import urlopen


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    sha256: str


FACE_PARSING_MODEL = ModelAsset(
    filename="face-parsing-resnet18.onnx",
    url=(
        "https://github.com/yakhyo/face-parsing/releases/download/weights/"
        "resnet18.onnx"
    ),
    sha256="0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, asset: ModelAsset = FACE_PARSING_MODEL) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != asset.sha256:
        raise ValueError(
            f"Model SHA-256 mismatch for {path}: expected {asset.sha256}, "
            f"got {actual}"
        )
    return actual


Opener = Callable[[str], BinaryIO]


def download_model(
    destination: Path,
    asset: ModelAsset = FACE_PARSING_MODEL,
    opener: Opener = urlopen,
) -> dict:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with opener(asset.url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        digest = verify_model(temporary, asset)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "source": asset.url,
    }
