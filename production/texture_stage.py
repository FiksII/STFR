from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

from production.runtime import CommandSpec, Emitter, emit_json, run_checked
from production.unwrap_2dgs_uv import unwrap_mesh


def filter_transforms(metadata: dict, selected_names: set[str]) -> dict:
    filtered = copy.deepcopy(metadata)
    filtered["frames"] = [
        frame
        for frame in metadata.get("frames", [])
        if Path(frame["file_path"]).name in selected_names
    ]
    if not filtered["frames"]:
        raise ValueError("No selected images match transforms.json")
    return filtered


def prepare_dataset(source_root: Path, dataset_root: Path) -> dict:
    source_root = Path(source_root)
    dataset_root = Path(dataset_root)
    selected_root = source_root / "refinement" / "sample" / "image"
    transforms_path = source_root / "transforms.json"
    source_masks = source_root / "mask"
    selected_images = sorted(selected_root.glob("*.png"))
    if not selected_images:
        raise FileNotFoundError(f"No selected PNG frames under {selected_root}")
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)

    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    selected_names = {image.name for image in selected_images}
    filtered = filter_transforms(metadata, selected_names)
    output_images = dataset_root / "image"
    output_masks = dataset_root / "mask"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    for image in selected_images:
        mask = source_masks / image.name
        if not mask.is_file():
            raise FileNotFoundError(mask)
        shutil.copy2(image, output_images / image.name)
        shutil.copy2(mask, output_masks / image.name)
    for frame in filtered["frames"]:
        frame["file_path"] = str(
            (output_images / Path(frame["file_path"]).name).resolve()
        )
    (dataset_root / "transforms.json").write_text(
        json.dumps(filtered, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "selected_images": len(selected_images),
        "matched_cameras": len(filtered["frames"]),
        "dataset_root": str(dataset_root.resolve()),
    }


@dataclass(frozen=True)
class TextureConfig:
    code_root: Path
    source_root: Path
    source_mesh: Path
    output_root: Path
    python: str = "python"
    physical_gpu: int = 1
    iterations: int = 301
    uv_method: str = "cube"
    atlas_size: int = 1024
    uv_padding_pixels: int = 2

    def uv_options(self) -> dict[str, int]:
        return {
            "atlas_size": self.atlas_size,
            "padding_pixels": self.uv_padding_pixels,
        }

    def validate(self) -> None:
        if not self.source_mesh.is_file():
            raise FileNotFoundError(self.source_mesh)
        if self.physical_gpu < 0:
            raise ValueError("Physical GPU index cannot be negative")
        if self.iterations < 1:
            raise ValueError("Texture iterations must be positive")
        if self.uv_method != "cube":
            raise ValueError("Production texture stage requires cube UVs")
        if self.atlas_size < 1 or self.uv_padding_pixels < 0:
            raise ValueError("Atlas size must be positive and padding non-negative")
        pad = self.uv_padding_pixels / self.atlas_size
        if 2.0 * pad >= 1.0 / 3.0:
            raise ValueError("Padding leaves no usable cube-atlas tile area")


def run_texture_stage(
    config: TextureConfig,
    environment: Mapping[str, str] | None = None,
    emit: Emitter = emit_json,
) -> dict:
    config.validate()
    output_root = config.output_root.resolve()
    dataset_root = output_root / "texture_dataset"
    output_obj = output_root / "face_uv_source.obj"
    output_texture = output_root / "uv.png"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_report = prepare_dataset(config.source_root, dataset_root)
    uv_report = unwrap_mesh(
        config.source_mesh,
        output_obj,
        "face",
        output_texture.name,
        **config.uv_options(),
    )
    (output_root / "texture-prepare-report.json").write_text(
        json.dumps({"dataset": dataset_report, "uv": uv_report}, indent=2) + "\n",
        encoding="utf-8",
    )

    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    child_environment["CUDA_VISIBLE_DEVICES"] = str(config.physical_gpu)
    child_environment["PYTHONPATH"] = str(config.code_root.resolve())
    texture_root = config.code_root / "texture"
    commands = [
        CommandSpec(
            "render_texture_gbuffer",
            (
                config.python,
                str(texture_root / "render_gbuffer.py"),
                "--data_root",
                str(dataset_root),
                "--mesh_path",
                str(output_obj),
                "--device",
                "0",
            ),
            texture_root,
        ),
        CommandSpec(
            "train_texture",
            (
                config.python,
                str(texture_root / "build_texture.py"),
                "--data_root",
                str(dataset_root),
                "--mesh_path",
                str(output_obj),
                "--device",
                "0",
                "--iterations",
                str(config.iterations),
            ),
            texture_root,
        ),
    ]
    for command in commands:
        run_checked(command, child_environment, emit)

    trained_texture = (
        dataset_root
        / "texture"
        / "image"
        / f"{config.iterations - 1:05d}"
        / "uv.png"
    )
    if not trained_texture.is_file():
        raise FileNotFoundError(f"Texture trainer did not produce {trained_texture}")
    shutil.copy2(trained_texture, output_texture)
    return {
        "obj": str(output_obj),
        "mtl": str(output_obj.with_suffix(".mtl")),
        "texture": str(output_texture),
        "iterations": config.iterations,
        "physical_gpu": config.physical_gpu,
        "dataset": dataset_report,
        "uv": uv_report,
    }
