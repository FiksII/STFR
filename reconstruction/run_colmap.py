from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Callable, Sequence


Runner = Callable[..., object]


def build_colmap_commands(data_root: Path) -> list[tuple[str, ...]]:
    data_root = Path(data_root)
    image_root = data_root / "images"
    mask_root = data_root / "mask"
    database_path = data_root / "database.db"
    sparse_root = data_root / "sparse"
    reconstruction_root = sparse_root / "0"

    return [
        (
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_root),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.max_image_size",
            "4000",
            "--SiftExtraction.use_gpu",
            "1",
            "--ImageReader.mask_path",
            str(mask_root),
        ),
        (
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "1",
        ),
        (
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_root),
            "--output_path",
            str(sparse_root),
        ),
        (
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(reconstruction_root),
            "--output_path",
            str(reconstruction_root),
            "--BundleAdjustment.max_num_iterations",
            "100",
        ),
        (
            "colmap",
            "model_converter",
            "--input_path",
            str(reconstruction_root),
            "--output_path",
            str(reconstruction_root),
            "--output_type",
            "TXT",
        ),
    ]


def run_colmap(
    data_root: Path,
    runner: Runner = subprocess.run,
) -> None:
    data_root = Path(data_root)
    (data_root / "sparse").mkdir(parents=True, exist_ok=True)
    for command in build_colmap_commands(data_root):
        runner(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_colmap(args.data_root)


if __name__ == "__main__":
    main()
