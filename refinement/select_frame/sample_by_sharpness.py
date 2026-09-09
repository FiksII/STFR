from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import torch


def choose_grouped_frames(
    names: Sequence[str],
    sharpness: Mapping[str, float],
    num_views: int,
) -> list[str]:
    if num_views < 1:
        raise ValueError("num_views must be positive")
    if not names:
        return []

    count = min(num_views, len(names))
    selected = []
    for group_index in range(count):
        left = group_index * len(names) // count
        right = (group_index + 1) * len(names) // count
        group = names[left:right]
        selected.append(max(group, key=lambda name: sharpness[name]))
    return selected


def sample_frames(
    image_root: Path,
    camera_path: Path,
    output_root: Path,
    num_views: int = 16,
) -> list[str]:
    image_root = Path(image_root)
    output_root = Path(output_root)
    metadata = json.loads(Path(camera_path).read_text(encoding="utf-8"))
    frame_by_name = {
        Path(frame["file_path"]).name: frame for frame in metadata.get("frames", [])
    }
    image_names = sorted(
        path.name for path in image_root.glob("*.png") if path.is_file()
    )
    eligible_names = [name for name in image_names if name in frame_by_name]
    if not eligible_names:
        raise ValueError("No raw PNG frames match camera metadata")

    sharpness = torch.load(output_root / "sharpness.pkl", map_location="cpu")
    selected_names = choose_grouped_frames(eligible_names, sharpness, num_views)
    selected_image_root = output_root / "image"
    selected_image_root.mkdir(parents=True, exist_ok=True)
    for existing in selected_image_root.glob("*.png"):
        existing.unlink()
    for name in selected_names:
        shutil.copy2(image_root / name, selected_image_root / name)

    selected_metadata = dict(metadata)
    selected_metadata["frames"] = [frame_by_name[name] for name in selected_names]
    (output_root / "select_sharp.json").write_text(
        json.dumps(selected_metadata, indent=4) + "\n",
        encoding="utf-8",
    )
    return selected_names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--img_root", type=Path, required=True)
    parser.add_argument("--cam_path", type=Path, required=True)
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--num_view", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    sample_frames(
        options.img_root,
        options.cam_path,
        options.save_root,
        options.num_view,
    )


if __name__ == "__main__":
    main()
