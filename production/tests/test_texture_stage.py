import json
from pathlib import Path

from PIL import Image
import pytest

from production.texture_stage import TextureConfig, prepare_dataset


def test_prepare_dataset_keeps_only_refinement_frames(tmp_path: Path) -> None:
    source = tmp_path / "workspace"
    selected = source / "refinement" / "sample" / "image"
    selected.mkdir(parents=True)
    masks = source / "mask"
    masks.mkdir(parents=True)
    Image.new("RGB", (2, 2), "red").save(selected / "00001.png")
    Image.new("RGB", (2, 2), "blue").save(selected / "00003.png")
    Image.new("L", (2, 2), 255).save(masks / "00001.png")
    Image.new("L", (2, 2), 255).save(masks / "00003.png")
    metadata = {
        "w": 2,
        "h": 2,
        "fl_x": 2.0,
        "fl_y": 2.0,
        "cx": 1.0,
        "cy": 1.0,
        "frames": [
            {"file_path": f"/capture/{index:05d}.png", "transform_matrix": []}
            for index in range(1, 4)
        ],
    }
    (source / "transforms.json").write_text(json.dumps(metadata), encoding="utf-8")
    dataset = tmp_path / "dataset"

    report = prepare_dataset(source, dataset)

    result = json.loads((dataset / "transforms.json").read_text(encoding="utf-8"))
    assert report["selected_images"] == 2
    assert [Path(frame["file_path"]).name for frame in result["frames"]] == [
        "00001.png",
        "00003.png",
    ]
    assert sorted(path.name for path in (dataset / "image").glob("*.png")) == [
        "00001.png",
        "00003.png",
    ]
    assert sorted(path.name for path in (dataset / "mask").glob("*.png")) == [
        "00001.png",
        "00003.png",
    ]


def test_texture_config_exposes_production_cube_uv_options(tmp_path: Path) -> None:
    mesh = tmp_path / "face.ply"
    mesh.write_bytes(b"ply")
    config = TextureConfig(tmp_path, tmp_path, mesh, tmp_path / "artifacts")

    config.validate()

    assert config.uv_method == "cube"
    assert config.uv_options() == {"atlas_size": 1024, "padding_pixels": 2}


def test_texture_config_rejects_padding_that_consumes_a_tile(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "face.ply"
    mesh.write_bytes(b"ply")
    config = TextureConfig(
        tmp_path,
        tmp_path,
        mesh,
        tmp_path / "artifacts",
        atlas_size=6,
        uv_padding_pixels=1,
    )

    with pytest.raises(ValueError, match="no usable cube-atlas tile area"):
        config.validate()
