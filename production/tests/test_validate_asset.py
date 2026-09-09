from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from production.export_glb import export_canonical_asset
from production.unwrap_2dgs_uv import write_mtl, write_obj
from production.validate_asset import validate_asset


def build_asset(tmp_path: Path, uniform: bool = False):
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    source_path = tmp_path / "source.ply"
    source.export(source_path)
    normals = np.asarray(source.vertex_normals)
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    obj = tmp_path / "source.obj"
    write_obj(obj, vertices, normals, uvs, faces, "face")
    mtl = obj.with_suffix(".mtl")
    write_mtl(mtl, "face", "source.png")
    texture = tmp_path / "source.png"
    pixels = np.full((4, 4, 3), 128, dtype=np.uint8)
    if not uniform:
        pixels[:2, :2] = (220, 80, 40)
    Image.fromarray(pixels).save(texture)
    exported = export_canonical_asset(obj, mtl, texture, np.eye(4), tmp_path / "output")
    return source_path, exported


def test_validate_asset_accepts_reloadable_textured_glb(tmp_path: Path) -> None:
    source, exported = build_asset(tmp_path)

    report = validate_asset(
        source,
        Path(exported["obj"]),
        Path(exported["texture"]),
        Path(exported["glb"]),
        expected_texture_size=(4, 4),
    )

    assert report["source_faces"] == report["glb_faces"] == 4
    assert report["obj_mtl_binding"] is True
    assert report["glb_bytes"] > 0


def test_validate_asset_rejects_uniform_texture(tmp_path: Path) -> None:
    source, exported = build_asset(tmp_path, uniform=True)

    with pytest.raises(ValueError, match="nearly uniform"):
        validate_asset(
            source,
            Path(exported["obj"]),
            Path(exported["texture"]),
            Path(exported["glb"]),
            expected_texture_size=(4, 4),
        )
