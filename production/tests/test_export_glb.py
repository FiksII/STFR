import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from production.clean_face_mesh import transform_points
from production.export_glb import export_canonical_asset, load_output_transform
from production.unwrap_2dgs_uv import write_mtl, write_obj


def make_textured_triangle(root: Path) -> tuple[Path, Path, Path, np.ndarray]:
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.tile(np.array([[0, 0, 1]], dtype=np.float64), (3, 1))
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    obj = root / "source.obj"
    write_obj(obj, vertices, normals, uvs, faces, "face")
    mtl = obj.with_suffix(".mtl")
    write_mtl(mtl, "face", "source.png")
    texture = root / "source.png"
    Image.new("RGB", (4, 4), (120, 80, 40)).save(texture)
    return obj, mtl, texture, vertices


def test_export_transforms_positions_normals_and_reloads_glb(tmp_path: Path) -> None:
    obj, mtl, texture, vertices = make_textured_triangle(tmp_path)
    matrix = np.array(
        [
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [3, 4, 5, 1],
        ],
        dtype=np.float64,
    )

    report = export_canonical_asset(obj, mtl, texture, matrix, tmp_path / "output")

    scene = trimesh.load(report["glb"], process=False, force="scene")
    assert sum(len(mesh.faces) for mesh in scene.geometry.values()) == 1
    assert np.allclose(scene.bounds, np.array([transform_points(vertices, matrix).min(0), transform_points(vertices, matrix).max(0)]))
    output_obj = Path(report["obj"])
    normal_line = next(
        line for line in output_obj.read_text(encoding="ascii").splitlines() if line.startswith("vn ")
    )
    assert np.allclose(np.fromstring(normal_line[3:], sep=" "), [0, 0, 1])
    assert "map_Kd uv.png" in Path(report["mtl"]).read_text(encoding="ascii")


def test_export_writes_vertex_color_ply(tmp_path: Path) -> None:
    obj, mtl, texture, _ = make_textured_triangle(tmp_path)

    report = export_canonical_asset(obj, mtl, texture, np.eye(4), tmp_path / "output")

    loaded = trimesh.load_mesh(report["vertex_color_ply"], process=False)
    assert loaded.visual.kind == "vertex"
    assert len(loaded.visual.vertex_colors) == len(loaded.vertices)


def test_load_output_transform_uses_direct_cleanup_report_key(
    tmp_path: Path,
) -> None:
    report = tmp_path / "geometry-report.json"
    matrix = np.eye(4).tolist()
    report.write_text(
        json.dumps({"source_to_output_row_matrix": matrix}),
        encoding="utf-8",
    )

    np.testing.assert_array_equal(load_output_transform(report), np.eye(4))
