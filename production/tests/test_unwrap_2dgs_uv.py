from pathlib import Path

import numpy as np
import trimesh

from production.unwrap_2dgs_uv import unwrap_mesh


def make_asymmetric_tetrahedron(path: Path) -> Path:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.2, 1.2, 0.0],
                [0.3, 0.4, 1.8],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        ),
        process=False,
    )
    mesh.export(path)
    return path


def canonical_triangles(mesh: trimesh.Trimesh) -> list[tuple[tuple[float, ...], ...]]:
    triangles = []
    for triangle in mesh.vertices[mesh.faces]:
        points = sorted(tuple(np.round(point, 7)) for point in triangle)
        triangles.append(tuple(points))
    return sorted(triangles)


def test_unwrap_preserves_triangle_geometry(tmp_path: Path) -> None:
    source = make_asymmetric_tetrahedron(tmp_path / "source.obj")
    output = tmp_path / "face.obj"

    report = unwrap_mesh(source, output, "face", "uv.png")
    source_mesh = trimesh.load_mesh(source, process=False)
    result = trimesh.load_mesh(output, process=False)

    assert report["source_faces"] == report["output_faces"] == 4
    assert canonical_triangles(result) == canonical_triangles(source_mesh)


def test_unwrap_binds_texture_and_writes_finite_uvs(tmp_path: Path) -> None:
    source = make_asymmetric_tetrahedron(tmp_path / "source.obj")
    output = tmp_path / "face.obj"

    unwrap_mesh(source, output, "face", "uv.png")

    obj_text = output.read_text(encoding="ascii")
    mtl_text = output.with_suffix(".mtl").read_text(encoding="ascii")
    loaded = trimesh.load_mesh(output, process=False)
    assert "mtllib face.mtl" in obj_text
    assert "usemtl face" in obj_text
    assert "map_Kd uv.png" in mtl_text
    assert loaded.visual.uv.shape[0] == loaded.vertices.shape[0]
    assert np.isfinite(loaded.visual.uv).all()
    assert loaded.visual.uv.min() >= 0.0
    assert loaded.visual.uv.max() <= 1.0
