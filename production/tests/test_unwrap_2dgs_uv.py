from pathlib import Path

import numpy as np
import pytest
import trimesh

from production.unwrap_2dgs_uv import (
    CHART_NAMES,
    build_cube_atlas,
    unwrap_mesh,
    validate_cube_uv_padding,
)


@pytest.fixture
def fake_xatlas(monkeypatch: pytest.MonkeyPatch) -> None:
    def parametrize(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mapping = np.arange(len(vertices), dtype=np.int64)
        uvs = np.array(
            [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]],
            dtype=np.float32,
        )
        return mapping, np.asarray(faces, dtype=np.int64), uvs[: len(vertices)]

    monkeypatch.setattr(
        "production.unwrap_2dgs_uv._parametrize_with_xatlas",
        parametrize,
        raising=False,
    )


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


def test_cube_atlas_hits_all_charts_and_only_splits_seams() -> None:
    source = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.faces, dtype=np.int64)

    atlas = build_cube_atlas(vertices, faces, atlas_size=1024, padding_pixels=2)

    counts = np.bincount(atlas.face_charts, minlength=len(CHART_NAMES))
    assert counts.tolist() == [2, 2, 2, 2, 2, 2]
    assert len(atlas.vertex_mapping) == 24
    assert atlas.faces.shape == faces.shape
    np.testing.assert_array_equal(
        vertices[atlas.vertex_mapping][atlas.faces],
        vertices[faces],
    )


def test_cube_atlas_uses_x_first_for_equal_normal_components() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    first = build_cube_atlas(vertices, faces)
    second = build_cube_atlas(vertices, faces)

    assert first.face_charts.tolist() == [0]
    np.testing.assert_array_equal(first.vertex_mapping, second.vertex_mapping)
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_array_equal(first.uvs, second.uvs)


def test_cube_atlas_uvs_respect_two_pixel_tile_padding() -> None:
    source = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    atlas = build_cube_atlas(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        atlas_size=1024,
        padding_pixels=2,
    )

    validate_cube_uv_padding(atlas.uvs, atlas_size=1024, padding_pixels=2)
    assert np.isfinite(atlas.uvs).all()
    assert atlas.uvs.min() >= 0.0
    assert atlas.uvs.max() <= 1.0


def test_cube_atlas_rejects_degenerate_triangles() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="degenerate triangle"):
        build_cube_atlas(vertices, faces)


@pytest.mark.parametrize(
    ("vertices", "faces", "message"),
    [
        (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]
            ),
            np.array([[0, 1, 2]]),
            "non-finite",
        ),
        (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            np.array([[0, 1, 3]]),
            "invalid face indices",
        ),
    ],
)
def test_cube_atlas_rejects_invalid_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_cube_atlas(vertices, faces)


def test_unwrap_copies_source_normals_to_every_seam_vertex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_xatlas: None,
) -> None:
    source_path = make_asymmetric_tetrahedron(tmp_path / "source.obj")
    source = trimesh.load_mesh(source_path, process=False)
    captured: dict[str, np.ndarray] = {}

    def capture_obj(
        path: Path,
        vertices: np.ndarray,
        normals: np.ndarray,
        uvs: np.ndarray,
        faces: np.ndarray,
        material_name: str,
    ) -> None:
        captured["normals"] = normals.copy()

    monkeypatch.setattr("production.unwrap_2dgs_uv.write_obj", capture_obj)
    unwrap_mesh(source_path, tmp_path / "face.obj")

    np.testing.assert_allclose(
        captured["normals"],
        np.asarray(source.vertex_normals),
    )


def test_unwrap_preserves_triangle_geometry(
    tmp_path: Path,
    fake_xatlas: None,
) -> None:
    source = make_asymmetric_tetrahedron(tmp_path / "source.obj")
    output = tmp_path / "face.obj"

    report = unwrap_mesh(source, output, "face", "uv.png")
    source_mesh = trimesh.load_mesh(source, process=False)
    result = trimesh.load_mesh(output, process=False)

    assert report["source_faces"] == report["output_faces"] == 4
    assert report["method"] == "xatlas"
    assert canonical_triangles(result) == canonical_triangles(source_mesh)


def test_unwrap_binds_texture_and_writes_finite_uvs(
    tmp_path: Path,
    fake_xatlas: None,
) -> None:
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
