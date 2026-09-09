from pathlib import Path

import numpy as np
import trimesh

from production.clean_face_mesh import (
    CleanupConfig,
    clean_face_mesh,
    fit_correspondence_transform,
    load_closed_reference,
    transform_points,
)


def row_transform() -> np.ndarray:
    return np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, -3.0, 4.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_correspondence_recovers_affine_transform() -> None:
    source = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]],
        dtype=np.float64,
    )
    expected = row_transform()
    target = transform_points(source, expected)

    actual = fit_correspondence_transform(source, target)

    assert np.allclose(actual, expected)


def test_small_reference_holes_are_closed(tmp_path: Path) -> None:
    reference = trimesh.creation.box()
    reference.update_faces(np.arange(len(reference.faces)) != 0)
    path = tmp_path / "reference.obj"
    reference.export(path)

    closed = load_closed_reference(path, max_hole_size=50)

    assert len(closed.faces) > len(reference.faces)


def test_cleanup_keeps_nearby_2dgs_component_and_reports_transform(
    tmp_path: Path,
) -> None:
    face = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    noise = trimesh.Trimesh(
        vertices=np.array([[5, 5, 5], [5.1, 5, 5], [5, 5.1, 5]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    canonical = trimesh.util.concatenate((face, noise))
    matrix = row_transform()
    source = canonical.copy()
    source.vertices = transform_points(source.vertices, np.linalg.inv(matrix))
    source_path = tmp_path / "source.ply"
    canonical_path = tmp_path / "canonical.ply"
    reference_path = tmp_path / "reference.ply"
    output_source = tmp_path / "clean_source.ply"
    output_canonical = tmp_path / "clean_canonical.ply"
    source.export(source_path)
    canonical.export(canonical_path)
    face.export(reference_path)

    report = clean_face_mesh(
        source_path,
        canonical_path,
        reference_path,
        output_source,
        output_canonical,
        CleanupConfig(
            distance=0.02,
            smooth_iterations=0,
            mask_close_holes=0,
            minimum_faces=10,
            maximum_roughness_p90_degrees=180.0,
        ),
    )

    cleaned = trimesh.load_mesh(output_canonical, process=False)
    assert len(cleaned.faces) == len(face.faces)
    assert report["selected_faces"] == len(face.faces)
    assert report["component_face_counts"] == [len(face.faces)]
    assert report["correspondence_residual_max"] < 1e-6
    assert np.allclose(cleaned.bounds, face.bounds, atol=1e-6)
