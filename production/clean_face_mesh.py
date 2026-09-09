from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from trimesh.smoothing import filter_taubin

from production.unwrap_2dgs_uv import load_triangle_mesh


SOURCE_TO_GLTF_Y_UP_ROW_MATRIX = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CleanupConfig:
    smooth_iterations: int = 3
    target_face_height: float = 1.35
    minimum_faces: int = 1000
    minimum_largest_component_fraction: float = 0.5
    maximum_roughness_p90_degrees: float = 30.0
    output_orientation: str = "gltf_y_up"

    def validate(self) -> None:
        if self.smooth_iterations < 0:
            raise ValueError("Smooth iterations cannot be negative")
        if self.target_face_height <= 0:
            raise ValueError("Target face height must be positive")
        if self.minimum_faces < 1:
            raise ValueError("Minimum face count must be positive")
        if not 0 < self.minimum_largest_component_fraction <= 1:
            raise ValueError("Largest component fraction must be in (0, 1]")
        if self.output_orientation != "gltf_y_up":
            raise ValueError("Production STFR output orientation must be gltf_y_up")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Transform matrix must be 4x4")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix)[:, :3]


def load_camera_to_world_matrices(path: Path) -> np.ndarray:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    matrices = np.asarray(
        [frame["transform_matrix"] for frame in metadata.get("frames", [])],
        dtype=np.float64,
    )
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices):
        raise ValueError("Camera transforms must contain at least one 4x4 matrix")
    if not np.isfinite(matrices).all():
        raise ValueError("Camera transforms contain non-finite values")
    return matrices


def canonical_face_transform(
    vertices: np.ndarray,
    camera_to_world_matrices: np.ndarray,
    target_height: float,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    cameras = np.asarray(camera_to_world_matrices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Face vertices must have shape [N, 3] with N >= 3")
    if cameras.ndim != 3 or cameras.shape[1:] != (4, 4) or not len(cameras):
        raise ValueError("Camera transforms must have shape [N, 4, 4]")
    if not np.isfinite(vertices).all() or not np.isfinite(cameras).all():
        raise ValueError("Canonicalization inputs contain non-finite values")
    if target_height <= 0:
        raise ValueError("Target face height must be positive")

    bounds_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    camera_vectors = cameras[:, :3, 3] - bounds_center
    camera_distances = np.linalg.norm(camera_vectors, axis=1, keepdims=True)
    if np.any(camera_distances <= 1e-8):
        raise ValueError("A camera is too close to the face center")
    camera_facing = (camera_vectors / camera_distances).mean(axis=0)
    facing_length = np.linalg.norm(camera_facing)
    if facing_length <= 1e-8:
        raise ValueError("Camera positions do not define a stable face direction")
    camera_facing /= facing_length

    covariance = np.cov(vertices - vertices.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    forward = eigenvectors[:, int(np.argmin(eigenvalues))]
    if np.dot(forward, camera_facing) < 0:
        forward = -forward

    camera_up = (-cameras[:, :3, 1]).mean(axis=0)
    up = camera_up - forward * np.dot(camera_up, forward)
    up_length = np.linalg.norm(up)
    if up_length <= 1e-8:
        raise ValueError("Camera orientations do not define a stable up direction")
    up /= up_length
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    basis = np.column_stack((right, up, forward))
    projected = vertices @ basis
    projected_min = projected.min(axis=0)
    projected_max = projected.max(axis=0)
    height = projected_max[1] - projected_min[1]
    if height <= 1e-8:
        raise ValueError("Face mesh has no measurable height")
    scale = target_height / height

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = basis * scale
    matrix[3, :3] = -(projected_min + projected_max) * 0.5 * scale
    return matrix


def roughness_percentiles(mesh: trimesh.Trimesh) -> dict[str, float]:
    angles = np.rad2deg(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
    if len(angles) == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0}
    median, p90, p99 = np.quantile(angles, (0.5, 0.9, 0.99))
    return {"median": float(median), "p90": float(p90), "p99": float(p99)}


def largest_face_component(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, list[int]]:
    face_count = len(mesh.faces)
    if face_count == 0:
        raise ValueError("2DGS mesh has no triangle faces")
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
        columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(face_count, face_count),
        ).tocsr()
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(face_count, dtype=np.int64)
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    largest_indices = np.flatnonzero(labels == largest_label)
    if len(largest_indices) == face_count:
        largest = mesh
    else:
        largest = mesh.submesh([largest_indices], append=True, repair=False)
    return largest, sorted(counts.astype(int).tolist(), reverse=True)


def clean_face_mesh(
    source_path: Path,
    output_path: Path,
    config: CleanupConfig = CleanupConfig(),
    camera_to_world_matrices: np.ndarray | None = None,
) -> dict:
    config.validate()
    source = load_triangle_mesh(source_path)
    cleaned_input = source.copy()
    cleaned_input.update_faces(cleaned_input.nondegenerate_faces())
    cleaned_input.update_faces(cleaned_input.unique_faces())
    cleaned_input.remove_unreferenced_vertices()

    cleaned, component_face_counts = largest_face_component(cleaned_input)
    if len(cleaned.faces) < config.minimum_faces:
        raise ValueError(
            f"Largest component has {len(cleaned.faces)} triangles, fewer than "
            f"the required {config.minimum_faces}"
        )
    component_fraction = len(cleaned.faces) / len(cleaned_input.faces)
    if component_fraction < config.minimum_largest_component_fraction:
        raise ValueError(
            f"Largest component covers only {component_fraction:.3f} of the mesh"
        )

    before_smoothing = np.asarray(cleaned.vertices).copy()
    if config.smooth_iterations:
        filter_taubin(
            cleaned,
            lamb=0.2,
            nu=0.21,
            iterations=config.smooth_iterations,
        )

    displacement = np.linalg.norm(cleaned.vertices - before_smoothing, axis=1)
    roughness = roughness_percentiles(cleaned)
    if roughness["p90"] > config.maximum_roughness_p90_degrees:
        raise ValueError(
            f"Mesh roughness p90 {roughness['p90']:.2f} degrees exceeds "
            f"{config.maximum_roughness_p90_degrees:.2f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(output_path)

    if camera_to_world_matrices is None:
        source_to_output = SOURCE_TO_GLTF_Y_UP_ROW_MATRIX
        canonicalization = "fixed_gltf_y_up"
    else:
        source_to_output = canonical_face_transform(
            np.asarray(cleaned.vertices),
            camera_to_world_matrices,
            config.target_face_height,
        )
        canonicalization = "camera_pca"

    return {
        "config": asdict(config),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "sanitized_vertices": int(len(cleaned_input.vertices)),
        "sanitized_faces": int(len(cleaned_input.faces)),
        "component_face_counts": component_face_counts[:20],
        "largest_component_fraction": float(component_fraction),
        "output_vertices": int(len(cleaned.vertices)),
        "output_faces": int(len(cleaned.faces)),
        "canonicalization": canonicalization,
        "source_to_output_row_matrix": source_to_output.tolist(),
        "smooth_displacement_mean": float(displacement.mean()),
        "smooth_displacement_p95": float(np.quantile(displacement, 0.95)),
        "smooth_displacement_max": float(displacement.max()),
        "roughness_degrees": roughness,
        "bounds": cleaned.bounds.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a detailed STFR 2DGS mesh without template registration."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--target-face-height", type=float, default=1.35)
    parser.add_argument("--transforms", type=Path)
    args = parser.parse_args()
    report = clean_face_mesh(
        args.source,
        args.output,
        CleanupConfig(
            smooth_iterations=args.smooth_iterations,
            target_face_height=args.target_face_height,
        ),
        camera_to_world_matrices=(
            load_camera_to_world_matrices(args.transforms)
            if args.transforms is not None
            else None
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
