from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.smoothing import filter_laplacian

from production.unwrap_2dgs_uv import load_triangle_mesh


SOURCE_TO_GLTF_Y_UP_ROW_MATRIX = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CleanupConfig:
    smooth_iterations: int = 20
    minimum_faces: int = 1000
    minimum_largest_component_fraction: float = 0.5
    maximum_roughness_p90_degrees: float = 30.0
    output_orientation: str = "gltf_y_up"

    def validate(self) -> None:
        if self.smooth_iterations < 0:
            raise ValueError("Smooth iterations cannot be negative")
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


def roughness_percentiles(mesh: trimesh.Trimesh) -> dict[str, float]:
    angles = np.rad2deg(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
    if len(angles) == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0}
    median, p90, p99 = np.quantile(angles, (0.5, 0.9, 0.99))
    return {"median": float(median), "p90": float(p90), "p99": float(p99)}


def clean_face_mesh(
    source_path: Path,
    output_path: Path,
    config: CleanupConfig = CleanupConfig(),
) -> dict:
    config.validate()
    source = load_triangle_mesh(source_path)
    cleaned_input = source.copy()
    cleaned_input.update_faces(cleaned_input.nondegenerate_faces())
    cleaned_input.update_faces(cleaned_input.unique_faces())
    cleaned_input.remove_unreferenced_vertices()

    components = list(cleaned_input.split(only_watertight=False, repair=False))
    if not components:
        raise ValueError("2DGS mesh has no connected triangle components")
    component_face_counts = sorted(
        (len(component.faces) for component in components), reverse=True
    )
    cleaned = max(components, key=lambda component: len(component.faces))
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
        filter_laplacian(
            cleaned,
            lamb=0.2,
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
        "source_to_output_row_matrix": SOURCE_TO_GLTF_Y_UP_ROW_MATRIX.tolist(),
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
    parser.add_argument("--smooth-iterations", type=int, default=20)
    args = parser.parse_args()
    report = clean_face_mesh(
        args.source,
        args.output,
        CleanupConfig(smooth_iterations=args.smooth_iterations),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
