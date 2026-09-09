from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import xatlas


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded).__name__}")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError("Only triangular meshes are supported")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("Mesh must contain vertices and faces")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates")
    return loaded


def write_mtl(path: Path, material_name: str, texture_name: str) -> None:
    content = (
        f"newmtl {material_name}\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.020000 0.020000 0.020000\n"
        "Ns 8.000000\n"
        "illum 2\n"
        f"map_Kd {texture_name}\n"
    )
    path.write_text(content, encoding="ascii")


def write_obj(
    path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    material_name: str,
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        handle.write("o 2dgs_recon_uv\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            handle.write(f"vt {u:.9g} {v:.9g}\n")
        for nx, ny, nz in normals:
            handle.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
        handle.write(f"usemtl {material_name}\n")
        for face in faces + 1:
            a, b, c = (int(index) for index in face)
            handle.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def unwrap_mesh(
    input_path: Path,
    output_obj: Path,
    material_name: str = "face",
    texture_name: str = "uv.png",
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_obj = Path(output_obj)
    source = load_triangle_mesh(input_path)
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.uint32)
    source_normals = np.asarray(source.vertex_normals, dtype=np.float64)

    vertex_mapping, atlas_faces, atlas_uvs = xatlas.parametrize(
        source_vertices.astype(np.float32), source_faces
    )
    vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64)
    atlas_faces = np.asarray(atlas_faces, dtype=np.int64)
    atlas_uvs = np.asarray(atlas_uvs, dtype=np.float32)
    if not np.isfinite(atlas_uvs).all():
        raise ValueError("xatlas produced non-finite UV coordinates")
    tolerance = 1e-6
    if atlas_uvs.min() < -tolerance or atlas_uvs.max() > 1.0 + tolerance:
        raise ValueError("xatlas produced UV coordinates outside [0, 1]")
    atlas_uvs = np.clip(atlas_uvs, 0.0, 1.0)

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        output_obj,
        source_vertices[vertex_mapping],
        source_normals[vertex_mapping],
        atlas_uvs,
        atlas_faces,
        material_name,
    )
    write_mtl(output_obj.with_suffix(".mtl"), material_name, texture_name)
    return {
        "source": str(input_path.resolve()),
        "output": str(output_obj.resolve()),
        "source_vertices": int(len(source_vertices)),
        "output_vertices": int(len(vertex_mapping)),
        "source_faces": int(len(source_faces)),
        "output_faces": int(len(atlas_faces)),
        "uv_min": atlas_uvs.min(axis=0).astype(float).tolist(),
        "uv_max": atlas_uvs.max(axis=0).astype(float).tolist(),
        "bounds_min": source_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": source_vertices.max(axis=0).astype(float).tolist(),
        "material": material_name,
        "texture": texture_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an xatlas UV layout for a cleaned STFR 2DGS mesh."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", default="face")
    parser.add_argument("--texture", default="uv.png")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = unwrap_mesh(args.input, args.output, args.material, args.texture)
    report_path = args.report or args.output.with_suffix(".uv-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
