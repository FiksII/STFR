from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from production.clean_face_mesh import transform_points
from production.unwrap_2dgs_uv import (
    load_triangle_mesh,
    validate_cube_uv_padding,
)


def validate_asset(
    source_path: Path,
    obj_path: Path,
    texture_path: Path,
    glb_path: Path,
    expected_texture_size: tuple[int, int] = (1024, 1024),
    uv_padding_pixels: int = 2,
    source_to_output_matrix: np.ndarray | None = None,
) -> dict:
    if expected_texture_size[0] != expected_texture_size[1]:
        raise ValueError("Cube UV validation requires a square texture")
    source = load_triangle_mesh(source_path)
    textured = load_triangle_mesh(obj_path)
    matrix = (
        np.eye(4, dtype=np.float64)
        if source_to_output_matrix is None
        else np.asarray(source_to_output_matrix, dtype=np.float64)
    )
    transformed_source_vertices = transform_points(source.vertices, matrix)
    transformed_source_bounds = np.array(
        [transformed_source_vertices.min(axis=0), transformed_source_vertices.max(axis=0)]
    )
    texture = Image.open(texture_path).convert("RGB")
    pixels = np.asarray(texture, dtype=np.float32)
    mtl_path = obj_path.with_suffix(".mtl")
    obj_text = obj_path.read_text(encoding="ascii", errors="strict")
    mtl_text = mtl_path.read_text(encoding="ascii", errors="strict")

    if len(source.faces) != len(textured.faces):
        raise ValueError("UV export changed the triangle count")
    if not np.allclose(transformed_source_bounds, textured.bounds, atol=1e-6):
        raise ValueError("UV export changed geometry bounds")
    if textured.visual.kind != "texture":
        raise ValueError(f"Expected texture visuals, got {textured.visual.kind}")
    if textured.visual.uv.shape != (len(textured.vertices), 2):
        raise ValueError("UV count does not match the seam-split vertex count")
    if not np.isfinite(textured.visual.uv).all():
        raise ValueError("UV coordinates contain non-finite values")
    if textured.visual.uv.min() < -1e-6 or textured.visual.uv.max() > 1.0 + 1e-6:
        raise ValueError("UV coordinates are outside [0, 1]")
    validate_cube_uv_padding(
        textured.visual.uv,
        atlas_size=expected_texture_size[0],
        padding_pixels=uv_padding_pixels,
    )
    if texture.size != expected_texture_size:
        raise ValueError(f"Unexpected texture size: {texture.size}")
    if float(pixels.std()) < 5.0:
        raise ValueError("Texture is nearly uniform")
    if f"mtllib {mtl_path.name}" not in obj_text:
        raise ValueError("OBJ does not reference its MTL")
    if f"map_Kd {texture_path.name}" not in mtl_text:
        raise ValueError("MTL does not reference the texture")

    glb_scene = trimesh.load(glb_path, process=False, force="scene")
    if not glb_scene.geometry:
        raise ValueError("GLB contains no geometry")
    glb_faces = sum(len(mesh.faces) for mesh in glb_scene.geometry.values())
    if glb_faces != len(textured.faces):
        raise ValueError("GLB reload changed the triangle count")
    if not np.allclose(glb_scene.bounds, textured.bounds, atol=1e-6):
        raise ValueError("GLB reload changed geometry bounds")
    if not any(mesh.visual.kind == "texture" for mesh in glb_scene.geometry.values()):
        raise ValueError("GLB does not contain texture visuals")

    return {
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "textured_vertices": int(len(textured.vertices)),
        "textured_faces": int(len(textured.faces)),
        "visual_kind": textured.visual.kind,
        "uv_method": "cube",
        "uv_padding_pixels": uv_padding_pixels,
        "uv_min": textured.visual.uv.min(axis=0).astype(float).tolist(),
        "uv_max": textured.visual.uv.max(axis=0).astype(float).tolist(),
        "texture_size": list(texture.size),
        "texture_mean_rgb": pixels.mean(axis=(0, 1)).astype(float).tolist(),
        "texture_std_rgb": pixels.std(axis=(0, 1)).astype(float).tolist(),
        "obj_mtl_binding": True,
        "glb_faces": int(glb_faces),
        "glb_bytes": int(Path(glb_path).stat().st_size),
        "bounds": textured.bounds.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a direct textured STFR face asset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate_asset(args.source, args.obj, args.texture, args.glb)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
