# Direct Cube-UV Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the production xatlas unwrap with deterministic linear-time cube UV generation on the full cleaned `2dgs_recon` mesh and publish a validated textured GLB without Wrap, FLAME, or decimation.

**Architecture:** `production.unwrap_2dgs_uv` assigns each source triangle to one of six dominant-normal charts, splits only `(source vertex, chart)` seams, and packs the charts into a padded 3-by-2 atlas. `TextureConfig` carries the fixed production atlas contract into texture training and resume state, while final validation independently checks topology, bounds, UV range, chart padding, texture binding, and GLB reloadability.

**Tech Stack:** Python 3.10, NumPy 1.26.4, trimesh 5.1.0, Pillow 12.3.0, pytest 8, uv 0.12.10, STFR/PyTorch CUDA texture renderer.

**Spec:** `docs/superpowers/specs/2026-09-09-direct-cube-uv-pipeline-design.md`

## Global Constraints

- Preserve every triangle produced by direct cleanup; UV generation must not remove, reorder, flip, smooth, repair, or decimate geometry.
- Production uses the full cleaned 1.7-million-vertex, 3.3-million-face integration mesh.
- Generated vertices are uniquely identified by `(source_vertex_index, chart_id)` and positions/normals are copied from that source vertex.
- Chart order is `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`; dominant-axis ties prefer `X`, then `Y`, then `Z`.
- Atlas layout is three columns by two rows, texture size is 1024 by 1024, and chart padding is two pixels.
- Production remains independent of Faceform Wrap, FLAME registration, xatlas, pymeshlab, and mesh decimation.
- Keep the existing 30,000-step 2DGS reconstruction, mesh resolution 1024, 20 cleanup smoothing iterations, 301 texture iterations, and physical GPU index 1 defaults.
- Preserve existing user changes. Do not reset the working tree; commit only the files owned by each task.
- Run every `uv` and pytest command in `/home/ii/STFR-production` on the Ubuntu server after synchronizing the files owned by that red/green step; the Windows checkout cannot resolve the Linux-only lock environment.

---

## File Map

- `production/unwrap_2dgs_uv.py`: cube chart assignment, seam remap, padded atlas projection, OBJ/MTL serialization, CLI, and UV report.
- `production/tests/test_unwrap_2dgs_uv.py`: cube topology, deterministic mapping, padding, invalid input, and material tests.
- `production/texture_stage.py`: production atlas configuration and handoff to the unwrap function.
- `production/run_video_to_glb.py`: mesh fingerprint in resume configuration and atlas-aware final validation.
- `production/tests/test_texture_stage.py`: atlas configuration validation and forwarding contract.
- `production/tests/test_run_video_to_glb.py`: source-mesh fingerprint behavior and unchanged stage graph.
- `production/validate_asset.py`: independent cube tile-padding validation in the final publication gate.
- `production/export_glb.py`: direct-output terminology and identity transform report key used by its CLI.
- `production/tests/test_validate_asset.py`: accepted padded UVs and rejection of tile-edge UVs.
- `production/tests/test_export_glb.py`: direct transform report compatibility.
- `pyproject.toml`, `uv.lock`, `production/tests/test_pyproject.py`: remove the xatlas production dependency and lock entry.
- `README.md`, `ENV.md`, `AGENTS.md`: document deterministic cube UVs and the absence of xatlas from the production environment.

---

### Task 0: Preserve the Verified Direct No-Wrap Baseline

**Files:**
- Modify: `production/clean_face_mesh.py`
- Modify: `production/reconstruction_stage.py`
- Modify: `production/tests/test_clean_face_mesh.py`
- Modify: `production/tests/test_reconstruction_stage.py`

**Interfaces:**
- Consumes: existing accepted direct-cleanup work already present in the checkout.
- Produces: a committed baseline where reconstruction still selects sharp frames but does not register, and cleanup emits full source-space largest-component geometry plus `source_to_output_row_matrix`.

- [ ] **Step 1: Inspect the baseline diff without changing it**

```bash
git diff -- production/clean_face_mesh.py production/reconstruction_stage.py production/tests/test_clean_face_mesh.py production/tests/test_reconstruction_stage.py
git diff --cached -- production/clean_face_mesh.py production/reconstruction_stage.py production/tests/test_clean_face_mesh.py production/tests/test_reconstruction_stage.py
```

Expected: reconstruction retains `refinement` for sharp-frame selection and removes only registration; cleanup has no Wrap/reference input, keeps the largest component, applies 20 Laplacian passes, and returns an identity `source_to_output_row_matrix`.

- [ ] **Step 2: Synchronize these four files and re-run the verified tests on the server**

```bash
uv run pytest -q production/tests/test_clean_face_mesh.py production/tests/test_reconstruction_stage.py
```

Expected: all direct cleanup and reconstruction tests pass.

- [ ] **Step 3: Commit only the direct baseline files**

```bash
git add production/clean_face_mesh.py production/reconstruction_stage.py production/tests/test_clean_face_mesh.py production/tests/test_reconstruction_stage.py
git commit --only -m "feat: use direct 2DGS geometry in production" -- production/clean_face_mesh.py production/reconstruction_stage.py production/tests/test_clean_face_mesh.py production/tests/test_reconstruction_stage.py
```

---

### Task 1: Deterministic Cube Atlas Core

**Files:**
- Modify: `production/unwrap_2dgs_uv.py:1-133`
- Modify: `production/tests/test_unwrap_2dgs_uv.py:1-65`

**Interfaces:**
- Consumes: triangle vertex positions `float64[V,3]` and face indices `int64[F,3]`.
- Produces: `CubeAtlas(vertex_mapping, faces, uvs, face_charts)`, `build_cube_atlas(vertices, faces, atlas_size=1024, padding_pixels=2)`, `validate_cube_uv_padding(uvs, atlas_size, padding_pixels)`, and the extended `unwrap_mesh(..., *, atlas_size=1024, padding_pixels=2)`.

- [ ] **Step 1: Replace xatlas-oriented unit fixtures with cube-atlas expectations**

Add these imports and tests while retaining the current triangle-geometry and MTL assertions:

```python
import pytest

from production.unwrap_2dgs_uv import (
    CHART_NAMES,
    build_cube_atlas,
    unwrap_mesh,
    validate_cube_uv_padding,
)


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
            np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0, 1, 2]]),
            "non-finite",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0, 1, 3]]),
            "invalid face indices",
        ),
    ],
)
def test_cube_atlas_rejects_invalid_arrays(vertices, faces, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_cube_atlas(vertices, faces)


def test_unwrap_copies_source_normals_to_every_seam_vertex(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = make_asymmetric_tetrahedron(tmp_path / "source.obj")
    source = trimesh.load_mesh(source_path, process=False)
    atlas = build_cube_atlas(source.vertices, source.faces)
    captured = {}

    def capture_obj(path, vertices, normals, uvs, faces, material_name):
        captured["normals"] = normals.copy()

    monkeypatch.setattr("production.unwrap_2dgs_uv.write_obj", capture_obj)
    unwrap_mesh(source_path, tmp_path / "face.obj")

    np.testing.assert_allclose(
        captured["normals"],
        np.asarray(source.vertex_normals)[atlas.vertex_mapping],
    )
```

- [ ] **Step 2: Run the new tests and verify the red state**

Run:

```bash
uv run pytest -q production/tests/test_unwrap_2dgs_uv.py
```

Expected: collection fails because `CHART_NAMES`, `CubeAtlas`, `build_cube_atlas`, and `validate_cube_uv_padding` do not exist.

- [ ] **Step 3: Implement chart assignment, seam splitting, and padded projection**

Remove `import xatlas` and add the following public data contract and equivalent vectorized implementation:

```python
from dataclasses import dataclass

CHART_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
ATLAS_COLUMNS = 3
ATLAS_ROWS = 2


@dataclass(frozen=True)
class CubeAtlas:
    vertex_mapping: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    face_charts: np.ndarray


def _normalized_positions(vertices: np.ndarray) -> np.ndarray:
    bounds_min = vertices.min(axis=0)
    extent = vertices.max(axis=0) - bounds_min
    scale = max(float(extent.max()), 1.0)
    normalized = np.full(vertices.shape, 0.5, dtype=np.float64)
    usable = extent > scale * 1e-12
    normalized[:, usable] = (
        vertices[:, usable] - bounds_min[usable]
    ) / extent[usable]
    return normalized


def _local_chart_uv(normalized: np.ndarray, charts: np.ndarray) -> np.ndarray:
    x, y, z = normalized.T
    local = np.empty((len(normalized), 2), dtype=np.float64)
    projections = (
        (1.0 - z, y),
        (z, y),
        (x, 1.0 - z),
        (x, z),
        (x, y),
        (1.0 - x, y),
    )
    for chart_id, (u, v) in enumerate(projections):
        selected = charts == chart_id
        local[selected, 0] = u[selected]
        local[selected, 1] = v[selected]
    return local


def build_cube_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    atlas_size: int = 1024,
    padding_pixels: int = 2,
) -> CubeAtlas:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Faces must have shape [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh contains invalid face indices")
    if atlas_size < 1 or padding_pixels < 0:
        raise ValueError("Atlas size must be positive and padding non-negative")

    pad = padding_pixels / atlas_size
    tile_width = 1.0 / ATLAS_COLUMNS
    tile_height = 1.0 / ATLAS_ROWS
    if 2.0 * pad >= min(tile_width, tile_height):
        raise ValueError("Padding leaves no usable cube-atlas tile area")

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    mesh_scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    if np.any(lengths <= mesh_scale * mesh_scale * 1e-14):
        raise ValueError("Mesh contains a degenerate triangle")
    axes = np.argmax(np.abs(face_normals), axis=1)
    signs = face_normals[np.arange(len(faces)), axes] < 0.0
    face_charts = axes * 2 + signs.astype(np.int64)

    corner_keys = (
        face_charts[:, None] * len(vertices) + faces
    ).reshape(-1)
    unique_keys, inverse = np.unique(corner_keys, return_inverse=True)
    vertex_mapping = unique_keys % len(vertices)
    vertex_charts = unique_keys // len(vertices)
    remapped_faces = inverse.reshape(faces.shape)

    normalized = _normalized_positions(vertices)[vertex_mapping]
    local = _local_chart_uv(normalized, vertex_charts)
    columns = vertex_charts % ATLAS_COLUMNS
    rows = vertex_charts // ATLAS_COLUMNS
    uvs = np.empty_like(local)
    uvs[:, 0] = (
        columns * tile_width
        + pad
        + local[:, 0] * (tile_width - 2.0 * pad)
    )
    uvs[:, 1] = (
        rows * tile_height
        + pad
        + local[:, 1] * (tile_height - 2.0 * pad)
    )
    validate_cube_uv_padding(uvs, atlas_size, padding_pixels)
    return CubeAtlas(vertex_mapping, remapped_faces, uvs, face_charts)
```

Implement `validate_cube_uv_padding` by inferring each UV's row and column, then checking it against that tile's inset:

```python
def validate_cube_uv_padding(
    uvs: np.ndarray,
    atlas_size: int,
    padding_pixels: int,
) -> None:
    uvs = np.asarray(uvs, dtype=np.float64)
    if atlas_size < 1 or padding_pixels < 0:
        raise ValueError("Atlas size must be positive and padding non-negative")
    pad = padding_pixels / atlas_size
    if 2.0 * pad >= 1.0 / ATLAS_COLUMNS:
        raise ValueError("Padding leaves no usable cube-atlas tile area")
    if uvs.ndim != 2 or uvs.shape[1] != 2 or not len(uvs):
        raise ValueError("UV coordinates must have shape [N, 2]")
    if not np.isfinite(uvs).all():
        raise ValueError("UV coordinates contain non-finite values")
    tolerance = 1e-9
    if uvs.min() < -tolerance or uvs.max() > 1.0 + tolerance:
        raise ValueError("UV coordinates are outside [0, 1]")
    columns = np.minimum((uvs[:, 0] * ATLAS_COLUMNS).astype(np.int64), 2)
    rows = np.minimum((uvs[:, 1] * ATLAS_ROWS).astype(np.int64), 1)
    lower_u = columns / ATLAS_COLUMNS + pad
    upper_u = (columns + 1) / ATLAS_COLUMNS - pad
    lower_v = rows / ATLAS_ROWS + pad
    upper_v = (rows + 1) / ATLAS_ROWS - pad
    valid = (
        (uvs[:, 0] >= lower_u - tolerance)
        & (uvs[:, 0] <= upper_u + tolerance)
        & (uvs[:, 1] >= lower_v - tolerance)
        & (uvs[:, 1] <= upper_v + tolerance)
    )
    if not np.all(valid):
        raise ValueError("UV coordinates violate cube-atlas tile padding")
```

- [ ] **Step 4: Replace `xatlas.parametrize` inside `unwrap_mesh`**

Keep the first four positional parameters backward compatible and make atlas controls keyword-only:

```python
def unwrap_mesh(
    input_path: Path,
    output_obj: Path,
    material_name: str = "face",
    texture_name: str = "uv.png",
    *,
    atlas_size: int = 1024,
    padding_pixels: int = 2,
) -> dict[str, Any]:
    source = load_triangle_mesh(Path(input_path))
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    source_normals = np.asarray(source.vertex_normals, dtype=np.float64)
    if not np.isfinite(source_normals).all():
        raise ValueError("Mesh contains non-finite vertex normals")

    atlas = build_cube_atlas(
        source_vertices,
        source_faces,
        atlas_size=atlas_size,
        padding_pixels=padding_pixels,
    )
    output_obj = Path(output_obj)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        output_obj,
        source_vertices[atlas.vertex_mapping],
        source_normals[atlas.vertex_mapping],
        atlas.uvs,
        atlas.faces,
        material_name,
    )
    write_mtl(output_obj.with_suffix(".mtl"), material_name, texture_name)
    counts = np.bincount(atlas.face_charts, minlength=len(CHART_NAMES))
    return {
        "method": "cube",
        "atlas_size": atlas_size,
        "padding_pixels": padding_pixels,
        "chart_face_counts": dict(zip(CHART_NAMES, counts.astype(int).tolist())),
        "source": str(Path(input_path).resolve()),
        "output": str(output_obj.resolve()),
        "source_vertices": int(len(source_vertices)),
        "output_vertices": int(len(atlas.vertex_mapping)),
        "referenced_source_vertices": int(len(np.unique(source_faces))),
        "duplicated_seam_vertices": int(
            len(atlas.vertex_mapping) - len(np.unique(source_faces))
        ),
        "source_faces": int(len(source_faces)),
        "output_faces": int(len(atlas.faces)),
        "uv_min": atlas.uvs.min(axis=0).astype(float).tolist(),
        "uv_max": atlas.uvs.max(axis=0).astype(float).tolist(),
        "bounds_min": source_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": source_vertices.max(axis=0).astype(float).tolist(),
        "material": material_name,
        "texture": texture_name,
    }
```

Add CLI flags `--atlas-size` and `--padding-pixels`, pass them to `unwrap_mesh`, and change the description to "Create deterministic cube UVs for a cleaned STFR 2DGS mesh."

- [ ] **Step 5: Run focused tests and verify the green state**

Run:

```bash
uv run pytest -q production/tests/test_unwrap_2dgs_uv.py
```

Expected: all unwrap tests pass; the report has method `cube`, six chart counts, atlas size 1024, padding two, and unchanged face count.

- [ ] **Step 6: Commit the cube atlas core**

```bash
git add production/unwrap_2dgs_uv.py production/tests/test_unwrap_2dgs_uv.py
git commit --only -m "feat: replace xatlas with deterministic cube UVs" -- production/unwrap_2dgs_uv.py production/tests/test_unwrap_2dgs_uv.py
```

---

### Task 2: Texture Configuration and Resume Fingerprint

**Files:**
- Modify: `production/texture_stage.py:68-163`
- Modify: `production/run_video_to_glb.py:45-275`
- Modify: `production/tests/test_texture_stage.py:1-48`
- Modify: `production/tests/test_run_video_to_glb.py:1-85`

**Interfaces:**
- Consumes: `unwrap_mesh(..., atlas_size, padding_pixels)` from Task 1 and the cleaned `face_geometry.ply`.
- Produces: `TextureConfig.uv_method`, `TextureConfig.atlas_size`, `TextureConfig.uv_padding_pixels`, `TextureConfig.uv_options()`, and `build_texture_resume_config(config)` with `source_mesh_sha256`.

- [ ] **Step 1: Write failing configuration and fingerprint tests**

Append:

```python
# production/tests/test_texture_stage.py
import pytest

from production.texture_stage import TextureConfig


def test_texture_config_exposes_production_cube_uv_options(tmp_path: Path) -> None:
    mesh = tmp_path / "face.ply"
    mesh.write_bytes(b"ply")
    config = TextureConfig(tmp_path, tmp_path, mesh, tmp_path / "artifacts")

    config.validate()

    assert config.uv_method == "cube"
    assert config.uv_options() == {"atlas_size": 1024, "padding_pixels": 2}


def test_texture_config_rejects_padding_that_consumes_a_tile(tmp_path: Path) -> None:
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
```

```python
# production/tests/test_run_video_to_glb.py
from production.run_video_to_glb import build_texture_resume_config
from production.texture_stage import TextureConfig


def test_texture_resume_fingerprint_changes_with_same_size_mesh_content(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "face.ply"
    mesh.write_bytes(b"aaaa")
    config = TextureConfig(tmp_path, tmp_path, mesh, tmp_path / "artifacts")
    first = build_texture_resume_config(config)

    mesh.write_bytes(b"bbbb")
    second = build_texture_resume_config(config)

    assert first["uv_method"] == second["uv_method"] == "cube"
    assert first["atlas_size"] == second["atlas_size"] == 1024
    assert first["uv_padding_pixels"] == second["uv_padding_pixels"] == 2
    assert first["source_mesh_sha256"] != second["source_mesh_sha256"]
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```bash
uv run pytest -q production/tests/test_texture_stage.py production/tests/test_run_video_to_glb.py
```

Expected: failures for the missing `TextureConfig` fields/method and missing `build_texture_resume_config`.

- [ ] **Step 3: Add and validate the production UV contract in `TextureConfig`**

Extend the dataclass and its validation:

```python
@dataclass(frozen=True)
class TextureConfig:
    code_root: Path
    source_root: Path
    source_mesh: Path
    output_root: Path
    python: str = "python"
    physical_gpu: int = 1
    iterations: int = 301
    uv_method: str = "cube"
    atlas_size: int = 1024
    uv_padding_pixels: int = 2

    def uv_options(self) -> dict[str, int]:
        return {
            "atlas_size": self.atlas_size,
            "padding_pixels": self.uv_padding_pixels,
        }

    def validate(self) -> None:
        if not self.source_mesh.is_file():
            raise FileNotFoundError(self.source_mesh)
        if self.physical_gpu < 0:
            raise ValueError("Physical GPU index cannot be negative")
        if self.iterations < 1:
            raise ValueError("Texture iterations must be positive")
        if self.uv_method != "cube":
            raise ValueError("Production texture stage requires cube UVs")
        if self.atlas_size < 1 or self.uv_padding_pixels < 0:
            raise ValueError("Atlas size must be positive and padding non-negative")
        pad = self.uv_padding_pixels / self.atlas_size
        if 2.0 * pad >= 1.0 / 3.0:
            raise ValueError("Padding leaves no usable cube-atlas tile area")
```

Change the unwrap call to:

```python
uv_report = unwrap_mesh(
    config.source_mesh,
    output_obj,
    "face",
    output_texture.name,
    **config.uv_options(),
)
```

- [ ] **Step 4: Add the source-mesh hash to texture resume state**

Add next to `file_sha256`:

```python
def build_texture_resume_config(config: TextureConfig) -> dict:
    payload = asdict(config)
    payload["source_mesh_sha256"] = file_sha256(config.source_mesh)
    return payload
```

Build `TextureConfig` before the texture stage as today, then replace
`asdict(texture_config)` in `execute_stage("texture", ...)` with
`build_texture_resume_config(texture_config)`. Keep the hash out of subprocess
arguments; it exists only to invalidate stale resume state.

- [ ] **Step 5: Run focused and state tests**

Run:

```bash
uv run pytest -q production/tests/test_texture_stage.py production/tests/test_run_video_to_glb.py production/tests/test_state.py
```

Expected: all tests pass, including a same-byte-length mesh content change invalidating the texture resume configuration.

- [ ] **Step 6: Commit texture integration**

```bash
git add production/texture_stage.py production/run_video_to_glb.py production/tests/test_texture_stage.py production/tests/test_run_video_to_glb.py
git commit --only -m "feat: bind cube UV settings to texture resume state" -- production/texture_stage.py production/run_video_to_glb.py production/tests/test_texture_stage.py production/tests/test_run_video_to_glb.py
```

---

### Task 3: Independent Publication Validation and Direct Export CLI

**Files:**
- Modify: `production/validate_asset.py:1-92`
- Modify: `production/export_glb.py:55-130`
- Modify: `production/run_video_to_glb.py:230-275`
- Modify: `production/tests/test_validate_asset.py:1-65`
- Modify: `production/tests/test_export_glb.py`

**Interfaces:**
- Consumes: `validate_cube_uv_padding` from Task 1 and `TextureConfig` from Task 2.
- Produces: `validate_asset(..., uv_padding_pixels=2)` with cube-layout validation and `load_output_transform(report_path)` using `source_to_output_row_matrix`.

- [ ] **Step 1: Write failing final-validation tests**

Change existing `validate_asset` test calls that use a 4-by-4 synthetic texture to pass `uv_padding_pixels=0`. Extend the fixture so its texture size can be selected:

```python
def build_asset(
    tmp_path: Path,
    uniform: bool = False,
    texture_size: int = 4,
):
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
    pixels = np.full((texture_size, texture_size, 3), 128, dtype=np.uint8)
    if not uniform:
        split = max(texture_size // 2, 1)
        pixels[:split, :split] = (220, 80, 40)
    Image.fromarray(pixels).save(texture)
    exported = export_canonical_asset(
        obj, mtl, texture, np.eye(4), tmp_path / "output"
    )
    return source_path, exported
```

Then add:

```python
def test_validate_asset_rejects_uv_on_cube_tile_edge(tmp_path: Path) -> None:
    source, exported = build_asset(tmp_path, texture_size=12)

    with pytest.raises(ValueError, match="tile padding"):
        validate_asset(
            source,
            Path(exported["obj"]),
            Path(exported["texture"]),
            Path(exported["glb"]),
            expected_texture_size=(12, 12),
            uv_padding_pixels=1,
        )
```

The fixture has UVs on `[0, 1]` boundaries, so it is valid with zero padding and intentionally violates one-pixel padding in a valid 12-by-12 atlas.

Add a transform-report test:

```python
# production/tests/test_export_glb.py
import json

from production.export_glb import load_output_transform


def test_load_output_transform_uses_direct_cleanup_report_key(tmp_path: Path) -> None:
    report = tmp_path / "geometry-report.json"
    matrix = np.eye(4).tolist()
    report.write_text(
        json.dumps({"source_to_output_row_matrix": matrix}),
        encoding="utf-8",
    )

    np.testing.assert_array_equal(load_output_transform(report), np.eye(4))
```

- [ ] **Step 2: Run validation tests and verify the red state**

Run:

```bash
uv run pytest -q production/tests/test_validate_asset.py production/tests/test_export_glb.py
```

Expected: failures because `validate_asset` lacks `uv_padding_pixels` and `load_output_transform` is undefined.

- [ ] **Step 3: Enforce cube padding in the final asset validator**

Import `validate_cube_uv_padding`, extend the signature, and call it after checking the UV array shape:

```python
def validate_asset(
    source_path: Path,
    obj_path: Path,
    texture_path: Path,
    glb_path: Path,
    expected_texture_size: tuple[int, int] = (1024, 1024),
    uv_padding_pixels: int = 2,
) -> dict:
    source = load_triangle_mesh(source_path)
    textured = load_triangle_mesh(obj_path)
    texture = Image.open(texture_path).convert("RGB")
    pixels = np.asarray(texture, dtype=np.float32)
    validate_cube_uv_padding(
        textured.visual.uv,
        atlas_size=expected_texture_size[0],
        padding_pixels=uv_padding_pixels,
    )
```

Reject non-square expected textures because cube UV generation accepts one pixel
dimension:

```python
if expected_texture_size[0] != expected_texture_size[1]:
    raise ValueError("Cube UV validation requires a square texture")
```

Add `uv_method: "cube"` and `uv_padding_pixels` to the returned quality report. Change stale "Canonical export changed geometry bounds" text to "UV export changed geometry bounds" and change the CLI description from canonical face to direct textured face.

- [ ] **Step 4: Pass the production atlas contract into publication validation**

In `run_video_to_glb.py`, derive validation state and the call from `texture_config`:

```python
validation_config = {
    "texture_size": [texture_config.atlas_size, texture_config.atlas_size],
    "uv_method": texture_config.uv_method,
    "uv_padding_pixels": texture_config.uv_padding_pixels,
}

quality = validate_asset(
    clean_geometry,
    Path(export_report["obj"]),
    Path(export_report["texture"]),
    Path(export_report["glb"]),
    expected_texture_size=(texture_config.atlas_size,) * 2,
    uv_padding_pixels=texture_config.uv_padding_pixels,
)
```

This makes a changed padding contract invalidate both texture and publication resume records.

- [ ] **Step 5: Repair the standalone export CLI's direct transform key**

Add and use:

```python
def load_output_transform(report_path: Path) -> np.ndarray:
    cleanup = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return np.asarray(cleanup["source_to_output_row_matrix"], dtype=np.float64)
```

Change `main()` to pass `load_output_transform(args.clean_report)` and change CLI/user-facing messages from "canonical" to "output" or "direct". Keep `export_canonical_asset` as a compatibility function name because the orchestrator and existing callers already import it; its matrix is the identity in the direct production route.

- [ ] **Step 6: Run export, validation, and orchestrator tests**

Run:

```bash
uv run pytest -q production/tests/test_validate_asset.py production/tests/test_export_glb.py production/tests/test_run_video_to_glb.py
```

Expected: all tests pass; zero-padding generic fixtures remain valid, positive-padding boundary UVs fail, and the direct cleanup matrix loads correctly.

- [ ] **Step 7: Commit the publication gate**

```bash
git add production/validate_asset.py production/export_glb.py production/run_video_to_glb.py production/tests/test_validate_asset.py production/tests/test_export_glb.py
git commit --only -m "fix: validate padded cube UV assets before publish" -- production/validate_asset.py production/export_glb.py production/run_video_to_glb.py production/tests/test_validate_asset.py production/tests/test_export_glb.py
```

---

### Task 4: Remove xatlas and Align Production Documentation

**Files:**
- Modify: `pyproject.toml:5-43`
- Modify: `uv.lock`
- Modify: `production/tests/test_pyproject.py:12-71`
- Modify: `README.md:3-27`
- Modify: `ENV.md:1-24`
- Modify: `AGENTS.md:1-140`

**Interfaces:**
- Consumes: cube UV implementation from Tasks 1-3.
- Produces: a frozen Linux dependency graph with no xatlas package and operator instructions that name cube UV as the production method.

- [ ] **Step 1: Change the manifest test to require no xatlas**

Replace:

```python
assert "xatlas==0.0.11" in dependencies
```

with:

```python
assert not any(dependency.startswith("xatlas") for dependency in dependencies)
```

Extend the operator-document test:

```python
assert "deterministic cube UV" in instructions
assert "xatlas is not required" in instructions
```

- [ ] **Step 2: Run the manifest test and verify the red state**

Run:

```bash
uv run pytest -q production/tests/test_pyproject.py
```

Expected: failure because `pyproject.toml` still contains `xatlas==0.0.11` and `AGENTS.md` does not yet state the new contract.

- [ ] **Step 3: Remove xatlas from the production manifest and update docs**

Delete only the `"xatlas==0.0.11",` dependency from `pyproject.toml`.

Update the production paragraph in `README.md` to say:

```text
20 volume-preserving Laplacian smoothing passes, deterministic cube UVs with
two-pixel chart padding, and 301 texture iterations.
```

Add to the recommended uv section in `ENV.md`:

```text
The production environment does not install xatlas; UVs are generated by the
repository's deterministic NumPy cube projection. The legacy upstream manual
installation section remains historical and may still mention xatlas.
```

Change the production asset paragraph in `AGENTS.md` to name deterministic cube UV projection and add the exact sentence `xatlas is not required by the production entry point.` Keep the legacy upstream research instructions intact.

- [ ] **Step 4: Regenerate the Linux lock without dependency upgrades**

On the verified Ubuntu server in `/home/ii/STFR-production`:

```bash
export CUDA_HOME=/usr
export PATH="/usr/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
uv lock
uv sync --frozen
uv tree | grep -i xatlas
```

The final command is expected to print nothing and return grep status 1. Copy the regenerated `/home/ii/STFR-production/uv.lock` back to the same repository path locally. Inspect `git diff -- uv.lock` and confirm the only removed package is xatlas and no pinned version or Git SHA changed.

- [ ] **Step 5: Run manifest and full local unit tests**

Run:

```bash
uv run pytest -q production/tests
git diff --check
```

Expected: all production tests pass, `git diff --check` reports no whitespace errors, and `rg -n "import xatlas|xatlas\.parametrize" production pyproject.toml` returns no matches.

- [ ] **Step 6: Commit dependency and documentation changes**

```bash
git add pyproject.toml uv.lock production/tests/test_pyproject.py README.md ENV.md AGENTS.md
git commit --only -m "build: remove xatlas from production environment" -- pyproject.toml uv.lock production/tests/test_pyproject.py README.md ENV.md AGENTS.md
```

---

### Task 5: Full-Mesh Server Benchmark and End-to-End Release Gate

**Files:**
- Verify: all files changed in Tasks 1-4
- Runtime fixture: `/home/ii/STFR/workspace/benchmark_fast5000_v1/2dgs_recon.obj`
- Runtime output: `/home/ii/STFR-production/benchmarks/cube-uv-full/`
- Video fixture: `D:\BPR_PRODUCTION\MetologyExamples\ljf.MOV`

**Interfaces:**
- Consumes: committed `pipeline` branch, server `.venv`, CUDA-enabled COLMAP, matting ONNX weight, second RTX 3090, and the full real mesh.
- Produces: measured cube-UV runtime/memory, validated full-mesh OBJ/MTL, and a final video-derived `face.glb` plus `result.json`.

- [ ] **Step 1: Run repository checks locally and tests on the server**

```bash
git diff --check
git status --short
git log --oneline -5
uv run pytest -q production/tests
```

Run the first three commands in the Windows checkout and pytest in `/home/ii/STFR-production`. Expected: every production test passes; no unexplained working-tree changes remain; the last commits correspond to Tasks 0-4 and the approved design/plan.

- [ ] **Step 2: Publish the committed `pipeline` branch through the server Git identity**

From PowerShell in the local repository, create and upload a branch bundle:

```powershell
New-Item -ItemType Directory -Force -Path 'C:\Users\Ilya\Documents\Codex\2026-09-07\dj\work'
git bundle create 'C:\Users\Ilya\Documents\Codex\2026-09-07\dj\work\stfr-cube-uv.bundle' pipeline
scp -P 2221 -i 'C:\Users\Ilya\.ssh\id_rsa_r_rsasa' 'C:\Users\Ilya\Documents\Codex\2026-09-07\dj\work\stfr-cube-uv.bundle' ii@95.79.44.129:/tmp/stfr-cube-uv.bundle
```

Publish it from a fresh temporary server clone:

```bash
publish_dir="$(mktemp -d /tmp/stfr-cube-publish.XXXXXX)"
git clone git@github.com:FiksII/ST/STFR.git "$publish_dir"
git -C "$publish_dir" fetch /tmp/stfr-cube-uv.bundle pipeline:refs/remotes/bundle/pipeline
git -C "$publish_dir" checkout pipeline
git -C "$publish_dir" merge --ff-only refs/remotes/bundle/pipeline
git -C "$publish_dir" push origin pipeline
```

Expected: GitHub's `pipeline` branch advances by fast-forward to the local commit.

- [ ] **Step 3: Deploy the exact tested file set to the production clone**

Before overwriting known files, save the server diff and status:

```bash
cd /home/ii/STFR-production
git status --short
git diff > /home/ii/STFR-production-before-cube-uv.patch
```

Stop if the status contains a file outside the known production change set. From the local repository, archive and upload the exact implementation files:

```powershell
tar -cf 'C:\Users\Ilya\Documents\Codex\2026-09-07\dj\work\stfr-cube-uv-files.tar' AGENTS.md ENV.md README.md pyproject.toml uv.lock production/clean_face_mesh.py production/export_glb.py production/reconstruction_stage.py production/run_video_to_glb.py production/texture_stage.py production/unwrap_2dgs_uv.py production/validate_asset.py production/tests/test_clean_face_mesh.py production/tests/test_export_glb.py production/tests/test_pyproject.py production/tests/test_reconstruction_stage.py production/tests/test_run_video_to_glb.py production/tests/test_state.py production/tests/test_texture_stage.py production/tests/test_unwrap_2dgs_uv.py production/tests/test_validate_asset.py
scp -P 2221 -i 'C:\Users\Ilya\.ssh\id_rsa_r_rsasa' 'C:\Users\Ilya\Documents\Codex\2026-09-07\dj\work\stfr-cube-uv-files.tar' ii@95.79.44.129:/tmp/stfr-cube-uv-files.tar
```

Extract and verify on the server:

```bash
cd /home/ii/STFR-production
tar -xf /tmp/stfr-cube-uv-files.tar -C /home/ii/STFR-production
export CUDA_HOME=/usr
export PATH="/usr/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
uv sync --frozen
uv run pytest -q production/tests
```

Expected: the GitHub `pipeline` tip matches the local tip, sync installs no xatlas package, and all server production tests pass.

- [ ] **Step 4: Generate cube UVs for the full cleaned integration mesh**

```bash
mkdir -p /home/ii/STFR-production/benchmarks/cube-uv-full
cd /home/ii/STFR-production
uv run python -m production.clean_face_mesh \
  --source /home/ii/STFR/workspace/benchmark_fast5000_v1/2dgs_recon.obj \
  --output /home/ii/STFR-production/benchmarks/cube-uv-full/face_geometry.ply \
  --report /home/ii/STFR-production/benchmarks/cube-uv-full/geometry-report.json \
  --smooth-iterations 20
/usr/bin/time -v uv run python -m production.unwrap_2dgs_uv \
  --input /home/ii/STFR-production/benchmarks/cube-uv-full/face_geometry.ply \
  --output /home/ii/STFR-production/benchmarks/cube-uv-full/face_uv_source.obj \
  --texture uv.png \
  --atlas-size 1024 \
  --padding-pixels 2 \
  --report /home/ii/STFR-production/benchmarks/cube-uv-full/texture-prepare-report.json
```

Expected: unwrap wall time is below 300 seconds, face count equals the cleaned PLY count, all six chart counts are nonzero for the real head mesh, UVs pass two-pixel padding validation, and peak resident memory stays below 15 GiB.

- [ ] **Step 5: Reload and compare the full OBJ without processing**

Run this exact check from the production root:

```bash
uv run python -c "from pathlib import Path; import numpy as np; from production.unwrap_2dgs_uv import load_triangle_mesh, validate_cube_uv_padding; a=load_triangle_mesh(Path('/home/ii/STFR-production/benchmarks/cube-uv-full/face_geometry.ply')); b=load_triangle_mesh(Path('/home/ii/STFR-production/benchmarks/cube-uv-full/face_uv_source.obj')); assert len(a.faces)==len(b.faces); assert np.allclose(a.bounds,b.bounds,atol=1e-6); validate_cube_uv_padding(b.visual.uv,1024,2); print({'source_faces':len(a.faces),'output_faces':len(b.faces),'output_vertices':len(b.vertices),'bounds_ok':True,'uv_ok':True})"
```

Expected: the command prints equal face counts with `bounds_ok` and `uv_ok` true.

- [ ] **Step 6: Run the complete production video-to-GLB pipeline on GPU 1**

Create `/home/ii/STFR-production/data`, copy `D:\BPR_PRODUCTION\MetologyExamples\ljf.MOV` there as `ljf.MOV`, and start the long run with durable logging:

```bash
cd /home/ii/STFR-production
mkdir -p logs data
nohup env CUDA_HOME=/usr CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST=8.6 \
  .venv/bin/python -m production.run_video_to_glb \
  --video /home/ii/STFR-production/data/ljf.MOV \
  --job-root /home/ii/STFR-production/workspace/production-cube-ljf \
  --output /home/ii/STFR-production/workspace/production-cube-ljf/result/face.glb \
  --physical-gpu 1 \
  > /home/ii/STFR-production/logs/production-cube-ljf.log 2>&1 &
```

Poll the PID and log without starting a second run. Expected: all five stages complete, the final command exits zero, and `result/face.glb` plus `result/result.json` exist and are non-empty.

- [ ] **Step 7: Inspect the final manifest and GLB**

```bash
cd /home/ii/STFR-production
.venv/bin/python -c "import json,trimesh; from pathlib import Path; root=Path('/home/ii/STFR-production/workspace/production-cube-ljf/result'); manifest=json.loads((root/'result.json').read_text()); scene=trimesh.load(root/'face.glb',process=False,force='scene'); print({'status':manifest['status'],'sha256':manifest['sha256'],'bytes':manifest['bytes'],'faces':sum(len(m.faces) for m in scene.geometry.values()),'textured':any(m.visual.kind=='texture' for m in scene.geometry.values()),'bounds':scene.bounds.tolist()})"
```

Expected: status `complete`, a 64-character SHA-256, nonzero bytes/faces, at least one textured GLB geometry, and finite bounds.

- [ ] **Step 8: Record release evidence**

Report the exact branch commit, production test count, cube unwrap wall time, maximum resident memory, source/output face counts, seam-split output vertex count, GLB byte size, GLB face count, texture presence, and final manifest SHA-256. Do not claim the pipeline complete if the full-mesh benchmark used a decimated input or if any validation assertion failed.
