# MediaPipe Face Crop and xatlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Crop the visible facial surface from an existing 2DGS reconstruction with MediaPipe and COLMAP cameras, then generate a non-overlapping xatlas texture and publish a validated GLB without Faceform Wrap.

**Architecture:** A new face-crop stage detects a padded MediaPipe face oval in each selected frame and intersects those masks with PyTorch3D face-index rasters from the corresponding cameras. It exports the union of visible 2DGS triangles; the existing cleanup stage sanitizes and lightly smooths that crop, then the texture stage uses xatlas instead of cube projection.

**Tech Stack:** Python 3.10, NumPy, OpenCV, MediaPipe 0.10.11, PyTorch 2.3.1, PyTorch3D 0.7.8, trimesh 5.1.0, xatlas 0.0.11, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-09-mediapipe-face-crop-xatlas-design.md`

## Global Constraints

- Keep the input/output contract as one video to one validated GLB.
- Keep 30,000 2DGS training iterations and allow resume after reconstruction.
- Use physical GPU index 1 by default and logical `cuda:0` inside child processes.
- Do not require or invoke Faceform Wrap.
- MediaPipe identifies the image-space face only; it does not replace or deform 2DGS geometry.
- Fail instead of publishing the full bust when fewer than three views detect a face or the crop is implausibly small.
- Produce a 1024x1024 UV texture with 301 optimization iterations.

---

### Task 1: Pure Face Selection Logic

**Files:**
- Create: `production/face_crop_stage.py`
- Create: `production/tests/test_face_crop_stage.py`

**Interfaces:**
- Produces: `FaceCropConfig`, `padded_face_oval_mask(landmarks, image_size, scale)`, `aggregate_visible_faces(face_rasters, oval_masks, face_count)`, and `expand_face_selection(selected, adjacency, rings)`.
- Consumes: NumPy arrays only; MediaPipe, CUDA, and file I/O are excluded from these helpers.

- [ ] **Step 1: Write failing tests for the face oval and visible-face aggregation**

```python
def test_padded_face_oval_mask_expands_about_landmark_center():
    landmarks = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])
    mask = padded_face_oval_mask(landmarks, (12, 12), scale=1.5)
    assert mask[5, 5]
    assert mask[3, 5]

def test_aggregate_visible_faces_ignores_background_and_pixels_outside_oval():
    raster = np.array([[-1, 0], [1, 2]])
    oval = np.array([[False, True], [True, False]])
    selected = aggregate_visible_faces([raster], [oval], face_count=3)
    assert selected.tolist() == [True, True, False]
```

- [ ] **Step 2: Run the new test module and verify import failures**

Run: `.venv/bin/python -m pytest production/tests/test_face_crop_stage.py -q`
Expected: FAIL because `production.face_crop_stage` does not exist.

- [ ] **Step 3: Implement the pure helpers and validated configuration**

```python
@dataclass(frozen=True)
class FaceCropConfig:
    oval_scale: float = 1.0
    adjacency_rings: int = 0
    minimum_detected_frames: int = 3
    minimum_selected_faces: int = 10_000

def aggregate_visible_faces(rasters, masks, face_count):
    selected = np.zeros(face_count, dtype=bool)
    for raster, mask in zip(rasters, masks, strict=True):
        ids = np.asarray(raster)[np.asarray(mask, dtype=bool)]
        ids = ids[(ids >= 0) & (ids < face_count)]
        selected[np.unique(ids)] = True
    return selected
```

Implement oval scaling around its centroid with `cv2.fillPoly`. Expand selected faces ring-by-ring through the `[N, 2]` face-adjacency array.

- [ ] **Step 4: Run the new unit tests**

Run: `.venv/bin/python -m pytest production/tests/test_face_crop_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the pure selection logic**

```bash
git add production/face_crop_stage.py production/tests/test_face_crop_stage.py
git commit -m "feat: add MediaPipe face selection logic"
```

### Task 2: MediaPipe and Camera-Matched Rasterization

**Files:**
- Modify: `production/face_crop_stage.py`
- Modify: `production/tests/test_face_crop_stage.py`

**Interfaces:**
- Produces: `crop_face_mesh(source_mesh, selected_frames_root, transforms_path, output_path, config, device) -> dict` and a CLI with `--source`, `--frames`, `--transforms`, `--output`, `--report`, and `--device`.
- Consumes: Task 1 helpers, `texture.mesh_renderer.MeshRenderer`, MediaPipe Face Mesh, and COLMAP camera matrices from `transforms.json`.

- [ ] **Step 1: Add failing orchestration tests with injected detector and rasterizer**

```python
def test_crop_requires_three_detected_frames(tmp_path):
    with pytest.raises(ValueError, match="at least 3"):
        crop_face_mesh(
            source_path=mesh_path,
            selected_frames_root=frames_root,
            transforms_path=transforms_path,
            output_path=tmp_path / "face.ply",
            config=FaceCropConfig(minimum_detected_frames=3),
            device="cuda:0",
            detector=lambda image: None,
            rasterizer=fake_rasterizer,
        )

def test_crop_exports_only_union_of_visible_face_ids(tmp_path):
    report = crop_face_mesh(
        source_path=mesh_path,
        selected_frames_root=frames_root,
        transforms_path=transforms_path,
        output_path=tmp_path / "face.ply",
        config=FaceCropConfig(minimum_selected_faces=1),
        device="cuda:0",
        detector=fake_detector,
        rasterizer=fake_rasterizer,
    )
    result = trimesh.load_mesh(report["output"], process=False)
    assert report["detected_frames"] == 3
    assert len(result.faces) == report["output_faces"]
```

- [ ] **Step 2: Run the orchestration tests and verify missing-interface failures**

Run: `.venv/bin/python -m pytest production/tests/test_face_crop_stage.py -q`
Expected: FAIL because `crop_face_mesh` and its injection points are missing.

- [ ] **Step 3: Implement detection, rasterization, export, and reporting**

Use `mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)` and the ordered MediaPipe oval indices. Load one mesh tensor on logical `cuda:0`, rasterize one frame at a time with its inverse camera transform and normalized intrinsics, collect `pix_to_face`, expand two adjacency rings, export a trimesh submesh, and record detected/missed frames plus source/selected/output counts.

- [ ] **Step 4: Run face-crop tests and static CLI help**

Run: `.venv/bin/python -m pytest production/tests/test_face_crop_stage.py -q`
Expected: PASS.

Run: `.venv/bin/python -m production.face_crop_stage --help`
Expected: exit 0 and list all required CLI arguments.

- [ ] **Step 5: Commit the runtime crop stage**

```bash
git add production/face_crop_stage.py production/tests/test_face_crop_stage.py
git commit -m "feat: crop visible 2DGS face from MediaPipe views"
```

### Task 3: Bounded Face-Mesh Cleanup

**Files:**
- Modify: `production/clean_face_mesh.py`
- Modify: `production/tests/test_clean_face_mesh.py`

**Interfaces:**
- Produces: `CleanupConfig(smooth_iterations=3)` using Taubin smoothing and the existing `clean_face_mesh(source_path, output_path, config) -> dict` signature.
- Consumes: the already cropped mesh from Task 2.

- [ ] **Step 1: Change tests to require three Taubin iterations and bounded displacement**

```python
def test_cleanup_defaults_to_three_taubin_passes():
    assert CleanupConfig().smooth_iterations == 3

def test_cleanup_uses_taubin_smoothing(monkeypatch, tmp_path):
    monkeypatch.setattr("production.clean_face_mesh.filter_taubin", capture_filter)
    clean_face_mesh(source, output, CleanupConfig())
    assert captured == {"lamb": 0.2, "nu": 0.21, "iterations": 3}
```

- [ ] **Step 2: Run cleanup tests and verify they fail against Laplacian smoothing**

Run: `.venv/bin/python -m pytest production/tests/test_clean_face_mesh.py -q`
Expected: FAIL because the default is 20 and `filter_taubin` is not used.

- [ ] **Step 3: Replace Laplacian smoothing with three Taubin passes**

Import `filter_taubin` from `trimesh.smoothing`, call it with `lamb=0.2`, `nu=0.21`, and preserve the displacement and roughness report.

- [ ] **Step 4: Run cleanup tests**

Run: `.venv/bin/python -m pytest production/tests/test_clean_face_mesh.py -q`
Expected: PASS.

- [ ] **Step 5: Commit cleanup correction**

```bash
git add production/clean_face_mesh.py production/tests/test_clean_face_mesh.py
git commit -m "fix: preserve cropped 2DGS face detail"
```

### Task 4: Restore Production xatlas UVs

**Files:**
- Modify: `production/unwrap_2dgs_uv.py`
- Modify: `production/texture_stage.py`
- Modify: `production/validate_asset.py`
- Modify: `production/tests/test_unwrap_2dgs_uv.py`
- Modify: `production/tests/test_texture_stage.py`
- Modify: `production/tests/test_validate_asset.py`
- Modify: `production/tests/test_pyproject.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `_parametrize_with_xatlas(vertices: np.ndarray, faces: np.ndarray)`, `unwrap_mesh(input_path: Path, output_obj: Path, material_name: str = "face", texture_name: str = "uv.png") -> dict`, and `validate_asset(source_path, obj_path, texture_path, glb_path, expected_texture_size, uv_method="xatlas", source_to_output_matrix=None) -> dict` validation for generic finite `[0,1]` UVs without cube-tile assumptions.
- Consumes: `xatlas==0.0.11` and the cropped, cleaned source-space mesh.

- [ ] **Step 1: Run the already-written xatlas contract tests and record red failures**

Run: `.venv/bin/python -m pytest production/tests/test_unwrap_2dgs_uv.py production/tests/test_texture_stage.py production/tests/test_validate_asset.py production/tests/test_pyproject.py production/tests/test_run_video_to_glb.py -q`
Expected: FAIL on cube method, missing xatlas dependency, cube validation, and resume metadata.

- [ ] **Step 2: Replace production cube unwrap with xatlas parametrization**

```python
def _parametrize_with_xatlas(vertices, faces):
    import xatlas
    return xatlas.parametrize(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )
```

Use returned vertex mapping for positions and normals, returned indices for faces, and returned UVs for OBJ output. Check finite coordinates and `[0,1]` bounds before writing.

- [ ] **Step 3: Update texture configuration, validation, and fingerprints**

Set `TextureConfig.uv_method` to `"xatlas"`, remove cube padding from production configuration and validation, and include `uv_method` plus `atlas_size` in validation resume state.

- [ ] **Step 4: Pin and lock xatlas, then update installation docs**

Add `xatlas==0.0.11` to `pyproject.toml`. Run `uv lock` on Linux, document that xatlas is installed by `uv sync --frozen`, and remove claims that production uses cube UVs.

- [ ] **Step 5: Run xatlas contract tests**

Run: `.venv/bin/python -m pytest production/tests/test_unwrap_2dgs_uv.py production/tests/test_texture_stage.py production/tests/test_validate_asset.py production/tests/test_pyproject.py production/tests/test_run_video_to_glb.py -q`
Expected: PASS.

- [ ] **Step 6: Commit xatlas restoration**

```bash
git add production pyproject.toml uv.lock AGENTS.md README.md
git commit -m "fix: restore non-overlapping xatlas texture UVs"
```

### Task 5: Integrate Face Crop into the Resumable Pipeline

**Files:**
- Modify: `production/run_video_to_glb.py`
- Modify: `production/tests/test_run_video_to_glb.py`
- Modify: `README.md`

**Interfaces:**
- Produces: pipeline stages `reconstruction`, `face_crop`, `clean_geometry`, `texture`, `asset_export`, `validate_publish`; crop resume fingerprint includes source mesh SHA-256, transforms SHA-256, selected image names and hashes, crop config, and device.
- Consumes: Task 2 `crop_face_mesh(source_path, selected_frames_root, transforms_path, output_path, config, device)`, Task 3 cleanup, and Task 4 xatlas texture stage.

- [ ] **Step 1: Write failing stage-order and crop-fingerprint tests**

```python
def test_pipeline_crops_before_cleaning_and_texturing():
    assert PIPELINE_STAGES[:4] == (
        "reconstruction", "face_crop", "clean_geometry", "texture"
    )

def test_crop_resume_fingerprint_changes_when_selected_frame_changes(tmp_path):
    first = build_face_crop_resume_config(config)
    frame.write_bytes(b"changed")
    second = build_face_crop_resume_config(config)
    assert first != second
```

- [ ] **Step 2: Run pipeline tests and verify stage/fingerprint failures**

Run: `.venv/bin/python -m pytest production/tests/test_run_video_to_glb.py -q`
Expected: FAIL because the face-crop stage and fingerprint builder are absent.

- [ ] **Step 3: Wire crop output into cleanup and xatlas texture stages**

Publish `artifacts/face_crop.ply` and `artifacts/face-crop-report.json`, clean it to `artifacts/face_geometry.ply`, and preserve the existing SHA-based invalidation through texture, export, validation, and publication. Add CLI options `--face-oval-scale` and `--face-adjacency-rings` with the production defaults from `FaceCropConfig`.

- [ ] **Step 4: Run the complete unit suite**

Run: `.venv/bin/python -m pytest production/tests -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit pipeline integration**

```bash
git add production/run_video_to_glb.py production/tests/test_run_video_to_glb.py README.md
git commit -m "feat: add resumable MediaPipe face crop stage"
```

### Task 6: Server Verification on Existing 30k Reconstruction

**Files:**
- Runtime outputs only under `/home/ii/STFR-production/workspace/production-cube-ljf-v2/`
- User deliverables under `C:/Users/Ilya/Documents/Codex/2026-09-07/dj/outputs/`

**Interfaces:**
- Produces: a new candidate GLB, UV image, crop report, validation report, and front/side renders.
- Consumes: the existing `workspace/2dgs_recon.obj`, transforms, selected frames, and GPU 1.

- [ ] **Step 1: Sync the implementation to the non-clean server checkout without deleting runtime artifacts**

Use a targeted archive or SCP for changed source files, then run `uv sync --frozen` in `/home/ii/STFR-production` with the documented CUDA environment.

- [ ] **Step 2: Run the full production test suite on the server**

Run: `.venv/bin/python -m pytest production/tests -q`
Expected: all tests PASS and `import xatlas` succeeds.

- [ ] **Step 3: Invalidate only crop-and-later state and resume the existing job**

Preserve reconstruction files. Run the pipeline with `--resume --physical-gpu 1`; verify emitted events skip `reconstruction` and execute `face_crop`, `clean_geometry`, `texture`, `asset_export`, and `validate_publish`.

- [ ] **Step 4: Validate quantitative output**

Confirm the crop has at least 10,000 faces, fewer faces than the original 2DGS bust, at least three detected frames, xatlas UVs within `[0,1]`, one 1024x1024 non-uniform texture, matching OBJ/GLB triangle counts, and a reloadable textured GLB.

- [ ] **Step 5: Render and compare front/side views**

Render the candidate with the same camera/framing used for `face_30000_uv.glb`. Confirm the shoulders are absent, facial features are sharper than the cube-UV output, and no gross UV seams or flipped orientation are visible.

- [ ] **Step 6: Copy verified deliverables and report exact paths and hashes**

Copy the candidate GLB, UV image, crop report, and front/side PNGs to the task output directory. Report file sizes, SHA-256, elapsed post-reconstruction time, tests, and any remaining visible limitations.
