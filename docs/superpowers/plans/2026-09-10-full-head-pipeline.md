# Full-Head Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MediaPipe face-oval crop with a neural multi-view crop that retains the observed head, hair, ears, and a short neck while producing the same validated video-to-GLB result.

**Architecture:** A pinned ResNet18 BiSeNet ONNX model produces 19-class face-parsing maps for the selected reconstruction views. MediaPipe landmarks anchor the correct component and bound the neck; camera-matched rasterization maps the resulting masks back to `2dgs_recon.obj`, while a separate face-anchor mesh stabilizes canonical orientation. Existing xatlas UV, texture optimization, GLB export, validation, and reconstruction resume behavior stay in place.

**Tech Stack:** Python 3.10, NumPy, OpenCV, MediaPipe, ONNX Runtime GPU, PyTorch3D, trimesh, scipy, xatlas, pytest

**Spec:** `docs/superpowers/specs/2026-09-10-full-head-pipeline-design.md`

## Global Constraints

- Work only on branch `full-head-pipeline`; leave `pipeline` unchanged.
- Preserve the public contract: one input video and one output GLB path.
- Include observed skin, facial features, hair, ears, and neck; exclude background, glasses, hats, earrings, necklaces, clothing, and shoulders.
- Limit retained neck pixels to 0.45 face heights below the MediaPipe chin.
- Do not synthesize an unobserved rear head surface.
- Reuse the saved 30,000-iteration reconstruction for acceptance testing.
- Run production inference on physical GPU 1, exposed inside the process as logical GPU 0.
- Keep the 1024 by 1024 xatlas texture path and existing validation gates.
- Do not commit model weights or runtime artifacts.

## File Structure

- Create `production/model_assets.py`: pinned model metadata, checksum verification, and atomic download.
- Create `production/download_models.py`: command-line installer for external model weights.
- Create `production/head_segmentation.py`: ONNX inference, MediaPipe anchors, semantic mask construction, and diagnostics.
- Create `production/head_crop_stage.py`: multi-view raster-to-mesh selection and crop export.
- Create `production/tests/test_model_assets.py`: model installer contract tests.
- Create `production/tests/test_head_segmentation.py`: semantic and geometric mask tests.
- Create `production/tests/test_head_crop_stage.py`: head crop and face-anchor mesh tests.
- Modify `production/clean_face_mesh.py`: canonicalize a head using a separate orientation mesh.
- Modify `production/tests/test_clean_face_mesh.py`: verify orientation-mesh behavior.
- Modify `production/run_video_to_glb.py`: replace `face_crop` with `head_crop` and update resume fingerprints.
- Modify `production/tests/test_run_video_to_glb.py`: verify CLI, stage order, model hash, and orientation input.
- Modify `.gitignore`, `README.md`, and `AGENTS.md`: external model installation and full-head production behavior.

---

### Task 1: Pinned Face-Parsing Model Asset

**Files:**
- Create: `production/model_assets.py`
- Create: `production/download_models.py`
- Create: `production/tests/test_model_assets.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `FACE_PARSING_MODEL: ModelAsset`
- Produces: `verify_model(path: Path, asset: ModelAsset = FACE_PARSING_MODEL) -> str`
- Produces: `download_model(destination: Path, asset: ModelAsset = FACE_PARSING_MODEL) -> dict`
- Consumes later: `FACE_PARSING_MODEL.sha256` and the verified local model path.

- [ ] **Step 1: Write failing model-asset tests**

```python
def test_face_parser_asset_is_pinned() -> None:
    assert FACE_PARSING_MODEL.url == (
        "https://github.com/yakhyo/face-parsing/releases/download/weights/"
        "resnet18.onnx"
    )
    assert FACE_PARSING_MODEL.sha256 == (
        "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
    )


def test_verify_model_rejects_wrong_content(tmp_path: Path) -> None:
    path = tmp_path / "resnet18.onnx"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_model(path)
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `python -m pytest -q production/tests/test_model_assets.py`

Expected: collection fails because `production.model_assets` does not exist.

- [ ] **Step 3: Implement the pinned asset and atomic downloader**

```python
@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    sha256: str


FACE_PARSING_MODEL = ModelAsset(
    filename="face-parsing-resnet18.onnx",
    url=(
        "https://github.com/yakhyo/face-parsing/releases/download/weights/"
        "resnet18.onnx"
    ),
    sha256="0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
)


def verify_model(path: Path, asset: ModelAsset = FACE_PARSING_MODEL) -> str:
    actual = file_sha256(Path(path))
    if actual != asset.sha256:
        raise ValueError(
            f"Model SHA-256 mismatch for {path}: expected {asset.sha256}, got {actual}"
        )
    return actual
```

`download_model` must stream into `.<filename>.tmp`, verify that file, and use
`os.replace` only after verification. `download_models.py` installs by default
to `models/face-parsing-resnet18.onnx` and prints a JSON report.

- [ ] **Step 4: Ignore external weights and run the focused tests**

Add `models/*.onnx` to `.gitignore`.

Run: `python -m pytest -q production/tests/test_model_assets.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the asset contract**

```bash
git add .gitignore production/model_assets.py production/download_models.py production/tests/test_model_assets.py
git commit -m "feat: add pinned face parsing model asset"
```

### Task 2: Semantic Full-Head Masks

**Files:**
- Create: `production/head_segmentation.py`
- Create: `production/tests/test_head_segmentation.py`

**Interfaces:**
- Produces: `HeadMaskConfig(neck_height_ratio=0.45, close_kernel_ratio=0.015, minimum_component_pixels=64)`
- Produces: `MediaPipeFaceAnchorDetector.__call__(image_path: Path) -> np.ndarray | None`
- Produces: `OnnxFaceParser.__call__(image_path: Path) -> np.ndarray`
- Produces: `build_head_mask(labels: np.ndarray, landmarks: np.ndarray, config: HeadMaskConfig) -> tuple[np.ndarray, dict]`
- Produces: `face_oval_mask(landmarks: np.ndarray, image_size: tuple[int, int]) -> np.ndarray`

- [ ] **Step 1: Write failing semantic-mask tests**

```python
def test_head_mask_keeps_head_and_short_neck_but_excludes_accessories() -> None:
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[3:8, 7:13] = 13       # hair
    labels[8:14, 6:14] = 1       # skin
    labels[9:12, 5:6] = 8        # ear
    labels[14:20, 8:12] = 17     # neck
    labels[8:10, 8:12] = 3       # glasses
    labels[18:20, :] = 18        # clothing
    landmarks = np.array([[6, 8], [14, 8], [14, 14], [6, 14]], dtype=float)

    mask, report = build_head_mask(labels, landmarks, HeadMaskConfig())

    assert mask[4, 10]
    assert mask[10, 5]
    assert not mask[9, 10]
    assert mask[15, 10]
    assert not mask[18, 10]
    assert report["neck_height_ratio"] == 0.45
```

Also test invalid label shapes, non-finite landmarks, component selection by
face-oval overlap, odd morphology kernel sizing, and exact output dimensions.

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `python -m pytest -q production/tests/test_head_segmentation.py`

Expected: collection fails because `production.head_segmentation` does not exist.

- [ ] **Step 3: Implement deterministic mask construction**

```python
HEAD_LABELS = frozenset({1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
NECK_LABEL = 17


def build_head_mask(labels, landmarks, config=HeadMaskConfig()):
    face_height = float(np.ptp(landmarks[:, 1]))
    chin_y = float(landmarks[:, 1].max())
    center_x = float(landmarks[:, 0].mean())
    face_width = float(np.ptp(landmarks[:, 0]))
    yy, xx = np.indices(labels.shape)
    head = np.isin(labels, tuple(HEAD_LABELS))
    neck = (
        (labels == NECK_LABEL)
        & (yy <= chin_y + config.neck_height_ratio * face_height)
        & (np.abs(xx - center_x) <= 0.55 * face_width)
    )
    candidate = close_mask(head | neck, face_height, config.close_kernel_ratio)
    anchor = face_oval_mask(landmarks, labels.shape)
    result = component_with_largest_anchor_overlap(candidate, anchor)
    return result, build_mask_report(labels, head, neck, result, config)
```

Implement `OnnxFaceParser` with ImageNet RGB normalization, NCHW float32 input
at 512 by 512, CUDA provider device 0 followed by CPU fallback, first-output
argmax, and nearest-neighbor restoration to source size. The constructor calls
`verify_model` before creating the inference session.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q production/tests/test_head_segmentation.py`

Expected: all tests pass without loading MediaPipe or ONNX Runtime because the
inference adapters are not instantiated by pure mask tests.

- [ ] **Step 5: Commit semantic masks**

```bash
git add production/head_segmentation.py production/tests/test_head_segmentation.py
git commit -m "feat: build neural full-head masks"
```

### Task 3: Multi-View Head Mesh Crop

**Files:**
- Create: `production/head_crop_stage.py`
- Create: `production/tests/test_head_crop_stage.py`
- Delete: `production/face_crop_stage.py`
- Delete: `production/tests/test_face_crop_stage.py`

**Interfaces:**
- Produces: `HeadCropConfig(maximum_hole_faces=1000, opening_rings=3, minimum_detected_frames=3, minimum_selected_faces=10_000)`
- Produces: `crop_head_mesh(source_path, selected_frames_root, transforms_path, output_path, face_anchor_path, diagnostics_root, model_path, config, device, detector=None, parser=None, rasterizer=None) -> dict`
- Consumes: `build_head_mask`, `face_oval_mask`, `MediaPipeFaceAnchorDetector`, and `OnnxFaceParser` from Task 2.

- [ ] **Step 1: Replace face-crop tests with failing head-crop tests**

```python
def test_crop_exports_head_union_and_separate_face_anchor(tmp_path: Path) -> None:
    mesh, mesh_path, frames, transforms = make_crop_inputs(tmp_path)
    head_ids = {"00000.png": 0, "00001.png": 1, "00002.png": 2}
    face_ids = {"00000.png": 3, "00001.png": 4, "00002.png": 5}

    report = crop_head_mesh(
        mesh_path,
        frames,
        transforms,
        tmp_path / "head.ply",
        tmp_path / "anchor.ply",
        tmp_path / "masks",
        tmp_path / "model.onnx",
        config=HeadCropConfig(
            maximum_hole_faces=0,
            opening_rings=0,
            minimum_selected_faces=3,
        ),
        detector=fake_detector,
        parser=fake_parser,
        rasterizer=fake_rasterizer(head_ids, face_ids),
    )

    assert report["output_faces"] == 3
    assert report["face_anchor_faces"] == 3
    assert sorted((tmp_path / "masks").glob("*.png"))
```

Preserve focused tests for `aggregate_visible_faces`, `fill_small_face_gaps`,
and `open_face_selection`, updating the default opening assertion from 10 to 3.

- [ ] **Step 2: Run the crop tests and confirm the missing module failure**

Run: `python -m pytest -q production/tests/test_head_crop_stage.py`

Expected: collection fails because `production.head_crop_stage` does not exist.

- [ ] **Step 3: Implement head and anchor face aggregation**

For each detected selected frame, compute one raster and use it with both masks:

```python
labels = active_parser(image_path)
head_mask, mask_report = build_head_mask(labels, landmarks, config.mask)
anchor_mask = face_oval_mask(landmarks, image_size)
raster = active_rasterizer(frames_by_name[image_path.name], image_size)
head_rasters.append(raster)
head_masks.append(head_mask)
anchor_rasters.append(raster)
anchor_masks.append(anchor_mask)
cv2.imwrite(str(diagnostics_root / image_path.name), head_mask.astype(np.uint8) * 255)
```

Aggregate head and anchor face IDs separately. Apply bounded hole filling and a
three-ring opening only to the head selection. Export both submeshes without
repairing or adding triangles. Record mask reports, model hash, frame names,
counts at every cleanup step, output bounds, and diagnostic paths.

- [ ] **Step 4: Run crop and legacy-independent tests**

Run: `python -m pytest -q production/tests/test_head_crop_stage.py production/tests/test_head_segmentation.py`

Expected: all tests pass and no test imports `production.face_crop_stage`.

- [ ] **Step 5: Commit the multi-view crop**

```bash
git add production/head_crop_stage.py production/tests/test_head_crop_stage.py production/face_crop_stage.py production/tests/test_face_crop_stage.py
git commit -m "feat: crop observed head from parsed views"
```

### Task 4: Face-Anchored Head Canonicalization

**Files:**
- Modify: `production/clean_face_mesh.py`
- Modify: `production/tests/test_clean_face_mesh.py`

**Interfaces:**
- Changes: `clean_face_mesh(..., camera_to_world_matrices=None, orientation_path: Path | None = None) -> dict`
- Behavior: cleanup applies to `source_path`; `canonical_face_transform` receives vertices loaded from `orientation_path` when provided.
- Produces report key: `orientation_vertices` and canonicalization value `camera_pca_orientation_mesh`.

- [ ] **Step 1: Write a failing orientation-mesh test**

```python
def test_cleanup_orients_full_head_from_face_anchor(tmp_path: Path) -> None:
    head = trimesh.creation.icosphere(subdivisions=2)
    anchor = make_rotated_face_patch()
    head_path = tmp_path / "head.ply"
    anchor_path = tmp_path / "anchor.ply"
    head.export(head_path)
    anchor.export(anchor_path)

    report = clean_face_mesh(
        head_path,
        tmp_path / "clean.ply",
        permissive_cleanup_config(),
        camera_to_world_matrices=make_front_cameras(anchor),
        orientation_path=anchor_path,
    )

    assert report["canonicalization"] == "camera_pca_orientation_mesh"
    assert report["orientation_vertices"] == len(anchor.vertices)
```

- [ ] **Step 2: Run the test and confirm the signature failure**

Run: `python -m pytest -q production/tests/test_clean_face_mesh.py::test_cleanup_orients_full_head_from_face_anchor`

Expected: failure because `clean_face_mesh` does not accept `orientation_path`.

- [ ] **Step 3: Implement orientation-mesh loading and validation**

```python
orientation = cleaned
canonicalization = "camera_pca"
if orientation_path is not None:
    orientation = load_triangle_mesh(Path(orientation_path))
    canonicalization = "camera_pca_orientation_mesh"
source_to_output = canonical_face_transform(
    np.asarray(orientation.vertices),
    camera_to_world_matrices,
    config.target_face_height,
)
```

Reject an empty or non-finite orientation mesh. Continue exporting cleaned head
geometry in source coordinates because the transform is applied during asset
export, matching current behavior.

- [ ] **Step 4: Run cleanup tests**

Run: `python -m pytest -q production/tests/test_clean_face_mesh.py`

Expected: all existing tests and the new anchor test pass.

- [ ] **Step 5: Commit canonicalization**

```bash
git add production/clean_face_mesh.py production/tests/test_clean_face_mesh.py
git commit -m "feat: orient head geometry from face anchor"
```

### Task 5: Pipeline, Resume, and Operator Wiring

**Files:**
- Modify: `production/run_video_to_glb.py`
- Modify: `production/tests/test_run_video_to_glb.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Adds CLI: `--head-parsing-model PATH`
- Adds CLI: `--head-neck-height-ratio FLOAT` with default `0.45`
- Adds CLI: `--head-maximum-hole-faces INT` with default `1000`
- Adds CLI: `--head-opening-rings INT` with default `3`
- Changes stage name: `face_crop` to `head_crop`
- Changes artifacts: `head_crop.ply`, `face_anchor.ply`, `head_geometry.ply`, `head_masks/`, and `export/head.glb`
- Produces: `build_head_crop_resume_config(...) -> dict`

- [ ] **Step 1: Write failing pipeline and resume tests**

```python
def test_pipeline_uses_full_head_crop() -> None:
    assert PIPELINE_STAGES == (
        "reconstruction",
        "head_crop",
        "clean_geometry",
        "texture",
        "asset_export",
        "validate_publish",
    )


def test_head_crop_resume_tracks_model_and_segmentation_code(tmp_path: Path) -> None:
    config, mesh, transforms, frames, model = make_head_resume_inputs(tmp_path)
    first = build_head_crop_resume_config(config, mesh, transforms, frames, model)
    model.write_bytes(b"changed-model")
    second = build_head_crop_resume_config(config, mesh, transforms, frames, model)
    assert first["model_sha256"] != second["model_sha256"]
    assert len(first["segmentation_code_sha256"]) == 64
```

Update the dry-run test to pass a model path without requiring it to exist.

- [ ] **Step 2: Run pipeline tests and confirm old stage assertions fail**

Run: `python -m pytest -q production/tests/test_run_video_to_glb.py`

Expected: failure because `PIPELINE_STAGES` still contains `face_crop`.

- [ ] **Step 3: Wire head crop and face-anchor orientation**

```python
head_report = execute_stage(
    "head_crop",
    build_head_crop_resume_config(...),
    (head_crop_geometry, face_anchor_geometry, diagnostics_root),
    head_crop_report_path,
    lambda: crop_head_mesh(...),
    state,
    args.resume,
)

clean_report = execute_stage(
    "clean_geometry",
    build_clean_resume_config(
        cleanup_config, head_crop_geometry, transforms_path, face_anchor_geometry
    ),
    (clean_geometry,),
    clean_report_path,
    lambda: clean_face_mesh(
        head_crop_geometry,
        clean_geometry,
        cleanup_config,
        camera_to_world_matrices=load_camera_to_world_matrices(transforms_path),
        orientation_path=face_anchor_geometry,
    ),
    state,
    args.resume,
)
```

Rename internal export stem from `face` to `head`, but keep the caller-provided
final output path authoritative. Include hashes of `head_crop_stage.py`,
`head_segmentation.py`, the renderer, source mesh, transforms, selected frames,
and ONNX model in the crop fingerprint. Include the face-anchor hash in the
cleanup fingerprint.

- [ ] **Step 4: Update operator documentation and run the full suite**

Document this installation command in `AGENTS.md` and `README.md`:

```bash
uv run python -m production.download_models --output models/face-parsing-resnet18.onnx
uv run python -m production.run_video_to_glb \
  --video /path/capture.mov \
  --job-root /path/job \
  --output /path/result/head.glb \
  --head-parsing-model models/face-parsing-resnet18.onnx \
  --physical-gpu 1 \
  --resume
```

Run: `python -m pytest -q production/tests`

Expected: the full production suite passes.

- [ ] **Step 5: Commit pipeline integration**

```bash
git add production/run_video_to_glb.py production/tests/test_run_video_to_glb.py README.md AGENTS.md
git commit -m "feat: run full-head video to GLB pipeline"
```

### Task 6: Server Installation and Visual Acceptance

**Files:**
- Runtime only: `/home/ii/STFR-production/models/face-parsing-resnet18.onnx`
- Runtime only: `/home/ii/STFR-production/workspace/production-full-head-ljf/`
- Deliver: `outputs/ljf-full-head/head.glb`
- Deliver: `outputs/ljf-full-head/uv.png`
- Deliver: `outputs/ljf-full-head/head-masks.jpg`
- Deliver: `outputs/ljf-full-head/front.png`
- Deliver: `outputs/ljf-full-head/left.png`
- Deliver: `outputs/ljf-full-head/right.png`

**Interfaces:**
- Consumes the existing reconstruction at `/home/ii/STFR-production/workspace/production-cube-ljf-v2/workspace`.
- Produces acceptance artifacts and measured wall-clock stage timings.

- [ ] **Step 1: Verify branch state and synchronize only tracked implementation files**

Run locally: `git status --short --branch && git log -6 --oneline`

Expected: branch is `full-head-pipeline`; only intentional commits are present.

Transfer changed tracked files to the server without resetting or cleaning its
dirty runtime checkout.

- [ ] **Step 2: Install and verify the parser model on the server**

```bash
cd /home/ii/STFR-production
.venv/bin/python -m production.download_models \
  --output models/face-parsing-resnet18.onnx
sha256sum models/face-parsing-resnet18.onnx
```

Expected SHA-256:
`0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`.

- [ ] **Step 3: Run server tests and a real parser smoke test**

```bash
cd /home/ii/STFR-production
.venv/bin/python -m pytest -q production/tests
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c \
  "from pathlib import Path; from production.head_segmentation import OnnxFaceParser; print(OnnxFaceParser(Path('models/face-parsing-resnet18.onnx'))(Path('workspace/production-cube-ljf-v2/workspace/images/00033.png')).shape)"
```

Expected: all tests pass and the parser prints the source image height and width.

- [ ] **Step 4: Resume the saved reconstruction and time downstream stages**

Seed a new job root with the saved reconstruction outputs and matching pipeline
state, then run:

```bash
cd /home/ii/STFR-production
CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v .venv/bin/python -m production.run_video_to_glb \
  --video data/ljf.MOV \
  --job-root workspace/production-full-head-ljf \
  --output workspace/production-full-head-ljf/result/head.glb \
  --head-parsing-model models/face-parsing-resnet18.onnx \
  --physical-gpu 1 \
  --resume
```

Expected: reconstruction is skipped through validated resume and all downstream
stages complete. Record each stage duration and total post-reconstruction time.

- [ ] **Step 5: Render and inspect acceptance views**

Render the GLB at front, left, and right viewpoints with a neutral background.
Build a contact sheet from saved binary head masks. Reject and tune the crop if
any view contains shirt, shoulders, a missing ear, a clipped hairline, narrow
floating geometry, face texture cuts, or a tilted canonical pose.

- [ ] **Step 6: Run final verification and commit any measured tuning**

Run on the server:

```bash
.venv/bin/python -m pytest -q production/tests
.venv/bin/python -m production.validate_asset \
  --source workspace/production-full-head-ljf/artifacts/head_geometry.ply \
  --obj workspace/production-full-head-ljf/artifacts/export/head.obj \
  --texture workspace/production-full-head-ljf/artifacts/export/uv.png \
  --glb workspace/production-full-head-ljf/result/head.glb \
  --report workspace/production-full-head-ljf/artifacts/final-validation-report.json
```

Expected: tests and asset validation pass after any visual tuning.

Commit measured tuning separately:

```bash
git add production README.md AGENTS.md
git commit -m "fix: tune full-head crop on production capture"
```

- [ ] **Step 7: Copy acceptance artifacts to the user output directory**

Copy the final GLB, UV texture, mask contact sheet, and three renders to
`C:\Users\Ilya\Documents\Codex\2026-09-07\dj\outputs\ljf-full-head\`. Report
the GLB SHA-256, face count, texture size, stage timings, and any remaining
front/side limitation.
