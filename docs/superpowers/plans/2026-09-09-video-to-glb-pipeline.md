# STFR Video-to-GLB Production Pipeline Implementation Plan

> Superseded for production by
> `2026-09-09-direct-cube-uv-pipeline.md`. This plan is retained only as
> implementation history for the earlier Wrap/xatlas design.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add worker-callable scripts that turn one face video into one validated, canonically oriented, textured GLB.

**Architecture:** A public orchestrator invokes existing STFR stages as checked subprocesses, then calls focused Python modules for 2DGS face cleanup, UV generation, neural texture training, canonical export, and validation. Stage state and JSON progress make failures visible and allow validated resume without changing the worker.

**Tech Stack:** Python 3.10, subprocess, ffmpeg, COLMAP, 2DGS, numpy, trimesh, Open3D, PyMeshLab, xatlas, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-video-to-glb-pipeline-design.md`

## Global Constraints

- The public input is one video path and the required product output is one GLB path.
- Production uses 30,000 2DGS iterations, mesh resolution 1024, one retained mesh cluster, mask distance 0.05, three Taubin iterations, and 301 texture iterations by default.
- The physical GPU is selected through `CUDA_VISIBLE_DEVICES`; child scripts use logical `cuda:0`.
- `final_hack.obj` is only a crop reference; output geometry always comes from `2dgs_recon.obj`.
- No generative texture fill, beauty filtering, or skin retouching is allowed.
- A partial or invalid GLB is never published at the requested output path.

---

### Task 1: Explicit Mesh and Texture Controls

**Files:**
- Modify: `texture/render_gbuffer.py`
- Modify: `texture/build_texture.py`
- Create: `production/__init__.py`
- Create: `production/tests/test_texture_interfaces.py`

**Interfaces:**
- Consumes: existing STFR texture scripts.
- Produces: `--mesh_path PATH` for both scripts and `--iterations N` for texture training.

- [ ] **Step 1: Write failing AST-based interface tests**

```python
def test_texture_scripts_accept_explicit_mesh_and_iterations():
    assert argument_default(ROOT / "texture/render_gbuffer.py", "mesh_path") is None
    assert argument_default(ROOT / "texture/build_texture.py", "mesh_path") is None
    assert argument_default(ROOT / "texture/build_texture.py", "iterations") == 301
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q production/tests/test_texture_interfaces.py`
Expected: FAIL because the options do not exist.

- [ ] **Step 3: Add the options and replace hard-coded paths/iteration count**

Use `mesh_path = opt.mesh_path or <legacy path>` and loop over `range(opt.iterations)`.
Use `os.environ.setdefault("CUDA_VISIBLE_DEVICES", ...)` so the parent controls physical GPU mapping.

- [ ] **Step 4: Run the focused test**

Run: `pytest -q production/tests/test_texture_interfaces.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add texture production
git commit -m "feat: expose production texture controls"
```

### Task 2: UV Export and Face Geometry Cleanup

**Files:**
- Create: `production/unwrap_2dgs_uv.py`
- Create: `production/clean_face_mesh.py`
- Create: `production/tests/test_unwrap_2dgs_uv.py`
- Create: `production/tests/test_clean_face_mesh.py`

**Interfaces:**
- Produces: `unwrap_mesh(input_path, output_obj, material_name, texture_name) -> dict`.
- Produces: `fit_correspondence_transform(source, canonical) -> ndarray`.
- Produces: `clean_face_mesh(source_path, canonical_path, reference_path, output_source, output_canonical, config) -> dict`.

- [ ] **Step 1: Write failing tests with asymmetric synthetic meshes**

```python
def test_correspondence_recovers_affine_transform():
    matrix = fit_correspondence_transform(source, transform_points(source, expected))
    assert np.allclose(matrix, expected)

def test_unwrap_preserves_triangles_and_binds_texture(tmp_path):
    report = unwrap_mesh(source, tmp_path / "face.obj", "face", "uv.png")
    assert report["source_faces"] == report["output_faces"]
    assert "map_Kd uv.png" in (tmp_path / "face.mtl").read_text()
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `pytest -q production/tests/test_unwrap_2dgs_uv.py production/tests/test_clean_face_mesh.py`
Expected: collection FAIL.

- [ ] **Step 3: Implement deterministic xatlas OBJ/MTL export**

Write seam-split vertices, UVs, transformed normals, and one material with ASCII output.

- [ ] **Step 4: Implement correspondence, eye-safe mask closure, crop, largest-component selection, and Taubin smoothing**

Reject non-corresponding source/canonical topology and report residual, component sizes,
roughness, displacement, bounds, and face counts.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q production/tests/test_unwrap_2dgs_uv.py production/tests/test_clean_face_mesh.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add production
git commit -m "feat: clean and unwrap detailed 2dgs face mesh"
```

### Task 3: Checked Reconstruction and Texture Stages

**Files:**
- Create: `production/runtime.py`
- Create: `production/reconstruction_stage.py`
- Create: `production/texture_stage.py`
- Create: `production/tests/test_reconstruction_stage.py`
- Create: `production/tests/test_runtime.py`

**Interfaces:**
- Produces: `CommandSpec(stage: str, argv: tuple[str, ...], cwd: Path)`.
- Produces: `build_reconstruction_commands(config) -> list[CommandSpec]`.
- Produces: `run_checked(spec, environment, emit) -> None`.
- Produces: `run_texture_stage(config, emit) -> dict`.

- [ ] **Step 1: Write failing command-construction and failure-propagation tests**

```python
def test_reconstruction_uses_full_checkpoint_and_one_cluster(config):
    commands = build_reconstruction_commands(config)
    assert "--iterations" in train.argv and "30000" in train.argv
    assert "--iteration" in render.argv and "30000" in render.argv
    assert "--mesh_res" in render.argv and "1024" in render.argv
    assert "--num_cluster" in render.argv and "1" in render.argv

def test_run_checked_raises_on_nonzero_exit(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_checked(CommandSpec("bad", (sys.executable, "-c", "raise SystemExit(7)"), tmp_path), {}, lambda _: None)
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `pytest -q production/tests/test_reconstruction_stage.py production/tests/test_runtime.py`
Expected: collection FAIL.

- [ ] **Step 3: Implement JSON events and signal-forwarding checked subprocesses**

Events contain `stage`, `status`, `timestamp`, and optional `duration_seconds`/`command`.

- [ ] **Step 4: Implement exact ffmpeg, matting, COLMAP, 2DGS, conversion, refinement, and registration commands**

Validate the 30,000 checkpoint and `2dgs_recon.obj` before completing reconstruction.

- [ ] **Step 5: Implement selected-frame dataset preparation and texture subprocesses**

Call `render_gbuffer.py` and `build_texture.py` with explicit cleaned mesh and logical device zero.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q production/tests/test_reconstruction_stage.py production/tests/test_runtime.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add production
git commit -m "feat: add checked STFR production stages"
```

### Task 4: Canonical Export, Validation, and Resume State

**Files:**
- Create: `production/export_glb.py`
- Create: `production/validate_asset.py`
- Create: `production/state.py`
- Create: `production/tests/test_export_glb.py`
- Create: `production/tests/test_validate_asset.py`
- Create: `production/tests/test_state.py`

**Interfaces:**
- Produces: `export_canonical_asset(input_obj, input_mtl, texture, matrix, output_dir) -> dict`.
- Produces: `validate_asset(source_mesh, obj, texture, glb) -> dict`.
- Produces: `PipelineState.can_resume(stage, config, required_outputs) -> bool`.

- [ ] **Step 1: Write failing transform, GLB reload, and resume tests**

```python
def test_export_transforms_positions_and_normals(tmp_path):
    result = export_canonical_asset(obj, mtl, texture, matrix, tmp_path)
    loaded = trimesh.load(result["glb"], process=False, force="scene")
    assert np.allclose(loaded.bounds, expected_bounds)

def test_resume_rejects_changed_configuration(tmp_path):
    state.complete("texture", {"iterations": 301}, [texture])
    assert not state.can_resume("texture", {"iterations": 151}, [texture])
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `pytest -q production/tests/test_export_glb.py production/tests/test_validate_asset.py production/tests/test_state.py`
Expected: collection FAIL.

- [ ] **Step 3: Implement canonical OBJ normal/position transformation and embedded GLB export**

Write GLB to a temporary artifact path; publication is owned by the orchestrator.

- [ ] **Step 4: Implement topology, UV, texture, material, bounds, and GLB reload validation**

Return JSON-serializable metrics and raise `ValueError` on every failed gate.

- [ ] **Step 5: Implement atomic JSON state with configuration matching**

Write state through a sibling temporary file followed by `os.replace`.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q production/tests/test_export_glb.py production/tests/test_validate_asset.py production/tests/test_state.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add production
git commit -m "feat: export and validate canonical face assets"
```

### Task 5: Public Video-to-GLB Orchestrator

**Files:**
- Create: `production/run_video_to_glb.py`
- Create: `production/tests/test_run_video_to_glb.py`
- Modify: `ENV.md`
- Modify: `README.md`

**Interfaces:**
- Produces CLI: `python -m production.run_video_to_glb --video PATH --job-root DIR --output PATH --physical-gpu N [--resume] [--dry-run]`.
- Produces: requested GLB and sibling `result.json`.

- [ ] **Step 1: Write failing dry-run and publication tests**

```python
def test_dry_run_lists_all_stages_without_gpu_work(tmp_path, capsys):
    assert main(["--video", str(video), "--job-root", str(job), "--output", str(out), "--dry-run"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["stage"] for event in events if event["status"] == "planned"] == EXPECTED_STAGES

def test_publish_uses_atomic_replace(tmp_path, monkeypatch):
    publish_validated_glb(staged, output, report)
    assert output.read_bytes() == staged.read_bytes()
    assert json.loads(output.with_name("result.json").read_text())["status"] == "complete"
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q production/tests/test_run_video_to_glb.py`
Expected: collection FAIL.

- [ ] **Step 3: Implement argument validation, stage orchestration, resume checks, and atomic publication**

Reject an existing non-empty job root unless `--resume` is supplied. Keep native STFR
outputs under `workspace`, product inputs under `artifacts`, and publish only after validation.

- [ ] **Step 4: Document the worker command and dependencies**

Add xatlas and pymeshlab to `ENV.md`; add one production invocation and artifact contract to `README.md`.

- [ ] **Step 5: Run all production tests and static checks**

Run: `pytest -q production/tests`
Expected: all PASS.

Run: `python -m production.run_video_to_glb --video examples/input.mov --job-root /tmp/stfr-dry --output /tmp/stfr-dry/result/face.glb --dry-run`
Expected: JSON `planned` events and exit code zero without touching CUDA.

Run: `python -m compileall -q production texture`
Expected: exit code zero.

- [ ] **Step 6: Run server integration from the existing 30,000-step workspace**

Use `--resume` against `/home/ii/STFR/workspace/test`, verify GLB reload, texture binding,
face count, canonical bounds, and hashes. Do not retrain 2DGS when the validated 30,000
checkpoint and mesh already exist.

- [ ] **Step 7: Commit**

```bash
git add production ENV.md README.md
git commit -m "feat: add video-to-glb production entrypoint"
```

### Task 6: Final Review and Verification

**Files:**
- Review: all files changed since `22a691d`

**Interfaces:**
- Consumes: complete public CLI and test suite.
- Produces: reviewed branch ready for the worker repository to reference.

- [ ] **Step 1: Review the branch diff for command injection, accidental absolute paths, and silent failures**

Run: `git diff --check 22a691d..HEAD`
Expected: no output.

- [ ] **Step 2: Run final tests**

Run: `pytest -q production/tests`
Expected: all PASS.

- [ ] **Step 3: Confirm clean repository state and summarize commits**

Run: `git status --short --branch`
Expected: clean `pipeline` branch.
