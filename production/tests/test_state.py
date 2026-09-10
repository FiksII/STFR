from pathlib import Path

from production.state import PipelineState


def test_resume_requires_matching_configuration_and_output(tmp_path: Path) -> None:
    output = tmp_path / "mesh.ply"
    output.write_bytes(b"mesh")
    state = PipelineState(tmp_path / "pipeline-state.json")
    state.complete("geometry", {"distance": 0.05}, [output], {"faces": 10})

    reloaded = PipelineState(tmp_path / "pipeline-state.json")
    assert reloaded.can_resume("geometry", {"distance": 0.05}, [output])
    assert not reloaded.can_resume("geometry", {"distance": 0.02}, [output])
    output.unlink()
    assert not reloaded.can_resume("geometry", {"distance": 0.05}, [output])


def test_resume_rejects_same_size_output_content_change(tmp_path: Path) -> None:
    output = tmp_path / "mesh.ply"
    output.write_bytes(b"mesh-a")
    state = PipelineState(tmp_path / "pipeline-state.json")
    state.complete("geometry", {"distance": 0.05}, [output])

    output.write_bytes(b"mesh-b")

    assert not state.can_resume("geometry", {"distance": 0.05}, [output])


def test_resume_tracks_directory_tree_contents(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "00001.png").write_bytes(b"frame-a")
    state = PipelineState(tmp_path / "pipeline-state.json")
    state.complete("reconstruction", {"iterations": 30000}, [frames])

    assert state.can_resume("reconstruction", {"iterations": 30000}, [frames])

    (frames / "00001.png").write_bytes(b"frame-b")

    assert not state.can_resume("reconstruction", {"iterations": 30000}, [frames])


def test_state_writes_valid_json_after_failure(tmp_path: Path) -> None:
    state = PipelineState(tmp_path / "pipeline-state.json")

    state.fail("texture", {"iterations": 301}, "out of memory")

    reloaded = PipelineState(tmp_path / "pipeline-state.json")
    assert reloaded.data["stages"]["texture"]["status"] == "failed"
