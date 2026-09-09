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


def test_state_writes_valid_json_after_failure(tmp_path: Path) -> None:
    state = PipelineState(tmp_path / "pipeline-state.json")

    state.fail("texture", {"iterations": 301}, "out of memory")

    reloaded = PipelineState(tmp_path / "pipeline-state.json")
    assert reloaded.data["stages"]["texture"]["status"] == "failed"
