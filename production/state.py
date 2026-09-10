from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_fingerprint(path: Path) -> dict[str, Any] | None:
    path = Path(path).resolve()
    if path.is_file():
        size = path.stat().st_size
        if size == 0:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(path),
            "kind": "file",
            "bytes": size,
            "sha256": digest.hexdigest(),
        }
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            return None
        digest = hashlib.sha256()
        total_size = 0
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            item_fingerprint = artifact_fingerprint(item)
            if item_fingerprint is None:
                return None
            total_size += int(item_fingerprint["bytes"])
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(str(item_fingerprint["sha256"])))
        return {
            "path": str(path),
            "kind": "directory",
            "bytes": total_size,
            "files": len(files),
            "sha256": digest.hexdigest(),
        }
    return None


class PipelineState:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 2, "stages": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def start(self, stage: str, config: Any) -> None:
        self.data["stages"][stage] = {
            "status": "running",
            "config": json_value(config),
            "started_at": timestamp(),
        }
        self.save()

    def complete(
        self,
        stage: str,
        config: Any,
        outputs: Iterable[Path],
        report: Mapping[str, Any] | None = None,
    ) -> None:
        output_paths = [Path(path).resolve() for path in outputs]
        fingerprints = [artifact_fingerprint(path) for path in output_paths]
        missing = [
            path
            for path, fingerprint in zip(output_paths, fingerprints, strict=True)
            if fingerprint is None
        ]
        if missing:
            raise FileNotFoundError("Stage outputs are missing or empty: " + ", ".join(map(str, missing)))
        self.data["version"] = 2
        self.data["stages"][stage] = {
            "status": "completed",
            "config": json_value(config),
            "completed_at": timestamp(),
            "outputs": fingerprints,
            "report": json_value(report or {}),
        }
        self.save()

    def fail(self, stage: str, config: Any, error: str) -> None:
        self.data["stages"][stage] = {
            "status": "failed",
            "config": json_value(config),
            "failed_at": timestamp(),
            "error": error,
        }
        self.save()

    def can_resume(
        self,
        stage: str,
        config: Any,
        required_outputs: Iterable[Path],
    ) -> bool:
        record = self.data.get("stages", {}).get(stage)
        if not record or record.get("status") != "completed":
            return False
        if record.get("config") != json_value(config):
            return False
        expected = [Path(path).resolve() for path in required_outputs]
        recorded = {item["path"]: item for item in record.get("outputs", [])}
        for path in expected:
            item = recorded.get(str(path))
            current = artifact_fingerprint(path)
            if item is None or current is None:
                return False
            comparable_keys = ("kind", "bytes", "files", "sha256")
            if any(item.get(key) != current.get(key) for key in comparable_keys):
                return False
        return True
