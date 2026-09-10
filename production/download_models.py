from __future__ import annotations

import argparse
import json
from pathlib import Path

from production.model_assets import FACE_PARSING_MODEL, download_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install verified external model weights for STFR production."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / FACE_PARSING_MODEL.filename,
    )
    args = parser.parse_args(argv)
    print(json.dumps(download_model(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
