#!/usr/bin/env python3
"""Download and extract Fill50K into `training/fill50k`."""

import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "lllyasviel/ControlNet"
ZIP_IN_REPO = "training/fill50k.zip"


def main() -> int:
    root = Path(__file__).resolve().parent
    training_dir = root / "training"
    expected = training_dir / "fill50k" / "prompt.json"
    if expected.is_file():
        print(f"Already present: {expected.parent}")
        return 0

    print("Downloading fill50k.zip from Hugging Face...")
    zip_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ZIP_IN_REPO,
            repo_type="model",
        )
    )

    print(f"Extracting to {training_dir}...")
    training_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(training_dir)

    if not expected.is_file():
        print(f"ERROR: Missing expected file after extract: {expected}", file=sys.stderr)
        return 1

    print(f"Finished downloading Fill50K dataset to {expected.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
