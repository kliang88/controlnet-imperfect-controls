#!/usr/bin/env python3
"""Download SD 1.5 ``v1-5-pruned.ckpt`` into the repo-root ``models/`` directory."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
FILENAME = "v1-5-pruned.ckpt"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / FILENAME

    if dest.is_file():
        print(f"Already present: {dest}")
        return 0

    print(f"Downloading {FILENAME} from {REPO_ID}...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(out_dir),
        repo_type="model"
    )

    print(f"Finished downloading SD 1.5 checkpoint to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
