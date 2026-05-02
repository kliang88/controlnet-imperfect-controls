#!/usr/bin/env python3
"""Download CLIP tokenizer + CLIP text weights (``openai/clip-vit-large-patch14``) to ``models/``."""

import inspect
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "openai/clip-vit-large-patch14"

# PyTorch CLIP text path + tokenizer; excludes flax/tf duplicate weights (~3.5GB+).
_REPO_FILES = (
    "config.json",
    "pytorch_model.bin",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _token_kw(download_fn):
    tok = os.getenv("HF_TOKEN")
    if not tok:
        return {}
    p = inspect.signature(download_fn).parameters
    if "token" in p:
        return {"token": tok}
    if "use_auth_token" in p:
        return {"use_auth_token": tok}
    return {}


def _local_ok(path: Path) -> bool:
    try:
        CLIPTokenizer.from_pretrained(str(path), local_files_only=True)
        CLIPTextModel.from_pretrained(str(path), local_files_only=True)
        return True
    except Exception:
        return False


def _snapshot_into(out: Path) -> None:
    token_kw = _token_kw(snapshot_download)
    sig = inspect.signature(snapshot_download).parameters
    kwargs = dict(
        repo_id=MODEL_ID,
        repo_type="model",
        local_dir=str(out),
        **token_kw,
    )
    if "allow_patterns" in sig:
        kwargs["allow_patterns"] = list(_REPO_FILES)
    elif "ignore_patterns" in sig:
        # Old hub without allow_patterns — skip bulky non-PyTorch weights.
        kwargs["ignore_patterns"] = ["*.msgpack", "*.h5", "model.safetensors"]

    snapshot_download(**kwargs)


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "models/clip-vit-large-patch14"
    out.mkdir(parents=True, exist_ok=True)
    if _local_ok(out):
        print(f"OK (cached): {out}")
        return 0

    print(f"Downloading {MODEL_ID} ...")
    try:
        _snapshot_into(out)
    except Exception as e:
        print(
            "Could not fetch from Hugging Face (network, hub downtime, or bad HF_TOKEN).\n",
            e,
            file=sys.stderr,
        )
        return 1

    if not _local_ok(out):
        print(
            "Files on disk do not load; try: pip install -U transformers huggingface_hub tokenizers.",
            file=sys.stderr,
        )
        return 1
    print(f"Done: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
