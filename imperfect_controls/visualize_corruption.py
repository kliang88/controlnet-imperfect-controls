#!/usr/bin/env python3
"""Preview control (hint) corruption side-by-side with the original.

Reads one sample from ``training/fill50k`` and applies the same corruption
functions as ``imperfect_fill50k_dataset.py`` (default: edge_segment_remove).

Examples:
  python visualize_corruption.py --index 42
  python visualize_corruption.py --index 0 --seed 7 --output /tmp/hint_corrupt.png
  python visualize_corruption.py --type blur --index 100
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from imperfect_fill50k_dataset import CORRUPTION_FUNCS  # noqa: E402


def load_hint_float(dataset_root: Path, global_index: int) -> np.ndarray:
    prompt_path = dataset_root / "prompt.json"
    with open(prompt_path, "rt") as f:
        lines = f.readlines()
    if global_index < 0 or global_index >= len(lines):
        raise SystemExit(
            "index {} out of range [0, {})".format(global_index, len(lines))
        )
    item = json.loads(lines[global_index])
    path = dataset_root / item["source"]
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit("failed to read image: {}".format(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Global Fill50k row index (order in prompt.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for corruption (same as dataset: deterministic per seed).",
    )
    parser.add_argument(
        "--type",
        dest="corruption_type",
        default="edge_segment_remove",
        choices=sorted(CORRUPTION_FUNCS.keys()),
        help="Corruption to apply.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Save figure to this path instead of only showing interactively.",
    )
    args = parser.parse_args()

    dataset_root = _SCRIPT_DIR / "training" / "fill50k"
    if not (dataset_root / "prompt.json").is_file():
        raise SystemExit(
            "missing {}; run download_fill50k.py first".format(dataset_root / "prompt.json")
        )

    hint_clean = load_hint_float(dataset_root, args.index)
    rng = np.random.default_rng(args.seed)
    fn = CORRUPTION_FUNCS[args.corruption_type]
    hint_corrupt = fn(hint_clean, rng)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.clip(hint_clean, 0.0, 1.0))
    axes[0].set_title("hint (original)")
    axes[0].axis("off")
    axes[1].imshow(np.clip(hint_corrupt, 0.0, 1.0))
    axes[1].set_title("hint ({})".format(args.corruption_type))
    axes[1].axis("off")
    fig.suptitle("Fill50k index {}  |  seed {}".format(args.index, args.seed))
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("wrote {}".format(out_path))
    plt.show()


if __name__ == "__main__":
    main()
