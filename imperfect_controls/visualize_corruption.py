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

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from imperfect_fill50k_dataset import CORRUPTION_FUNCS  # noqa: E402


def _read_rgb_float(path: Path) -> np.ndarray:
    """Read image to float32 RGB in [0, 1], preferring cv2 but falling back to PIL."""
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit("failed to read image: {}".format(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    except ModuleNotFoundError:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return (np.asarray(img).astype(np.float32) / 255.0).clip(0.0, 1.0)


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
    return _read_rgb_float(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Global Fill50k row index (order in prompt.json).",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to an image to corrupt (RGB). "
            "If set, this bypasses Fill50k loading and ignores --index."
        ),
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
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful on headless machines).",
    )
    args = parser.parse_args()

    if args.input_image:
        hint_clean = _read_rgb_float(Path(args.input_image).expanduser())
        index_str = str(Path(args.input_image).name)
    else:
        dataset_root = _SCRIPT_DIR / "training" / "fill50k"
        if not (dataset_root / "prompt.json").is_file():
            raise SystemExit(
                "missing {}; run download_fill50k.py first (or pass --input-image)".format(
                    dataset_root / "prompt.json"
                )
            )
        hint_clean = load_hint_float(dataset_root, args.index)
        index_str = str(args.index)
    rng = np.random.default_rng(args.seed)
    fn = CORRUPTION_FUNCS[args.corruption_type]
    hint_corrupt = fn(hint_clean, rng)

    if args.output:
        # Prefer matplotlib for nice labels, but fall back to a simple PIL concatenation
        # if matplotlib isn't installed in this environment.
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(np.clip(hint_clean, 0.0, 1.0))
            axes[0].set_title("hint (original)")
            axes[0].axis("off")
            axes[1].imshow(np.clip(hint_corrupt, 0.0, 1.0))
            axes[1].set_title("hint ({})".format(args.corruption_type))
            axes[1].axis("off")
            fig.suptitle("{}  |  seed {}".format(index_str, args.seed))
            fig.tight_layout()
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ModuleNotFoundError:
            from PIL import Image, ImageDraw, ImageFont

            a = (np.clip(hint_clean, 0.0, 1.0) * 255.0).astype(np.uint8)
            b = (np.clip(hint_corrupt, 0.0, 1.0) * 255.0).astype(np.uint8)
            im_a = Image.fromarray(a)
            im_b = Image.fromarray(b)
            gap = 12
            label_h = 26
            canvas = Image.new(
                "RGB",
                (im_a.width + gap + im_b.width, max(im_a.height, im_b.height) + label_h),
                (255, 255, 255),
            )
            canvas.paste(im_a, (0, label_h))
            canvas.paste(im_b, (im_a.width + gap, label_h))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            draw.text((4, 6), f"{index_str} | seed {args.seed}", fill=(0, 0, 0), font=font)
            draw.text((4, label_h + 4), "original", fill=(0, 0, 0), font=font)
            draw.text((im_a.width + gap + 4, label_h + 4), args.corruption_type, fill=(0, 0, 0), font=font)
            canvas.save(out_path)
        print("wrote {}".format(out_path))

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(np.clip(hint_clean, 0.0, 1.0))
            axes[0].set_title("hint (original)")
            axes[0].axis("off")
            axes[1].imshow(np.clip(hint_corrupt, 0.0, 1.0))
            axes[1].set_title("hint ({})".format(args.corruption_type))
            axes[1].axis("off")
            fig.suptitle("{}  |  seed {}".format(index_str, args.seed))
            fig.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            # No interactive display available without matplotlib.
            pass


if __name__ == "__main__":
    main()
