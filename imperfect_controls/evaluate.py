"""Quantitative evaluation on the Fill50K test split (pixel-mask IoU).

Runs inference with a checkpoint (unless ``--score-dir`` is set) and scores each sample:

    IoU = |foreground_true ∩ foreground_pred| / |foreground_true ∪ foreground_pred|

Foreground masks are built from RGB (corner background estimate), optionally reduced to the
largest connected component per image.

Use exactly one of ``--checkpoint`` (full inference) or ``--score-dir`` (read PNGs only; no GPU).

Examples:

  python imperfect_controls/evaluate.py --checkpoint imperfect_controls/checkpoints/run/latest.ckpt

  python imperfect_controls/evaluate.py \\
      --checkpoint path/to.ckpt --max-samples 100 --output-json metrics.json

  python imperfect_controls/evaluate.py --checkpoint path/to.ckpt --corrupt-fraction 0.5

  # Folders: test_idx_*/target.png & generated.png (e.g. from a separate qualitative run)
  python imperfect_controls/evaluate.py --score-dir imperfect_controls/generated/clean
"""

# TODO: need for circle with cropping / cut off at edges or corners

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

for _p in (_REPO_ROOT, _SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from share import *
import config

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from dataset import Fill50KDataset
from eval_mask_iou import pixel_mask_iou_from_images
from generate_image import generate_image
from imperfect_fill50k_dataset import DisjointCorruptFill50KDataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def target_to_uint8_rgb(jpg: np.ndarray) -> np.ndarray:
    """Dataset target tensor: float RGB [-1, 1] -> uint8 RGB."""
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def load_rgb_png(path: Path) -> np.ndarray:
    """RGB uint8 H×W×3."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def load_model(checkpoint: Path, device: torch.device):
    model = create_model(str(_REPO_ROOT / "models/cldm_v15.yaml")).to(device)
    model.load_state_dict(load_state_dict(str(checkpoint), location=str(device)))
    model.sd_locked = True
    model.eval()
    return model


def iter_saved_samples(score_dir: Path) -> Iterator[Tuple[int, Path, Path]]:
    """Yield (dataset_index, target_path, generated_path) from ``test_idx_*`` folders."""
    for folder in sorted(score_dir.glob("test_idx_*")):
        if not folder.is_dir():
            continue
        m = re.match(r"test_idx_(\d+)$", folder.name)
        if not m:
            continue
        dataset_i = int(m.group(1))
        tp = folder / "target.png"
        gp = folder / "generated.png"
        if tp.is_file() and gp.is_file():
            yield dataset_i, tp, gp


def _summarize(ious: List[float]) -> Dict[str, Any]:
    if not ious:
        return {
            "count": 0,
            "mask_iou_mean": None,
            "mask_iou_median": None,
            "mask_iou_std": None,
        }
    arr = np.asarray(ious, dtype=np.float64)
    return {
        "count": len(ious),
        "mask_iou_mean": float(arr.mean()),
        "mask_iou_median": float(np.median(arr)),
        "mask_iou_std": float(arr.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Lightning .ckpt (required unless --score-dir is used alone).",
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default=None,
        help=(
            "Directory containing test_idx_*/target.png and generated.png. "
            "If set without --checkpoint, only aggregates IoU from disk (no inference)."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--a-prompt", default="")
    parser.add_argument("--n-prompt", default="")
    parser.add_argument(
        "--mask-margin",
        type=float,
        default=18.0,
        help="Foreground/background separation margin (RGB L2); tune if masks look wrong.",
    )
    parser.add_argument(
        "--no-keep-largest",
        action="store_true",
        help="Do not reduce masks to the largest connected component before IoU.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write detailed results and summary to this path.",
    )
    parser.add_argument(
        "--corrupt-fraction",
        type=float,
        default=None,
        metavar="P",
        help=(
            "If set (0..1), evaluate with DisjointCorruptFill50KDataset on the test split "
            "(same scheme as training imperfect controls)."
        ),
    )
    parser.add_argument(
        "--corruption-type",
        type=str,
        default="edge_segment_remove",
        help="Corruption name when --corrupt-fraction is set.",
    )

    args = parser.parse_args()

    score_path = Path(args.score_dir).resolve() if args.score_dir else None
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else None

    if (score_path is None) == (ckpt_path is None):
        parser.error("Provide exactly one of: --checkpoint (inference) or --score-dir (disk-only).")

    if ckpt_path is not None and not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.corrupt_fraction is not None and score_path is not None:
        parser.error("--corrupt-fraction applies only to --checkpoint runs.")

    if args.corrupt_fraction is not None and not 0.0 <= args.corrupt_fraction <= 1.0:
        parser.error("--corrupt-fraction must be in [0, 1].")

    seed_everything(args.seed)

    records: List[Dict[str, Any]] = []
    ious: List[float] = []
    keep_largest = not args.no_keep_largest

    # --- Disk-only scoring ---
    if ckpt_path is None and score_path is not None:
        pairs = list(iter_saved_samples(score_path))
        if args.max_samples is not None:
            pairs = pairs[: args.max_samples]
        for dataset_i, tp, gp in tqdm(pairs, desc="score-disk"):
            tgt = load_rgb_png(tp)
            pred = load_rgb_png(gp)
            iou, _, _ = pixel_mask_iou_from_images(
                tgt,
                pred,
                margin=args.mask_margin,
                keep_largest=keep_largest,
            )
            ious.append(iou)
            rec: Dict[str, Any] = {
                "dataset_index": dataset_i,
                "target_path": str(tp),
                "generated_path": str(gp),
                "mask_iou": iou,
            }
            records.append(rec)

        summary = _summarize(ious)
        summary["mode"] = "score_dir"
        summary["score_dir"] = str(score_path)
        summary["keep_largest_component"] = keep_largest

        print(json.dumps(summary, indent=2))
        if args.output_json:
            out_path = Path(args.output_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "per_sample": records}, f, indent=2)
            print(f"Wrote {out_path}")
        return

    # --- Inference + scoring ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference (omit --checkpoint and use --score-dir for CPU-only).")

    device = torch.device("cuda")
    model = load_model(ckpt_path, device)
    sampler = DDIMSampler(model)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    if args.corrupt_fraction is not None:
        test_ds = DisjointCorruptFill50KDataset(
            split="test",
            corrupt_fraction=args.corrupt_fraction,
            corruption_type=args.corruption_type,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
    else:
        test_ds = Fill50KDataset(
            split="test",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty.")

    n = len(test_ds)
    if args.max_samples is not None:
        n = min(n, args.max_samples)

    for i in tqdm(range(n), desc="eval"):
        item = test_ds[i]
        dataset_i = int(test_ds.indices[i])
        prompt = item["txt"]

        target_rgb = target_to_uint8_rgb(item["jpg"])
        pred_rgb = generate_image(
            model,
            sampler,
            item["hint"],
            prompt,
            a_prompt=args.a_prompt,
            n_prompt=args.n_prompt,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            strength=args.strength,
        )

        iou, _, _ = pixel_mask_iou_from_images(
            target_rgb,
            pred_rgb,
            margin=args.mask_margin,
            keep_largest=keep_largest,
        )
        ious.append(iou)
        records.append(
            {
                "local_index": i,
                "dataset_index": dataset_i,
                "prompt": prompt,
                "mask_iou": iou,
            }
        )

    summary = _summarize(ious)
    summary["mode"] = "inference"
    summary["checkpoint"] = str(ckpt_path)
    summary["split"] = "test"
    if args.corrupt_fraction is not None:
        summary["corrupt_fraction"] = args.corrupt_fraction
        summary["corruption_type"] = args.corruption_type
    summary["ddim_steps"] = args.ddim_steps
    summary["guidance_scale"] = args.guidance_scale
    summary["strength"] = args.strength
    summary["keep_largest_component"] = keep_largest

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": records}, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
