"""Quantitative evaluation on the Fill50K test split (pixel-mask IoU).

Runs inference with a checkpoint and scores each sample:

    IoU = |foreground_true ∩ foreground_pred| / |foreground_true ∪ foreground_pred|

Foreground masks are built from RGB (corner background estimate), optionally reduced to the
largest connected component per image.

Examples:

  python imperfect_controls/evaluate.py --checkpoint imperfect_controls/checkpoints/run/latest.ckpt

  python imperfect_controls/evaluate.py \\
      --checkpoint path/to.ckpt --max-samples 100 --output-json metrics.json

  python imperfect_controls/evaluate.py --checkpoint path/to.ckpt --corrupt-fraction 0.5
"""

# TODO: need for circle with cropping / cut off at edges or corners

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from dataset import Fill50KDataset
from eval_helpers import (
    color_errors_from_true_mask,
    load_model,
    mask_center,
    mask_radius,
    mask_roundness,
    pixel_mask_iou_from_images,
    summarize_metrics,
    target_to_uint8_rgb,
)
from generate_image import generate_image
from imperfect_fill50k_dataset import DisjointCorruptFill50KDataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning .ckpt used for inference evaluation.",
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
        default=8.0,
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
        "--debug-mask-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to save per-sample binary masks used for IoU "
            "(writes *_true_mask.png and *_pred_mask.png)."
        ),
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

    ckpt_path = Path(args.checkpoint).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.corrupt_fraction is not None and not 0.0 <= args.corrupt_fraction <= 1.0:
        parser.error("--corrupt-fraction must be in [0, 1].")

    seed_everything(args.seed)

    records: List[Dict[str, Any]] = []
    ious: List[float] = []
    roundness_deltas: List[float] = []
    radius_errors: List[float] = []
    center_errors: List[float] = []
    circle_color_errors: List[float] = []
    background_color_errors: List[float] = []
    total_color_errors: List[float] = []
    keep_largest = not args.no_keep_largest
    debug_mask_dir = Path(args.debug_mask_dir).resolve() if args.debug_mask_dir else None
    if debug_mask_dir is not None:
        debug_mask_dir.mkdir(parents=True, exist_ok=True)

    # --- Inference + scoring ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference evaluation.")

    device = torch.device("cuda")
    model = load_model(ckpt_path, device, _REPO_ROOT)
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
        debug_prefix = (
            str(debug_mask_dir / f"eval_{i:06d}_dataset_idx_{dataset_i:06d}")
            if debug_mask_dir is not None
            else None
        )

        iou, true_mask, pred_mask = pixel_mask_iou_from_images(
            target_rgb,
            pred_rgb,
            margin=args.mask_margin,
            keep_largest=keep_largest,
            debug_save_prefix=debug_prefix,
        )
        true_roundness = mask_roundness(true_mask)
        pred_roundness = mask_roundness(pred_mask)
        roundness_abs_delta = abs(pred_roundness - true_roundness)
        true_radius = mask_radius(true_mask)
        pred_radius = mask_radius(pred_mask)
        radius_error = abs(pred_radius - true_radius) / true_radius if true_radius > 0.0 else 0.0
        true_cx, true_cy = mask_center(true_mask)
        pred_cx, pred_cy = mask_center(pred_mask)
        center_dist = np.sqrt((pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2)
        center_error = float(center_dist / true_radius) if true_radius > 0.0 else 0.0
        circle_color_error, background_color_error, total_color_error = color_errors_from_true_mask(
            target_rgb, pred_rgb, true_mask
        )
        ious.append(iou)
        roundness_deltas.append(roundness_abs_delta)
        radius_errors.append(radius_error)
        center_errors.append(center_error)
        circle_color_errors.append(circle_color_error)
        background_color_errors.append(background_color_error)
        total_color_errors.append(total_color_error)
        records.append(
            {
                "local_index": i,
                "dataset_index": dataset_i,
                "prompt": prompt,
                "mask_iou": iou,
                "roundness_true": true_roundness,
                "roundness_pred": pred_roundness,
                "roundness_abs_delta": roundness_abs_delta,
                "radius_true": true_radius,
                "radius_pred": pred_radius,
                "radius_error": radius_error,
                "center_true_x": true_cx,
                "center_true_y": true_cy,
                "center_pred_x": pred_cx,
                "center_pred_y": pred_cy,
                "center_error": center_error,
                "circle_color_error": circle_color_error,
                "background_color_error": background_color_error,
                "total_color_error": total_color_error,
            }
        )

    summary = summarize_metrics(
        ious,
        roundness_deltas,
        radius_errors,
        center_errors,
        circle_color_errors,
        background_color_errors,
        total_color_errors,
    )
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
