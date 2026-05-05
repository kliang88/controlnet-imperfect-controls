"""Quantitative evaluation on the Fill50K test split (pixel-mask IoU).

Runs inference with a checkpoint and scores each sample:

    IoU = |foreground_true ∩ foreground_pred| / |foreground_true ∪ foreground_pred|

When both foreground masks are empty, IoU is defined as 1.0 (vacuous agreement).

Color error metrics use the same per-pixel RGB difference, but circle vs
background means are taken over the **predicted** extracted mask (not the target mask).

Foreground masks use Lab distance from an estimated dominant border background.
A fixed margin threshold is used by default; Otsu thresholding on normalized distance
is used when margin <= 0 or when the fixed threshold produces an empty mask.

Examples:

  python imperfect_controls/evaluate.py --checkpoint imperfect_controls/checkpoints/run/latest.ckpt

  python imperfect_controls/evaluate.py \\
      --checkpoint path/to.ckpt --max-samples 100 --output-json metrics.json

  python imperfect_controls/evaluate.py \\
      --checkpoint path/to.ckpt \\
      --output-csv imperfect_controls/eval_results/run.csv \\
      --output-summary-csv imperfect_controls/eval_results/run_summary.csv

  python imperfect_controls/evaluate.py --checkpoint path/to.ckpt --corrupt-fraction 0.5
"""

# TODO: need for circle with cropping / cut off at edges or corners

from __future__ import annotations

import argparse
import csv
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from dataset import Fill50KDataset
from eval_helpers import (
    color_errors_from_mask,
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


CSV_FIELDNAMES = [
    "local_index",
    "dataset_index",
    "prompt",
    "mask_iou",
    "roundness_true",
    "roundness_pred",
    "roundness_abs_delta",
    "radius_true",
    "radius_pred",
    "radius_error",
    "center_true_x",
    "center_true_y",
    "center_pred_x",
    "center_pred_y",
    "center_error",
    "circle_color_error",
    "background_color_error",
    "total_color_error",
]


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_records_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write one row per evaluated sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    """Write aggregate summary statistics in a readable long format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        ("mask", "mask_iou"),
        ("shape", "roundness_abs_delta"),
        ("shape", "roundness_pred"),
        ("shape", "radius_error"),
        ("shape", "center_error"),
        ("color", "circle_color_error"),
        ("color", "background_color_error"),
        ("color", "total_color_error"),
    ]

    metadata_keys = [
        "mode",
        "checkpoint",
        "split",
        "count",
        "eval_sample_count",
        "corrupt_fraction",
        "corruption_type",
        "only_corrupted",
        "ddim_steps",
        "guidance_scale",
        "strength",
        "keep_largest_component",
    ]

    rows: List[Dict[str, str]] = []

    for section, metric in metric_groups:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "mean": _csv_scalar(summary.get(f"{metric}_mean")),
                "median": _csv_scalar(summary.get(f"{metric}_median")),
                "std": _csv_scalar(summary.get(f"{metric}_std")),
                "value": "",
            }
        )

    for key in metadata_keys:
        if key in summary:
            rows.append(
                {
                    "section": "run",
                    "metric": key,
                    "mean": "",
                    "median": "",
                    "std": "",
                    "value": _csv_scalar(summary.get(key)),
                }
            )

    fieldnames = ["section", "metric", "mean", "median", "std", "value"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_debug_mask_panel(
    out_path: Path,
    target_rgb: np.ndarray,
    true_mask: np.ndarray,
    pred_rgb: np.ndarray,
    pred_mask: np.ndarray,
) -> None:
    """2×2 layout: target | true mask; generated | pred mask."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Target")
    axes[0, 1].imshow(true_mask, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("True extracted mask")
    axes[1, 0].imshow(pred_rgb)
    axes[1, 0].set_title("Generated")
    axes[1, 1].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Pred extracted mask")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        default=18.0,
        help=(
            "Pixels farther than this Lab L2 distance from the estimated border background are "
            "foreground; if that yields an empty mask, Otsu on normalized distance is used. "
            "Use 0 for Otsu-only (no fixed Lab threshold)."
        ),
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
        "--output-csv",
        type=str,
        default=None,
        help="Write per-sample metrics to this CSV path.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default=None,
        help="Write aggregate summary statistics and run metadata to this CSV path.",
    )
    parser.add_argument(
        "--debug-mask-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for IoU debug artifacts per sample: *_true_mask.png, "
            "*_pred_mask.png, and *_panel.png (2×2: target | true mask; generated | pred mask)."
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
    parser.add_argument(
        "--only-corrupted",
        action="store_true",
        help=(
            "With --corrupt-fraction, only evaluate test indices with corrupted hints "
            "(same disjoint mask as training); summary reflects corrupted-only performance."
        ),
    )

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.corrupt_fraction is not None and not 0.0 <= args.corrupt_fraction <= 1.0:
        parser.error("--corrupt-fraction must be in [0, 1].")

    if args.only_corrupted and args.corrupt_fraction is None:
        parser.error("--only-corrupted requires --corrupt-fraction.")

    seed_everything(args.seed)

    records: List[Dict[str, Any]] = []
    ious: List[float] = []
    roundness_deltas: List[float] = []
    roundness_preds: List[float] = []
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

    if args.only_corrupted:
        eligible = [i for i in range(len(test_ds)) if test_ds.is_corrupted_index(i)]
    else:
        eligible = list(range(len(test_ds)))

    if args.max_samples is not None:
        eligible = eligible[: args.max_samples]

    if not eligible:
        raise RuntimeError("No samples to evaluate (empty eligible index list).")

    for i in tqdm(eligible, desc="eval"):
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
        if debug_prefix is not None:
            save_debug_mask_panel(
                Path(f"{debug_prefix}_panel.png"),
                target_rgb,
                true_mask,
                pred_rgb,
                pred_mask,
            )
        true_roundness = mask_roundness(true_mask)
        pred_roundness = mask_roundness(pred_mask)
        roundness_abs_delta = abs(pred_roundness - true_roundness)
        true_radius = mask_radius(true_mask)
        pred_radius = mask_radius(pred_mask)
        radius_error = (
            abs(pred_radius - true_radius) / true_radius
            if true_radius > 0.0
            else float("nan")
        )
        true_cx, true_cy = mask_center(true_mask)
        pred_cx, pred_cy = mask_center(pred_mask)
        center_dist = np.sqrt((pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2)
        center_error = (
            float(center_dist / true_radius) if true_radius > 0.0 else float("nan")
        )
        circle_color_error, background_color_error, total_color_error = color_errors_from_mask(
            target_rgb, pred_rgb, pred_mask
        )
        ious.append(iou)
        roundness_deltas.append(roundness_abs_delta)
        roundness_preds.append(pred_roundness)
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
        roundness_preds,
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
        summary["only_corrupted"] = bool(args.only_corrupted)
        summary["eval_sample_count"] = len(eligible)
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

    if args.output_csv:
        csv_path = Path(args.output_csv).resolve()
        write_records_csv(csv_path, records)
        print(f"Wrote {csv_path}")

    if args.output_summary_csv:
        summary_csv = Path(args.output_summary_csv).resolve()
        write_summary_csv(summary_csv, summary)
        print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
