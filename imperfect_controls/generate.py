"""Generate images from a trained ControlNet checkpoint using Fill50K test inputs.

By default controls (hints) are clean ``Fill50KDataset``. Pass ``--corrupt-fraction`` to use
``DisjointCorruptFill50KDataset`` (same imperfect-controls setup as training).
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

for p in (_REPO_ROOT, _SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from share import *
import config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from dataset import Fill50KDataset
from generate_image import generate_image
from imperfect_fill50k_dataset import (
    CORRUPTION_FUNCS,
    DisjointCorruptFill50KDataset,
)


def save_image(arr, path):
    Image.fromarray(arr).save(path)


def hint_to_uint8_rgb(hint):
    """Dataset hint is float32 RGB in [0, 1], shape H×W×3."""
    return (np.asarray(hint) * 255.0).clip(0, 255).astype(np.uint8)


def target_to_uint8_rgb(jpg):
    """Dataset target (jpg key) is float32 RGB in [-1, 1], shape H×W×3."""
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def save_comparison_plot(control_rgb, target_rgb, generated_rgb, prompt, path):
    """Save control, target, and generated images in one figure with the prompt as title."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ("Control", "Target", "Generated")
    for ax, img, title in zip(axes, (control_rgb, target_rgb, generated_rgb), titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    wrapped = textwrap.fill(prompt, width=96)
    fig.suptitle(wrapped, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--ddim-steps", type=int, default=20, help="number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=9.0, help="how strongly the model follows the text prompt")
    parser.add_argument("--eta", type=float, default=0.0, help="randomness in DDIM sampling (0 for deterministic sampling)")
    parser.add_argument("--strength", type=float, default=1.0, help="how strongly ControlNet follows the control image")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--a-prompt",
        default="",
        help="additional prompt to guide the image generation"
    )
    parser.add_argument(
        "--n-prompt",
        default="",
        help="negative prompt to guide the image generation"
    )
    parser.add_argument(
        "--corrupt-fraction",
        type=float,
        default=None,
        metavar="P",
        help=(
            "If set (0..1), use DisjointCorruptFill50KDataset (corrupted hints). "
            "If omitted, use plain Fill50KDataset (clean hints)."
        ),
    )
    parser.add_argument(
        "--corruption-type",
        type=str,
        default="edge_segment_remove",
        help="Corruption name when --corrupt-fraction is set.",
    )
    parser.add_argument(
        "--corruption-types",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Optional comma-separated list of corruption names (e.g. 'blur,noise'). "
            "When set with --corrupt-fraction, corrupted indices are split as evenly "
            "as possible across these types. If omitted, uses --corruption-type."
        ),
    )
    parser.add_argument(
        "--only-corrupted",
        action="store_true",
        help=(
            "With --corrupt-fraction, only use test indices whose hints are corrupted "
            "(same disjoint split as training). Ignores clean test samples for this run."
        ),
    )

    args = parser.parse_args()

    if args.only_corrupted and args.corrupt_fraction is None:
        raise ValueError("--only-corrupted requires --corrupt-fraction (imperfect dataset).")

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    corruption_types = None
    if args.corrupt_fraction is not None and args.corruption_types is not None:
        raw = [s.strip() for s in args.corruption_types.split(",")]
        corruption_types = [s for s in raw if s]
        if len(corruption_types) == 0:
            raise ValueError("--corruption-types must be a comma-separated list of corruption names")
        bad = [c for c in corruption_types if c not in CORRUPTION_FUNCS]
        if bad:
            raise ValueError(
                "unknown --corruption-types entries: {}; choose from: {}".format(
                    ", ".join(bad),
                    ", ".join(sorted(CORRUPTION_FUNCS.keys())),
                )
            )

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation script.")

    device = torch.device("cuda")
    seed_everything(args.seed)

    # Load test split. Keep this seed consistent with training / evaluate.py.
    if args.corrupt_fraction is not None:
        if not 0.0 <= args.corrupt_fraction <= 1.0:
            raise ValueError("--corrupt-fraction must be in [0, 1]")
        test_ds = DisjointCorruptFill50KDataset(
            split="test",
            corrupt_fraction=args.corrupt_fraction,
            corruption_type=args.corruption_type,
            corruption_types=corruption_types,
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
        raise RuntimeError("Test split is empty (check Fill50k prompt.json and split ratios).")

    if args.only_corrupted:
        eligible = [i for i in range(len(test_ds)) if test_ds.is_corrupted_index(i)]
    else:
        eligible = list(range(len(test_ds)))

    if not eligible:
        raise RuntimeError("No samples to generate (empty eligible index list).")

    n = min(args.num_samples, len(eligible))
    if args.num_samples > len(eligible):
        print(
            f"Note: --num-samples={args.num_samples} exceeds eligible count {len(eligible)}; using {n}."
        )
    selected = eligible[:n]

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        run_name = re.sub(r"[^\w.-]+", "_", ckpt.parent.name)[:64]
        if args.corrupt_fraction is not None:
            cf_tag = str(args.corrupt_fraction).replace(".", "p")
            if corruption_types is not None:
                ct_tag = "+".join(corruption_types)
            else:
                ct_tag = args.corruption_type
            run_name = f"{run_name}_imperfect_{cf_tag}_{ct_tag}"[:120]
            if args.only_corrupted:
                run_name = f"{run_name}_corrupted_only"[:120]
        out_dir = (_SCRIPT_DIR / "generated" / run_name).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = create_model(str(_REPO_ROOT / "models/cldm_v15.yaml")).to(device)
    model.load_state_dict(load_state_dict(str(ckpt), location=str(device)))
    # Match imperfect_controls/train.py (not stored in weight checkpoints).
    model.sd_locked = True
    model.eval()

    sampler = DDIMSampler(model)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    prompts_file = out_dir / "prompts.txt"

    with open(prompts_file, "w", encoding="utf-8") as f:
        f.write("local_i\tdataset_i\tsample_dir\thint_corrupt\tprompt\n")

        for i in selected:
            item = test_ds[i]
            dataset_i = test_ds.indices[i]
            sample_dir = out_dir / f"test_idx_{dataset_i:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            prompt = item["txt"]
            with open(sample_dir / "prompt.txt", "w", encoding="utf-8") as pf:
                pf.write(prompt)

            control_rgb = hint_to_uint8_rgb(item["hint"])
            target_rgb = target_to_uint8_rgb(item["jpg"])
            save_image(control_rgb, sample_dir / "control.png")
            save_image(target_rgb, sample_dir / "target.png")

            x = generate_image(
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

            save_image(x, sample_dir / "generated.png")
            save_comparison_plot(
                control_rgb, target_rgb, x, prompt, sample_dir / "comparison.png"
            )

            hint_corrupt = (
                int(test_ds.is_corrupted_index(i))
                if hasattr(test_ds, "is_corrupted_index")
                else 0
            )
            f.write(f"{i}\t{dataset_i}\t{sample_dir.name}\t{hint_corrupt}\t{prompt}\n")

    print(f"Wrote {n} sample folders under {out_dir}")


if __name__ == "__main__":
    main()