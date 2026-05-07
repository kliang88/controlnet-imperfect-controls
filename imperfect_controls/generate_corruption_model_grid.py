#!/usr/bin/env python3
"""Generate a corruption x model grid for one Fill50K test sample.

Rows: control corruption type (clean, blur, noise_edge_speckle, downsample).
Columns: control, target, and generated outputs from selected fine-tuned models.
Figure title: sample text prompt.
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from share import *  # noqa: F401,F403
import config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from dataset import Fill50KDataset
from generate_image import generate_image
from imperfect_fill50k_dataset import CORRUPTION_FUNCS


# --- editable settings ---
TEST_LOCAL_INDEX = 0
SEED = 42

# (row label shown in figure, corruption key)
ROW_SPECS: List[Tuple[str, str]] = [
    ("Clean", "clean"),
    ("Blurred", "blur"),
    ("Noisy", "noise_edge_speckle"),
    ("Downsampled", "downsample"),
]

MODEL_COLUMNS: List[Tuple[str, str]] = [
    ("Clean-Finetuned", "clean_finetune/fill50k-step-best-step=000299.ckpt"),
    ("Blur-Finetuned", "blur_finetune/imperfect50-p=0.20-step-best-step=000499.ckpt"),
    ("Noise-Finetuned", "noise_edge_speckle_finetune_2/imperfect50-p=0.20-step-best-step=000299.ckpt"),
    ("Downsample-Finetuned", "downsample_finetune/imperfect50-p=0.20-step-best-step=000349.ckpt"),
    ("Multiple-Corruption-Finetuned", "combo_finetune/imperfect50-p=0.20-step-best-step=000699.ckpt"),
]

CHECKPOINT_ROOT = _SCRIPT_DIR / "checkpoints"
OUT_DIR = _SCRIPT_DIR / "generated_grids"
OUT_NAME_TEMPLATE = "test_local_{index:04d}_corruption_model_grid.png"

A_PROMPT = ""
N_PROMPT = ""
DDIM_STEPS = 20
GUIDANCE_SCALE = 9.0
ETA = 0.0
STRENGTH = 1.0
# --- end editable settings ---


def hint_to_uint8_rgb(hint: np.ndarray) -> np.ndarray:
    return (np.asarray(hint) * 255.0).clip(0, 255).astype(np.uint8)


def target_to_uint8_rgb(jpg: np.ndarray) -> np.ndarray:
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def load_sample(local_idx: int) -> Tuple[dict, int]:
    ds = Fill50KDataset(
        split="test",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    if not 0 <= local_idx < len(ds):
        raise IndexError(f"TEST_LOCAL_INDEX out of range: {local_idx} (test size={len(ds)})")
    return ds[local_idx], int(ds.indices[local_idx])


def build_corrupted_controls(base_hint: np.ndarray, dataset_idx: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for row_label, corruption_key in ROW_SPECS:
        if corruption_key == "clean":
            out[row_label] = np.clip(base_hint, 0.0, 1.0).astype(np.float32, copy=True)
            continue
        fn = CORRUPTION_FUNCS.get(corruption_key)
        if fn is None:
            raise ValueError(
                f"Unknown corruption key {corruption_key!r}; options: {sorted(CORRUPTION_FUNCS)}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([SEED, dataset_idx, abs(hash(corruption_key)) % 10_000])
        )
        out[row_label] = np.clip(fn(base_hint, rng), 0.0, 1.0).astype(np.float32, copy=False)
    return out


def load_model_and_sampler(ckpt_path: Path, device: torch.device):
    model = create_model(str(_REPO_ROOT / "models/cldm_v15.yaml")).to(device)
    model.load_state_dict(load_state_dict(str(ckpt_path), location=str(device)))
    model.sd_locked = True
    model.eval()
    sampler = DDIMSampler(model)
    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)
    return model, sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-local-index",
        type=int,
        default=TEST_LOCAL_INDEX,
        help=f"Test-split local index (default: {TEST_LOCAL_INDEX})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    seed_everything(SEED)
    device = torch.device("cuda")

    test_local_index = args.test_local_index
    sample, dataset_idx = load_sample(test_local_index)
    prompt = sample["txt"]
    base_hint = sample["hint"]
    target_rgb = target_to_uint8_rgb(sample["jpg"])
    controls_by_row = build_corrupted_controls(base_hint, dataset_idx)

    # Cache generated images: generated[(row_name, model_label)] -> RGB uint8
    generated: Dict[Tuple[str, str], np.ndarray] = {}

    for model_label, rel_ckpt in MODEL_COLUMNS:
        ckpt_path = (CHECKPOINT_ROOT / rel_ckpt).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_label}: {ckpt_path}")
        print(f"Loading {model_label}: {ckpt_path}")
        model, sampler = load_model_and_sampler(ckpt_path, device)

        for row_label, _ in ROW_SPECS:
            g = generate_image(
                model,
                sampler,
                controls_by_row[row_label],
                prompt,
                a_prompt=A_PROMPT,
                n_prompt=N_PROMPT,
                ddim_steps=DDIM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                eta=ETA,
                strength=STRENGTH,
            )
            generated[(row_label, model_label)] = g

        del sampler
        del model
        torch.cuda.empty_cache()

    columns = ["Control", "Target"] + [label for label, _ in MODEL_COLUMNS]
    n_rows = len(ROW_SPECS)
    n_cols = len(columns)

    fig_w = max(2.8 * n_cols, 14.0)
    fig_h = max(2.6 * n_rows, 8.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, (row_label, _) in enumerate(ROW_SPECS):
        control_rgb = hint_to_uint8_rgb(controls_by_row[row_label])
        for c, col_name in enumerate(columns):
            ax = axes[r, c]
            if col_name == "Control":
                img = control_rgb
            elif col_name == "Target":
                img = target_rgb
            else:
                img = generated[(row_label, col_name)]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_name, fontsize=10)

    fig.suptitle(textwrap.fill(prompt, width=120), fontsize=12)
    # First lay out the axes and reserve more left margin.
    fig.tight_layout(rect=[0.12, 0, 1, 0.95])
    # Then draw row labels after layout is finalized.
    for r, (row_label, _) in enumerate(ROW_SPECS):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            bbox.x0 - 0.02,
            y_center,
            row_label,
            va="center",
            ha="right",
            fontsize=10,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME_TEMPLATE.format(index=test_local_index)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote grid: {out_path}")


if __name__ == "__main__":
    main()
