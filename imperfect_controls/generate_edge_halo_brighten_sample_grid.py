#!/usr/bin/env python3
"""Generate a sample x model grid for the ``edge_halo_brighten`` corruption.

Rows: the first ``N_SAMPLES`` Fill50K test samples (each with its own prompt
and corrupted control).
Columns: control (corrupted), target, and generated outputs from the
clean-finetuned and combo-finetuned models.
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
N_SAMPLES = 5
SEED = 42
CORRUPTION_KEY = "edge_halo_brighten"

MODEL_COLUMNS: List[Tuple[str, str]] = [
    ("Clean-Finetuned", "clean_finetune/fill50k-step-best-step=000299.ckpt"),
    ("Multiple-Corruption-Finetuned", "combo_finetune/imperfect50-p=0.20-step-best-step=000699.ckpt"),
]

CHECKPOINT_ROOT = _SCRIPT_DIR / "checkpoints"
OUT_DIR = _SCRIPT_DIR / "generated_grids"
OUT_NAME_TEMPLATE = "edge_halo_brighten_first{n:02d}_sample_model_grid.png"

A_PROMPT = ""
N_PROMPT = ""
DDIM_STEPS = 20
GUIDANCE_SCALE = 9.0
ETA = 0.0
STRENGTH = 1.0
ROW_LABEL_WRAP_WIDTH = 22
# --- end editable settings ---


def hint_to_uint8_rgb(hint: np.ndarray) -> np.ndarray:
    return (np.asarray(hint) * 255.0).clip(0, 255).astype(np.uint8)


def target_to_uint8_rgb(jpg: np.ndarray) -> np.ndarray:
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def load_test_samples(n: int) -> List[Tuple[dict, int]]:
    ds = Fill50KDataset(
        split="test",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    if n <= 0:
        raise ValueError(f"N_SAMPLES must be positive, got {n}")
    if n > len(ds):
        raise IndexError(f"N_SAMPLES={n} exceeds test size={len(ds)}")
    return [(ds[i], int(ds.indices[i])) for i in range(n)]


def corrupt_hint(base_hint: np.ndarray, dataset_idx: int) -> np.ndarray:
    fn = CORRUPTION_FUNCS.get(CORRUPTION_KEY)
    if fn is None:
        raise ValueError(
            f"Unknown corruption key {CORRUPTION_KEY!r}; options: {sorted(CORRUPTION_FUNCS)}"
        )
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, dataset_idx, abs(hash(CORRUPTION_KEY)) % 10_000])
    )
    return np.clip(fn(base_hint, rng), 0.0, 1.0).astype(np.float32, copy=False)


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
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help=f"Number of test samples to include (default: {N_SAMPLES})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    seed_everything(SEED)
    device = torch.device("cuda")

    n_samples = int(args.n_samples)
    samples = load_test_samples(n_samples)

    prompts: List[str] = []
    controls: List[np.ndarray] = []
    targets_rgb: List[np.ndarray] = []
    for sample, dataset_idx in samples:
        prompts.append(sample["txt"])
        controls.append(corrupt_hint(sample["hint"], dataset_idx))
        targets_rgb.append(target_to_uint8_rgb(sample["jpg"]))

    # Cache generated images: generated[(row_idx, model_label)] -> RGB uint8
    generated: Dict[Tuple[int, str], np.ndarray] = {}

    for model_label, rel_ckpt in MODEL_COLUMNS:
        ckpt_path = (CHECKPOINT_ROOT / rel_ckpt).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_label}: {ckpt_path}")
        print(f"Loading {model_label}: {ckpt_path}")
        model, sampler = load_model_and_sampler(ckpt_path, device)

        for r in range(n_samples):
            g = generate_image(
                model,
                sampler,
                controls[r],
                prompts[r],
                a_prompt=A_PROMPT,
                n_prompt=N_PROMPT,
                ddim_steps=DDIM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                eta=ETA,
                strength=STRENGTH,
            )
            generated[(r, model_label)] = g

        del sampler
        del model
        torch.cuda.empty_cache()

    columns = ["Control", "Target"] + [label for label, _ in MODEL_COLUMNS]
    n_rows = n_samples
    n_cols = len(columns)

    fig_w = max(2.8 * n_cols, 12.0)
    fig_h = max(2.6 * n_rows, 8.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r in range(n_rows):
        control_rgb = hint_to_uint8_rgb(controls[r])
        for c, col_name in enumerate(columns):
            ax = axes[r, c]
            if col_name == "Control":
                img = control_rgb
            elif col_name == "Target":
                img = targets_rgb[r]
            else:
                img = generated[(r, col_name)]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_name, fontsize=10)

    fig.suptitle("Generated Images for Brightness Corruption", fontsize=12)
    fig.tight_layout(rect=[0.16, 0, 1, 0.95])

    for r in range(n_rows):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        label = textwrap.fill(prompts[r], width=ROW_LABEL_WRAP_WIDTH)
        fig.text(
            bbox.x0 - 0.02,
            y_center,
            label,
            va="center",
            ha="right",
            fontsize=9,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME_TEMPLATE.format(n=n_samples)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote grid: {out_path}")


if __name__ == "__main__":
    main()
