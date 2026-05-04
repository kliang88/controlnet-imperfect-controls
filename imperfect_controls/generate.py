"""Generate images from a trained ControlNet checkpoint using Fill50K test inputs."""

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

for p in (_REPO_ROOT, _SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from share import *
import config

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from dataset import Fill50KDataset
from generate_image import generate_image


def save_image(arr, path):
    Image.fromarray(arr).save(path)


def hint_to_uint8_rgb(hint):
    """Dataset hint is float32 RGB in [0, 1], shape H×W×3."""
    return (np.asarray(hint) * 255.0).clip(0, 255).astype(np.uint8)


def target_to_uint8_rgb(jpg):
    """Dataset target (jpg key) is float32 RGB in [-1, 1], shape H×W×3."""
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


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
        default="best quality, extremely detailed",
        help="additional prompt to guide the image generation"
    )
    parser.add_argument(
        "--n-prompt",
        default="low quality, blurry, distorted",
        help="negative prompt to guide the image generation"
    )

    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation script.")

    device = torch.device("cuda")
    seed_everything(args.seed)

    # Load test split. Keep this seed consistent with training.
    test_ds = Fill50KDataset(
        split="test",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty (check Fill50k prompt.json and split ratios).")

    n = min(args.num_samples, len(test_ds))
    if args.num_samples > len(test_ds):
        print(
            f"Note: --num-samples={args.num_samples} exceeds test split size {len(test_ds)}; using {n}."
        )

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        run_name = re.sub(r"[^\w.-]+", "_", ckpt.parent.name)[:64]
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
        f.write("local_i\tdataset_i\tsample_dir\tprompt\n")

        for i in range(n):
            item = test_ds[i]
            dataset_i = test_ds.indices[i]
            sample_dir = out_dir / f"test_idx_{dataset_i:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            prompt = item["txt"]
            with open(sample_dir / "prompt.txt", "w", encoding="utf-8") as pf:
                pf.write(prompt)

            save_image(hint_to_uint8_rgb(item["hint"]), sample_dir / "control.png")
            save_image(target_to_uint8_rgb(item["jpg"]), sample_dir / "target.png")

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

            f.write(f"{i}\t{dataset_i}\t{sample_dir.name}\t{prompt}\n")

    print(f"Wrote {n} sample folders under {out_dir}")


if __name__ == "__main__":
    main()