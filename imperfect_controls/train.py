"""Train ControlNet on Fill50k (plain or disjoint corrupted controls).

Examples:
  python imperfect_controls/train.py --run-name run1
  python imperfect_controls/train.py --max-steps 50000 --batch-size 8 --learning-rate 2e-5
  python imperfect_controls/train.py --corrupt-fraction 0.5 --run-name imperfect50
  python imperfect_controls/train.py --corrupt-fraction 0.5 --num-gpus 1 --max-steps 10
  python imperfect_controls/train.py --finetune-from path/to/weights.ckpt --run-name ft1
  python imperfect_controls/train.py --continue-from-checkpoint last --run-name run1
"""
import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    metavar="NAME",
    help="Optional subdirectory for checkpoints (e.g. run1 -> imperfect_controls/checkpoints/run1/).",
)
_parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    metavar="N",
    help="Stop after N optimizer steps (default 10,000). Use -1 for no step cap (Lightning default).",
)
_parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Per-GPU DataLoader batch size (default 4). "
        "Gradient accumulation is chosen so global batch matches --effective-batch-size."
    ),
)
_parser.add_argument(
    "--effective-batch-size",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Global samples per optimizer step (default: --batch-size * --num-gpus, i.e. no grad accumulation). Must be divisible by "
        "(--batch-size * --num-gpus); sets Trainer accumulate_grad_batches."
    ),
)
_parser.add_argument(
    "--learning-rate",
    type=float,
    default=None,
    metavar="LR",
    help="AdamW learning rate (default 1e-5).",
)
_parser.add_argument(
    "--corrupt-fraction",
    type=float,
    default=None,
    metavar="P",
    help=(
        "If set (0..1), train on DisjointCorruptFill50KDataset with this fraction "
        "of corrupted controls per split; omit for plain Fill50KDataset."
    ),
)
_parser.add_argument(
    "--corruption-type",
    type=str,
    default="edge_segment_remove",
    metavar="NAME",
    help="Corruption name when --corrupt-fraction is set (default edge_segment_remove).",
)
_parser.add_argument(
    "--corruption-types",
    type=str,
    default=None,
    metavar="LIST",
    help=(
        "Optional comma-separated list of corruption names (e.g. 'blur,noise'). When set with "
        "--corrupt-fraction, the corrupted subset is split as evenly as possible across these types. "
        "If omitted, uses --corruption-type."
    ),
)
_parser.add_argument(
    "--num-gpus",
    type=int,
    default=None,
    metavar="N",
    help="Number of GPUs (default 2). Use 1 for single-GPU smoke tests.",
)
_parser.add_argument(
    "--val-every-n-steps",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Run validation every N optimizer (global) steps (default 500)."
    ),
)
_parser.add_argument(
    "--weights-save-every-n-steps",
    type=int,
    default=None,
    metavar="N",
    help=(
        "If set, save weights-only checkpoints every N optimizer/global steps via ModelCheckpoint "
        "every_n_train_steps. If omitted, only best (val/loss) and last.ckpt are saved."
    ),
)
_parser.add_argument(
    "--finetune-from",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "After the default SD init (models/control_sd15_ini.ckpt), load weights from this "
        ".ckpt or .safetensors for finetuning (strict=False). Starts a fresh optimizer schedule "
        "unless you use --continue-from-checkpoint (not combinable with that flag)."
    ),
)
_parser.add_argument(
    "--continue-from-checkpoint",
    "--resume-from-checkpoint",
    type=str,
    default=None,
    dest="continue_from_checkpoint",
    metavar="PATH",
    help=(
        "Resume full Lightning training state (weights, optimizer, global step) from this .ckpt. "
        "Use the literal 'last' for <checkpoints>/<run-name>/last.ckpt. Alias: --resume-from-checkpoint."
    ),
)
_cli = _parser.parse_args()
if _cli.run_name is not None:
    _rn = _cli.run_name.strip()
    if not _rn or not re.fullmatch(r"[A-Za-z0-9._-]+", _rn):
        sys.exit("--run-name must be non-empty and only use letters, digits, ., _, -")
    _RUN_SUBDIR = _rn
else:
    _RUN_SUBDIR = None

from share import *

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import Fill50KDataset
from imperfect_fill50k_dataset import CORRUPTION_FUNCS, DisjointCorruptFill50KDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def _exit_if_no_cuda(num_gpus: int) -> None:
    if torch.cuda.is_available():
        if torch.cuda.device_count() < num_gpus:
            sys.exit(
                "Requested --num-gpus {} but only {} CUDA device(s) are visible "
                "(nvidia-smi / CUDA_VISIBLE_DEVICES).".format(
                    num_gpus, torch.cuda.device_count()
                )
            )
        return
    lines = [
        "CUDA is not available, but this training script needs an NVIDIA GPU "
        "(the ControlNet stack moves models to CUDA).",
        "",
        "What to try:",
        "  • Cluster: run inside a GPU job (sbatch/salloc with GPUs), not only on a login node.",
        "  • Workstation: `nvidia-smi` should work; install a CUDA-enabled PyTorch build for your driver.",
        "  • Check: `python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"`.",
        "  • If a GPU exists but is hidden, set CUDA_VISIBLE_DEVICES (e.g. to 0).",
    ]
    if getattr(torch.version, "cuda", None) is None:
        lines += [
            "",
            "Your PyTorch build looks CPU-only (torch.version.cuda is None). If nvidia-smi already",
            "shows a GPU in this job, reinstall PyTorch+torchvision with CUDA in the same conda env, e.g.:",
            "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            "  (pick the cu* URL that matches your cluster docs; older ControlNet often used torch 1.12 + cu116.)",
        ]
    sys.exit("\n".join(lines) + "\n")


def main():
    num_gpus = _cli.num_gpus if _cli.num_gpus is not None else 2
    if num_gpus < 1:
        sys.exit("--num-gpus must be >= 1")
    _exit_if_no_cuda(num_gpus)
    if _cli.finetune_from and _cli.continue_from_checkpoint:
        sys.exit("Use either --finetune-from or --continue-from-checkpoint, not both.")
    # Configs
    resume_path = str(_REPO_ROOT / "models/control_sd15_ini.ckpt")
    _checkpoint_base = _SCRIPT_DIR / "checkpoints"
    checkpoint_dir = str(_checkpoint_base / _RUN_SUBDIR) if _RUN_SUBDIR else str(_checkpoint_base)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print("Checkpoints directory: {}".format(checkpoint_dir))
    batch_size = 4  # per-GPU micro-batch; accumulation targets effective_batch_size
    logger_freq = 300
    val_every_n_steps = 500
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    max_steps = 10_000  # optimizer steps cap; override with --max-steps

    if _cli.max_steps is not None:
        max_steps = _cli.max_steps
    if _cli.batch_size is not None:
        if _cli.batch_size < 1:
            sys.exit("--batch-size must be >= 1")
        batch_size = _cli.batch_size
    if _cli.effective_batch_size is not None:
        if _cli.effective_batch_size < 1:
            sys.exit("--effective-batch-size must be >= 1")
        effective_batch_size = _cli.effective_batch_size
    else:
        # Default to no gradient accumulation unless explicitly requested.
        effective_batch_size = batch_size * num_gpus
    if _cli.learning_rate is not None:
        if _cli.learning_rate <= 0:
            sys.exit("--learning-rate must be > 0")
        learning_rate = _cli.learning_rate
    if _cli.val_every_n_steps is not None:
        if _cli.val_every_n_steps < 1:
            sys.exit("--val-every-n-steps must be >= 1")
        val_every_n_steps = _cli.val_every_n_steps
    weights_save_every_n_steps = None
    if _cli.weights_save_every_n_steps is not None:
        if _cli.weights_save_every_n_steps < 1:
            sys.exit("--weights-save-every-n-steps must be >= 1")
        weights_save_every_n_steps = _cli.weights_save_every_n_steps

    corruption_types = None
    if _cli.corrupt_fraction is not None:
        if not 0.0 <= _cli.corrupt_fraction <= 1.0:
            sys.exit("--corrupt-fraction must be between 0 and 1")
        if _cli.corruption_types is not None:
            raw = [s.strip() for s in _cli.corruption_types.split(",")]
            corruption_types = [s for s in raw if s]
            if len(corruption_types) == 0:
                sys.exit("--corruption-types must be a comma-separated list of corruption names")
            bad = [c for c in corruption_types if c not in CORRUPTION_FUNCS]
            if bad:
                sys.exit(
                    "unknown --corruption-types entries: {}; choose from: {}".format(
                        ", ".join(bad),
                        ", ".join(sorted(CORRUPTION_FUNCS.keys())),
                    )
                )
        else:
            if _cli.corruption_type not in CORRUPTION_FUNCS:
                sys.exit(
                    "unknown --corruption-type; choose one of: {}".format(
                        ", ".join(sorted(CORRUPTION_FUNCS.keys()))
                    )
                )

    micro_per_optimizer_step = batch_size * num_gpus
    if effective_batch_size % micro_per_optimizer_step != 0:
        sys.exit(
            "Effective batch size {} requires (--batch-size * --num-gpus) to divide it; got {} * {} = {}.".format(
                effective_batch_size,
                batch_size,
                num_gpus,
                micro_per_optimizer_step,
            )
        )
    accumulate_grad_batches = effective_batch_size // micro_per_optimizer_step
    val_check_interval_batches = val_every_n_steps * accumulate_grad_batches
    print(
        "Batch: per_gpu={} gpus={} accumulate_grad_batches={} -> effective global batch {}".format(
            batch_size,
            num_gpus,
            accumulate_grad_batches,
            effective_batch_size,
        )
    )
    _wmsg = (
        "weights-only checkpoint every_n_train_steps={}".format(weights_save_every_n_steps)
        if weights_save_every_n_steps is not None
        else "periodic weights-only checkpoints disabled (best + last only)"
    )
    print(
        "Validation every {} optimizer steps -> val_check_interval={} training batches; {}".format(
            val_every_n_steps,
            val_check_interval_batches,
            _wmsg,
        )
    )

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(str(_REPO_ROOT / "models/cldm_v15.yaml")).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    if _cli.finetune_from:
        _ft = Path(_cli.finetune_from).expanduser()
        if not _ft.is_file():
            sys.exit("Finetune checkpoint not found: {}".format(_ft))
        model.load_state_dict(load_state_dict(str(_ft), location="cpu"), strict=False)
        print("Finetune: loaded weights on top of init from {}".format(_ft))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    _split_kw = dict(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    if _cli.corrupt_fraction is not None:
        _ds_kw = dict(
            corrupt_fraction=_cli.corrupt_fraction,
            corruption_type=_cli.corruption_type,
            corruption_types=corruption_types,
            **_split_kw,
        )
        train_dataset = DisjointCorruptFill50KDataset(split="train", **_ds_kw)
        val_dataset = DisjointCorruptFill50KDataset(split="val", **_ds_kw)
        test_dataset = DisjointCorruptFill50KDataset(split="test", **_ds_kw)
    else:
        train_dataset = Fill50KDataset(split="train", **_split_kw)
        val_dataset = Fill50KDataset(split="val", **_split_kw)
        test_dataset = Fill50KDataset(split="test", **_split_kw)
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size, shuffle=False)
    print(
        "Split sizes: train={}, val={}, test={}".format(
            len(train_dataset), len(val_dataset), len(test_dataset)
        )
    )
    if _cli.corrupt_fraction is not None:
        n_tr = len(train_dataset)
        n_corrupt_tr = sum(
            1 for i in range(n_tr) if train_dataset.is_corrupted_index(i)
        )
        ct_msg = (
            ",".join(corruption_types)
            if corruption_types is not None
            else _cli.corruption_type
        )
        print(
            "Imperfect dataset: corrupt_fraction={} type(s)={} | train corrupted indices: {}/{}".format(
                _cli.corrupt_fraction,
                ct_msg,
                n_corrupt_tr,
                n_tr,
            )
        )
    logger = ImageLogger(batch_frequency=logger_freq)
    # Optional weights-only every N steps; always keep best val/loss + last.ckpt (resume).
    _ckpt_tag = (
        "imperfect50-p={:.2f}-step".format(_cli.corrupt_fraction)
        if _cli.corrupt_fraction is not None
        else "fill50k-step"
    )
    _callbacks = [logger]
    if weights_save_every_n_steps is not None:
        _callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="weights-{}={{step:06d}}".format(_ckpt_tag),
                every_n_train_steps=weights_save_every_n_steps,
                save_weights_only=True,
                save_last=False,
                save_top_k=-1,
                monitor=None,
            )
        )
    best_and_last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{}-best-{{step:06d}}".format(_ckpt_tag),
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    _callbacks.append(best_and_last_cb)
    _trainer_kw = dict(
        gpus=num_gpus,
        precision=32,
        callbacks=_callbacks,
        val_check_interval=val_check_interval_batches,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    if num_gpus > 1:
        _trainer_kw["strategy"] = "ddp"
    if max_steps >= 0:
        _trainer_kw["max_steps"] = max_steps
    trainer = pl.Trainer(**_trainer_kw)

    ckpt_path = None
    if _cli.continue_from_checkpoint:
        _c = _cli.continue_from_checkpoint.strip()
        if _c.lower() == "last":
            ckpt_path = str(Path(checkpoint_dir) / "last.ckpt")
        else:
            ckpt_path = str(Path(_c).expanduser())
        if not Path(ckpt_path).is_file():
            sys.exit("Continue checkpoint not found: {}".format(ckpt_path))
        print("Continuing training from checkpoint: {}".format(ckpt_path))

    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    # trainer.test(model, dataloaders=test_loader, ckpt_path="best") # no meaningful test step yet, inherited from Lightning's test() method

if __name__ == "__main__":
    main()