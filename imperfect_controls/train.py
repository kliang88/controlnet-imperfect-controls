"""Train ControlNet on Fill50k.

Example:
  python imperfect_controls/train.py --run-name run1
    -> checkpoints under imperfect_controls/checkpoints/run1/
  python imperfect_controls/train.py --max-steps 50000 --batch-size 8 --learning-rate 2e-5
    -> hyperparameters override the Configs defaults.
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
    help="DataLoader batch size (default 4).",
)
_parser.add_argument(
    "--learning-rate",
    type=float,
    default=None,
    metavar="LR",
    help="AdamW learning rate (default 1e-5).",
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
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )


# Configs
resume_path = str(_REPO_ROOT / "models/control_sd15_ini.ckpt")
_checkpoint_base = _SCRIPT_DIR / "checkpoints"
checkpoint_dir = str(_checkpoint_base / _RUN_SUBDIR) if _RUN_SUBDIR else str(_checkpoint_base)
batch_size = 4
logger_freq = 300
# Save a checkpoint every N optimizer steps.
checkpoint_every_n_train_steps = 500
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
if _cli.learning_rate is not None:
    if _cli.learning_rate <= 0:
        sys.exit("--learning-rate must be > 0")
    learning_rate = _cli.learning_rate


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(str(_REPO_ROOT / "models/cldm_v15.yaml")).cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
train_dataset = Fill50KDataset(
    split="train",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
val_dataset = Fill50KDataset(
    split="val",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
test_dataset = Fill50KDataset(
    split="test",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
print(f"Split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
logger = ImageLogger(batch_frequency=logger_freq)
# keep the 3 lowest val/loss checkpoints and the latest checkpoint
checkpoint_cb = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="fill50k-step={step:06d}",
    save_last=True, # also save the latest checkpoint
    save_weights_only=True,
    save_top_k=3,
    monitor="val/loss",
    mode="min",
    every_n_train_steps=checkpoint_every_n_train_steps,
)
_trainer_kw = dict(
    gpus=1,
    precision=32,
    callbacks=[logger, checkpoint_cb],
    val_check_interval=checkpoint_every_n_train_steps,
)
if max_steps >= 0:
    _trainer_kw["max_steps"] = max_steps
trainer = pl.Trainer(**_trainer_kw)


# Train
trainer.fit(model, train_loader, val_loader)
# trainer.test(model, dataloaders=test_loader, ckpt_path="best") # no meaningful test step yet, inherited from Lightning's test() method
