#!/usr/bin/env python3
"""Plot train and validation loss curves for one model log folder.

Reads CSVs produced in:
  log_outputs/<model_name>/

Expected files:
  - train_loss_step.csv  (step,value,...)
  - val_loss.csv         (step,value,...)
"""

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_OUTPUTS_DIR = SCRIPT_DIR.parent / "log_outputs"

# --- editable defaults ---
MODEL_NAME = "combo_finetune"
OUT_DIR = SCRIPT_DIR / "eval_plots"
OUT_FILENAME_COMBINED = "combo_finetune_loss_combined.png"
OUT_FILENAME_TRAIN = "combo_finetune_loss_train.png"
OUT_FILENAME_VAL = "combo_finetune_loss_val.png"
TITLE = "Train vs Validation Loss for Multiple-Corruption-Finetuned Model"
TITLE_TRAIN = "Train Loss for Multiple-Corruption-Finetuned Model"
TITLE_VAL = "Validation Loss for Multiple-Corruption-Finetuned Model"
X_LABEL = "Step"
Y_LABEL = "Loss"
TRAIN_LABEL = "Train Loss"
VAL_LABEL = "Validation Loss"
# --- end editable defaults ---


def read_step_value_csv(path: Path) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_raw = (row.get("step") or "").strip()
            value_raw = (row.get("value") or "").strip()
            if not step_raw or not value_raw:
                continue
            xs.append(float(step_raw))
            ys.append(float(value_raw))
    return xs, ys


def main() -> None:
    model_dir = LOG_OUTPUTS_DIR / MODEL_NAME
    train_csv = model_dir / "train_loss_step.csv"
    val_csv = model_dir / "val_loss.csv"

    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing file: {train_csv}")
    if not val_csv.is_file():
        raise FileNotFoundError(f"Missing file: {val_csv}")

    train_x, train_y = read_step_value_csv(train_csv)
    val_x, val_y = read_step_value_csv(val_csv)

    model_out_dir = OUT_DIR / MODEL_NAME
    model_out_dir.mkdir(parents=True, exist_ok=True)
    out_path_combined = model_out_dir / OUT_FILENAME_COMBINED
    out_path_train = model_out_dir / OUT_FILENAME_TRAIN
    out_path_val = model_out_dir / OUT_FILENAME_VAL

    # Combined train + validation plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(train_x, train_y, marker="o", linewidth=2, alpha=0.85, label=TRAIN_LABEL)
    ax.plot(val_x, val_y, marker="s", linewidth=2, alpha=0.9, label=VAL_LABEL)

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Train-only plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(train_x, train_y, marker="o", linewidth=2, alpha=0.85, label=TRAIN_LABEL)
    ax.set_title(TITLE_TRAIN)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path_train, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Validation-only plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(val_x, val_y, marker="s", linewidth=2, alpha=0.9, label=VAL_LABEL)
    ax.set_title(TITLE_VAL)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path_val, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path_combined}")
    print(f"Wrote {out_path_train}")
    print(f"Wrote {out_path_val}")


if __name__ == "__main__":
    main()
