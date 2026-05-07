#!/usr/bin/env python3
"""Create line plots from eval_tables/downsample_levels.csv.

Produces one plot per metric:
- mask_iou
- center_error
- radius_error

X-axis is downsampling factor (0, 8, 16, 24). Y-axis is median.
Error bars show Q1/Q3 around the median.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "eval_tables/downsample_levels.csv"
OUT_DIR = SCRIPT_DIR / "eval_plots"

METRICS = ("mask_iou", "center_error", "radius_error")
COLUMN_BY_METRIC = {
    "mask_iou": "mask_iou med (Q1,Q3)",
    "center_error": "center_error med (Q1,Q3)",
    "radius_error": "radius_error med (Q1,Q3)",
}
TRAIN_MAP = {
    "clean": "clean-trained",
    "downsample x16": "downsample-trained",
}
SERIES_ORDER = ("clean-trained", "downsample-trained")

# --- editable plot text ---
X_LABEL = "Downsampling Factor"
Y_LABEL_BY_METRIC = {
    "mask_iou": "IOU",
    "center_error": "Center Error",
    "radius_error": "Radius Error",
}
TITLE_BY_METRIC = {
    "mask_iou": "IOU vs Downsampling Factor",
    "center_error": "Center Error vs Downsampling Factor",
    "radius_error": "Radius Error vs Downsampling Factor",
}
# Optional x-axis ticks; set to None to use matplotlib defaults.
X_TICKS = [0, 8, 16, 24]
# --- end editable plot text ---

# --- publication-style formatting ---
FIGSIZE = (6.2, 4.0)
SAVE_DPI = 300
LINE_WIDTH = 2.2
MARKER_SIZE = 6.5
CAP_SIZE = 3.5
LINE_ALPHA = 0.82
GRID_ALPHA = 0.2
LEGEND_LOC = "best"
SERIES_COLORS = {
    "clean-trained": "#1f4e79",
    "downsample-trained": "#b03a2e",
}
SERIES_MARKERS = {
    "clean-trained": "o",
    "downsample-trained": "s",
}
SERIES_ZORDER = {
    "clean-trained": 3,
    "downsample-trained": 2,
}
# --- end publication-style formatting ---

_MED_Q_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*$"
)


def parse_factor(eval_label: str) -> int:
    eval_label = eval_label.strip().lower()
    if eval_label == "clean":
        return 0
    return int(eval_label)


def parse_med_q(cell: str) -> Tuple[float, float, float]:
    m = _MED_Q_RE.match((cell or "").strip())
    if not m:
        raise ValueError(f"Could not parse median/Q1/Q3 cell: {cell!r}")
    med, q1, q3 = m.groups()
    return float(med), float(q1), float(q3)


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    current_eval = ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_label = (row.get("eval downsample factor") or "").strip()
            if eval_label:
                current_eval = eval_label
            if not current_eval:
                continue
            row["eval downsample factor"] = current_eval
            rows.append(row)
    return rows


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    rows = load_rows(CSV_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        col = COLUMN_BY_METRIC[metric]
        series: Dict[str, List[Tuple[int, float, float, float]]] = {name: [] for name in SERIES_ORDER}

        for row in rows:
            train_key = (row.get("train dataset") or "").strip()
            series_name = TRAIN_MAP.get(train_key)
            if series_name is None:
                continue

            x = parse_factor(row["eval downsample factor"])
            med, q1, q3 = parse_med_q((row.get(col) or "").strip())
            series[series_name].append((x, med, q1, q3))

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for series_name in SERIES_ORDER:
            pts = sorted(series[series_name], key=lambda t: t[0])
            xs = [p[0] for p in pts]
            meds = [p[1] for p in pts]
            q1s = [p[2] for p in pts]
            q3s = [p[3] for p in pts]
            yerr_lower = [m - q1 for m, q1 in zip(meds, q1s)]
            yerr_upper = [q3 - m for m, q3 in zip(meds, q3s)]

            ax.errorbar(
                xs,
                meds,
                yerr=[yerr_lower, yerr_upper],
                marker=SERIES_MARKERS.get(series_name, "o"),
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                capsize=CAP_SIZE,
                alpha=LINE_ALPHA,
                color=SERIES_COLORS.get(series_name),
                zorder=SERIES_ZORDER.get(series_name, 2),
                label=series_name,
            )

        ax.set_title(TITLE_BY_METRIC.get(metric, f"{metric} vs downsampling factor"), pad=10)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL_BY_METRIC.get(metric, f"{metric} median"))
        if X_TICKS is not None:
            ax.set_xticks(X_TICKS)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=GRID_ALPHA)
        ax.legend(loc=LEGEND_LOC, frameon=False)
        fig.tight_layout()

        out_path = OUT_DIR / f"{metric}_downsample_plot.png"
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
