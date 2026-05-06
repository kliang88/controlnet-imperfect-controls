#!/usr/bin/env python3
"""Build a comparison table from multiple evaluation summary.csv files.

Each summary follows imperfect_controls/evaluate.py format: metric rows use
section in {mask, shape, color} with mean, median, std, q1, q3; run rows are ignored.

Edit TITLE, SUMMARIES, and METRICS below, then run:

  python imperfect_controls/summary_metrics_table.py
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent

# --- edit below ---
# Printed above the text table; leave "" for no title (CSV output unchanged).
TITLE = "Evaluation on combination of corruption types"

# (train_dataset_label, path to summary.csv)
SUMMARIES = [
    ("clean", _SCRIPT_DIR / "eval_results/clean/combo/summary.csv"),
    ("blur", _SCRIPT_DIR / "eval_results/blur/combo/summary.csv"),
    ("noise", _SCRIPT_DIR / "eval_results/noise/combo/summary.csv"),
    ("downsample", _SCRIPT_DIR / "eval_results/downsample/combo/summary.csv"),
    ("combination", _SCRIPT_DIR / "eval_results/combo/combo/summary.csv"),
]

# Rows appear in this order; drop or reorder entries as needed.
# Use () to auto-include every metric found (first-seen order across SUMMARIES).
METRICS = (
    "mask_iou",
    # "roundness_abs_delta",
    "roundness_pred",
    "radius_error",
    "center_error",
    "circle_color_error",
    "background_color_error",
    "total_color_error",
)

DECIMALS = 4
# Set to a Path to write CSV; leave None to skip.
CSV_OUT = _SCRIPT_DIR / "eval_tables/eval_on_combo.csv"  # type: Optional[Path]
# --- edit above ---


EVAL_SECTIONS = frozenset({"mask", "shape", "color"})


def load_metric_rows(path: Path) -> List[Dict[str, str]]:
    rows = []  # type: List[Dict[str, str]]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = (row.get("section") or "").strip()
            if section not in EVAL_SECTIONS:
                continue
            metric = (row.get("metric") or "").strip()
            if not metric:
                continue
            rows.append(row)
    return rows


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def format_table(
    records: List[
        Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]
    ],
    decimals: int,
) -> str:
    """records: (metric, train_label, mean, sd, median, q1, q3)."""
    headers = ("metric", "train", "mean", "sd", "median", "Q1", "Q3")
    cells = []  # type: List[List[str]]
    for m, t, mean, sd, med, q1, q3 in records:

        def fmt(x: Optional[float]) -> str:
            if x is None:
                return ""
            return f"{x:.{decimals}f}"

        cells.append([m, t, fmt(mean), fmt(sd), fmt(med), fmt(q1), fmt(q3)])

    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad_row(vals: List[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    lines = [pad_row(list(headers)), pad_row(["-" * w for w in widths])]
    prev_metric = None
    for row in cells:
        metric_cell = row[0] if row[0] != prev_metric else ""
        prev_metric = row[0]
        lines.append(pad_row([metric_cell] + row[1:]))
    return "\n".join(lines)


def main():
    labeled_paths = SUMMARIES

    per_train = {}  # type: Dict[str, Dict[str, Dict[str, str]]]
    metric_order = []  # type: List[str]
    seen_metrics = set()  # type: set

    for train_label, path in labeled_paths:
        path = path.expanduser()
        metric_rows = load_metric_rows(path)
        by_metric = {r["metric"].strip(): r for r in metric_rows}
        per_train[train_label] = by_metric
        for r in metric_rows:
            m = r["metric"].strip()
            if m not in seen_metrics:
                seen_metrics.add(m)
                metric_order.append(m)

    if METRICS:
        metric_order = list(METRICS)

    combined = []  # type: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]]
    train_labels = [lbl for lbl, _ in labeled_paths]
    for m in metric_order:
        for train_label in train_labels:
            row = per_train[train_label].get(m)
            if row is None:
                combined.append((m, train_label, None, None, None, None, None))
                continue
            combined.append(
                (
                    m,
                    train_label,
                    parse_float(row, "mean"),
                    parse_float(row, "std"),
                    parse_float(row, "median"),
                    parse_float(row, "q1"),
                    parse_float(row, "q3"),
                )
            )

    if CSV_OUT is not None:
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "train", "mean", "sd", "median", "Q1", "Q3"])
            for row in combined:
                w.writerow(row)

    if TITLE.strip():
        print(TITLE.strip())
        print()
    print(format_table(combined, decimals=DECIMALS))


if __name__ == "__main__":
    main()
