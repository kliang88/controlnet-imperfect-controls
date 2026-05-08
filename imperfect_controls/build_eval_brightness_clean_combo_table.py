#!/usr/bin/env python3
"""Build a compact brightness evaluation CSV (clean vs combo models).

Output layout:
- Rows: one per train model (clean, combo)
- Columns: one per metric
- Cell format: "median (q1, q3)"
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SECTIONS = frozenset({"mask", "shape", "color"})

# (train_dataset_label, path to summary.csv)
SUMMARIES = [
    ("clean", SCRIPT_DIR / "eval_results/clean/brighten/summary.csv"),
    ("combo", SCRIPT_DIR / "eval_results/combo/brighten/summary.csv"),
]

# Leave empty tuple () to include all metrics found.
METRICS = (
    "mask_iou",
    "roundness_pred",
    "radius_error",
    "center_error",
    "circle_color_error",
    "background_color_error",
    "total_color_error",
)

CSV_OUT = SCRIPT_DIR / "eval_tables/eval_on_brightness_clean_vs_combo.csv"


def load_metric_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = (row.get("section") or "").strip()
            metric = (row.get("metric") or "").strip()
            if section in EVAL_SECTIONS and metric:
                rows.append(row)
    return rows


def fmt_median_iqr(row: Dict[str, str]) -> str:
    median = (row.get("median") or "").strip()
    q1 = (row.get("q1") or "").strip()
    q3 = (row.get("q3") or "").strip()
    if median == "" and q1 == "" and q3 == "":
        return ""
    return f"{median} ({q1}, {q3})"


def main() -> None:
    per_train: Dict[str, Dict[str, Dict[str, str]]] = {}
    metric_order: List[str] = []
    seen_metrics = set()

    for train_label, summary_path in SUMMARIES:
        metric_rows = load_metric_rows(summary_path.expanduser())
        by_metric = {r["metric"].strip(): r for r in metric_rows}
        per_train[train_label] = by_metric
        for row in metric_rows:
            metric = row["metric"].strip()
            if metric not in seen_metrics:
                seen_metrics.add(metric)
                metric_order.append(metric)

    if METRICS:
        metric_order = list(METRICS)

    headers = ["train"] + metric_order
    out_rows: List[List[str]] = []
    train_labels = [label for label, _ in SUMMARIES]
    for train_label in train_labels:
        row_out = [train_label]
        for metric in metric_order:
            metric_row = per_train.get(train_label, {}).get(metric)
            row_out.append("" if metric_row is None else fmt_median_iqr(metric_row))
        out_rows.append(row_out)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(out_rows)

    print(f"Wrote CSV to: {CSV_OUT}")


if __name__ == "__main__":
    main()
