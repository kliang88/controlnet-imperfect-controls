#!/usr/bin/env python3
"""Build a compact summary CSV from eval_tables/eval_on_*.csv files.

Output layout:
- Rows grouped by eval dataset, with train dataset as subrows
- Columns as "median (Q1, Q3)" for selected metrics
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_TABLES_DIR = SCRIPT_DIR / "eval_tables"
OUT_CSV = EVAL_TABLES_DIR / "eval_median_iqr_summary.csv"

TARGET_METRICS = ("mask_iou", "center_error", "radius_error")


def fmt_median_iqr(median: str, q1: str, q3: str) -> str:
    if median == "" and q1 == "" and q3 == "":
        return ""
    return f"{median} ({q1}, {q3})"


def eval_name_from_file(path: Path) -> str:
    # eval_on_blur.csv -> blur
    return path.stem.replace("eval_on_", "", 1)


def load_summary_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Return {(train, metric): row_dict} for quick lookup."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train = (row.get("train") or "").strip()
            metric = (row.get("metric") or "").strip()
            if train == "" or metric == "":
                continue
            out[(train, metric)] = row
    return out


def main() -> None:
    eval_files = sorted(EVAL_TABLES_DIR.glob("eval_on_*.csv"))
    if not eval_files:
        raise FileNotFoundError(f"No eval_on_*.csv files found in {EVAL_TABLES_DIR}")

    # Stable train order from first file appearance.
    train_order: List[str] = []
    seen_trains = set()
    for path in eval_files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train = (row.get("train") or "").strip()
                if train and train not in seen_trains:
                    seen_trains.add(train)
                    train_order.append(train)

    headers = ["eval_dataset", "train_dataset", "mask_iou", "center_error", "radius_error"]
    out_rows: List[List[str]] = []

    for path in eval_files:
        eval_name = eval_name_from_file(path)
        by_train_metric = load_summary_rows(path)
        for i, train in enumerate(train_order):
            row_out = [eval_name if i == 0 else "", train]
            for metric in TARGET_METRICS:
                src = by_train_metric.get((train, metric), {})
                value = fmt_median_iqr(
                    (src.get("median") or "").strip(),
                    (src.get("Q1") or "").strip(),
                    (src.get("Q3") or "").strip(),
                )
                row_out.append(value)
            out_rows.append(row_out)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(out_rows)

    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
