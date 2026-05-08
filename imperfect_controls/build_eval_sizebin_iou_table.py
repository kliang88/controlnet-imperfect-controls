#!/usr/bin/env python3
"""Build a size-binned mask IoU table from per-sample evaluation CSVs.

Input layout:
  eval_results/<train_dataset>/<eval_dataset>/per_sample_metrics.csv

Output layout (CSV):
- Rows grouped by eval dataset, with train dataset as subrows
- Columns are circle-size bins:
    - small:  lower 30% of radius_true
    - medium: next 30% of radius_true
    - large:  upper 30% of radius_true
- Cell value format: "median (Q1, Q3)" of mask_iou
"""

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR / "eval_results"
OUT_CSV = SCRIPT_DIR / "eval_tables" / "eval_sizebin_mask_iou_summary.csv"

SKIP_DATASETS = {"downsample_8", "downsample_24"}


def percentile(sorted_values: Sequence[float], p: float) -> Optional[float]:
    """Return percentile p in [0, 1] using linear interpolation."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    w = idx - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def median_q1_q3(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    s = sorted(values)
    return percentile(s, 0.5), percentile(s, 0.25), percentile(s, 0.75)


def fmt(median: Optional[float], q1: Optional[float], q3: Optional[float], decimals: int = 4) -> str:
    if median is None:
        return ""
    if q1 is None or q3 is None:
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} ({q1:.{decimals}f}, {q3:.{decimals}f})"


def parse_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_pairs(path: Path) -> List[Tuple[float, float]]:
    """Load (radius_true, mask_iou) from a per-sample metrics CSV."""
    pairs: List[Tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            radius_true = parse_float(row.get("radius_true", ""))
            mask_iou = parse_float(row.get("mask_iou", ""))
            if radius_true is None or mask_iou is None:
                continue
            pairs.append((radius_true, mask_iou))
    return pairs


def split_bins(pairs: Iterable[Tuple[float, float]]) -> Dict[str, List[float]]:
    """Split mask_iou values by size bins based on radius_true thresholds.

    Bins use quantile cutoffs from radius_true within each file:
      small  : radius_true <= p30
      medium : p30 < radius_true <= p60
      large  : radius_true >= p70
    """
    pairs_list = list(pairs)
    if not pairs_list:
        return {"small": [], "medium": [], "large": []}

    radii_sorted = sorted(r for r, _ in pairs_list)
    p30 = percentile(radii_sorted, 0.30)
    p60 = percentile(radii_sorted, 0.60)
    p70 = percentile(radii_sorted, 0.70)
    if p30 is None or p60 is None or p70 is None:
        return {"small": [], "medium": [], "large": []}

    out = {"small": [], "medium": [], "large": []}  # type: Dict[str, List[float]]
    for radius_true, mask_iou in pairs_list:
        if radius_true <= p30:
            out["small"].append(mask_iou)
        if p30 < radius_true <= p60:
            out["medium"].append(mask_iou)
        if radius_true >= p70:
            out["large"].append(mask_iou)
    return out


def normalize_dataset_name(name: str) -> Optional[str]:
    """Normalize dataset naming for reporting."""
    if name in SKIP_DATASETS:
        return None
    if name == "downsample_16":
        return "downsample"
    return name


def main() -> None:
    per_sample_files = sorted(EVAL_RESULTS_DIR.glob("*/*/per_sample_metrics.csv"))
    if not per_sample_files:
        raise FileNotFoundError(f"No per_sample_metrics.csv files found in {EVAL_RESULTS_DIR}")

    eval_order: List[str] = []
    train_order: List[str] = []
    seen_evals = set()
    seen_trains = set()

    # Store formatted values per (eval, train).
    table: Dict[Tuple[str, str], Dict[str, str]] = {}

    for path in per_sample_files:
        train = normalize_dataset_name(path.parts[-3])
        eval_dataset = normalize_dataset_name(path.parts[-2])
        if train is None or eval_dataset is None:
            continue

        if eval_dataset not in seen_evals:
            seen_evals.add(eval_dataset)
            eval_order.append(eval_dataset)
        if train not in seen_trains:
            seen_trains.add(train)
            train_order.append(train)

        bins = split_bins(load_pairs(path))
        table[(eval_dataset, train)] = {
            name: fmt(*median_q1_q3(values)) for name, values in bins.items()
        }

    headers = ["eval_dataset", "train_dataset", "small", "medium", "large"]
    out_rows: List[List[str]] = []

    for eval_dataset in eval_order:
        first = True
        for train in train_order:
            values = table.get((eval_dataset, train))
            if values is None:
                continue
            out_rows.append(
                [
                    eval_dataset if first else "",
                    train,
                    values.get("small", ""),
                    values.get("medium", ""),
                    values.get("large", ""),
                ]
            )
            first = False

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(out_rows)

    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
