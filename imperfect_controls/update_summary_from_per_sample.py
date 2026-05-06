#!/usr/bin/env python3
"""Update summary.csv robust stats from per_sample_metrics.csv.

Adds/updates:
- q1
- q3
- iqr
- trimmed_mean_5pct
- trimmed_std_5pct
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    pos = (n - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def robust_stats(
    values: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None, None, None, None, None

    sorted_vals = sorted(finite)
    q25 = percentile(sorted_vals, 25.0)
    q75 = percentile(sorted_vals, 75.0)
    if q25 is None or q75 is None:
        return None, None, None, None, None

    trim_n = int(math.floor(0.05 * len(sorted_vals)))
    if trim_n > 0:
        trimmed = sorted_vals[trim_n:-trim_n]
    else:
        trimmed = sorted_vals
    trimmed_mean = (sum(trimmed) / float(len(trimmed))) if trimmed else None
    if trimmed:
        mu = trimmed_mean
        assert mu is not None
        trimmed_var = sum((x - mu) ** 2 for x in trimmed) / float(len(trimmed))
        trimmed_std = math.sqrt(trimmed_var)
    else:
        trimmed_std = None
    iqr = float(q75 - q25)
    return float(q25), float(q75), iqr, trimmed_mean, trimmed_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample", required=True, type=Path, help="Path to per_sample_metrics.csv")
    parser.add_argument("--summary", required=True, type=Path, help="Path to summary.csv to update in-place")
    args = parser.parse_args()

    with args.per_sample.open("r", newline="", encoding="utf-8") as f:
        per_rows = list(csv.DictReader(f))

    if not per_rows:
        raise RuntimeError(f"No rows found in {args.per_sample}")

    with args.summary.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if "iqr" not in fieldnames:
        std_idx = fieldnames.index("std") if "std" in fieldnames else len(fieldnames)
        fieldnames.insert(std_idx + 1, "iqr")
    if "q1" not in fieldnames:
        std_idx = fieldnames.index("std") if "std" in fieldnames else len(fieldnames)
        fieldnames.insert(std_idx + 1, "q1")
    if "q3" not in fieldnames:
        q1_idx = fieldnames.index("q1") if "q1" in fieldnames else len(fieldnames)
        fieldnames.insert(q1_idx + 1, "q3")
    if "iqr" in fieldnames and "q3" in fieldnames and fieldnames.index("iqr") < fieldnames.index("q3"):
        fieldnames.remove("iqr")
        fieldnames.insert(fieldnames.index("q3") + 1, "iqr")
    if "trimmed_mean_5pct" not in fieldnames:
        insert_idx = fieldnames.index("iqr") + 1 if "iqr" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_idx, "trimmed_mean_5pct")
    if "trimmed_std_5pct" not in fieldnames:
        insert_idx = (
            fieldnames.index("trimmed_mean_5pct") + 1
            if "trimmed_mean_5pct" in fieldnames
            else len(fieldnames)
        )
        fieldnames.insert(insert_idx, "trimmed_std_5pct")

    metric_names = {r.get("metric", "") for r in rows if r.get("section") != "run"}
    metric_to_stats: Dict[
        str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]
    ] = {}
    for metric in metric_names:
        if metric not in per_rows[0]:
            continue
        vals: List[float] = []
        for r in per_rows:
            v = r.get(metric, "")
            if v in ("", None):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        metric_to_stats[metric] = robust_stats(vals)

    for row in rows:
        metric = row.get("metric", "")
        if row.get("section") == "run" or metric not in metric_to_stats:
            row.setdefault("q1", "")
            row.setdefault("q3", "")
            row.setdefault("iqr", "")
            row.setdefault("trimmed_mean_5pct", "")
            row.setdefault("trimmed_std_5pct", "")
            continue
        q1, q3, iqr, tmean, tstd = metric_to_stats[metric]
        row["q1"] = "" if q1 is None else str(q1)
        row["q3"] = "" if q3 is None else str(q3)
        row["iqr"] = "" if iqr is None else str(iqr)
        row["trimmed_mean_5pct"] = "" if tmean is None else str(tmean)
        row["trimmed_std_5pct"] = "" if tstd is None else str(tstd)

    with args.summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {args.summary}")


if __name__ == "__main__":
    main()
