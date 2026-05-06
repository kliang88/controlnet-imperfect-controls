#!/usr/bin/env python3
"""Pretty-print a grid of summary.csv stats for downsample eval sweeps.

Reads ``eval_results/<train>/<eval>/summary.csv`` (same layout as evaluate.py).
Rows are eval conditions (clean vs downsample factors); sub-rows are train runs.
Each metric column shows: median (Q1, Q3) — Q3 is used where quartiles exist (no Q2 in summaries).

Optional ``CSV_OUT`` writes the same columns and formatted cells as the printed table.

Edit constants below, then:

  python imperfect_controls/downsample_eval_table.py
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent

# --- edit below ---
TITLE = "Evaluation on downsampled data"

# (label shown in table, eval subdirectory under eval_results/<train>/…)
EVAL_ROWS = (
    ("clean", "clean"),
    ("8", "downsample_8"),
    ("16", "downsample_16"),
    ("24", "downsample_24"),
)

# (label shown in table, train subdirectory: eval_results/<this>/<eval>/summary.csv)
TRAIN_SUBROWS = (
    ("clean", "clean"),
    ("downsample x16", "downsample"),
)

DECIMALS = 4
# Set to a Path to save CSV matching the printed table; leave None to skip.
CSV_OUT = _SCRIPT_DIR / "eval_tables/downsample_levels.csv"  # type: Optional[Path]
# --- edit above ---

EVAL_SECTIONS = frozenset({"mask", "shape", "color"})
METRICS = ("mask_iou", "center_error", "radius_error")


def load_metric_map(path: Path):
    # type: (Path) -> Dict[str, Dict[str, str]]
    out = {}  # type: Dict[str, Dict[str, str]]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = (row.get("section") or "").strip()
            if section not in EVAL_SECTIONS:
                continue
            metric = (row.get("metric") or "").strip()
            if metric:
                out[metric] = row
    return out


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def metric_median_q1_q3(
    mmap: Dict[str, Dict[str, str]], metric: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    row = mmap.get(metric)
    if row is None:
        return None, None, None
    return (
        parse_float(row, "median"),
        parse_float(row, "q1"),
        parse_float(row, "q3"),
    )


def format_median_q(med: Optional[float], q1: Optional[float], q3: Optional[float], decimals: int) -> str:
    if med is None:
        return ""
    d = decimals
    if q1 is None or q3 is None:
        return f"{med:.{d}f}"
    return f"{med:.{d}f} ({q1:.{d}f}, {q3:.{d}f})"


def format_grid(
    rows: List[Tuple[str, str, str, str, str]],
    headers: Tuple[str, ...],
) -> str:
    cells = [list(r) for r in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad_row(vals: List[str]) -> str:
        return "  ".join(vals[i].ljust(widths[i]) for i in range(len(vals)))

    lines = [pad_row(list(headers)), pad_row(["-" * w for w in widths])]
    prev_eval = None
    for row in cells:
        eval_cell = row[0] if row[0] != prev_eval else ""
        prev_eval = row[0]
        lines.append(pad_row([eval_cell] + row[1:]))
    return "\n".join(lines)


def main():
    root = _SCRIPT_DIR / "eval_results"
    headers = (
        "eval downsample factor",
        "train dataset",
        "mask_iou med (Q1,Q3)",
        "center_error med (Q1,Q3)",
        "radius_error med (Q1,Q3)",
    )

    built = []  # type: List[Tuple[str, str, str, str, str]]

    for eval_label, eval_dir in EVAL_ROWS:
        for train_label, train_dir in TRAIN_SUBROWS:
            summary = root / train_dir / eval_dir / "summary.csv"
            mmap = load_metric_map(summary) if summary.is_file() else {}

            triples = [metric_median_q1_q3(mmap, m) for m in METRICS]
            cells = [format_median_q(med, q1, q3, DECIMALS) for med, q1, q3 in triples]
            built.append((eval_label, train_label, cells[0], cells[1], cells[2]))

    if CSV_OUT is not None:
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(headers))
            prev_eval = None  # type: Optional[str]
            for eval_label, train_label, c0, c1, c2 in built:
                eval_cell = eval_label if eval_label != prev_eval else ""
                prev_eval = eval_label
                writer.writerow([eval_cell, train_label, c0, c1, c2])

    if TITLE.strip():
        print(TITLE.strip())
        print()

    print(format_grid(built, headers))


if __name__ == "__main__":
    main()
