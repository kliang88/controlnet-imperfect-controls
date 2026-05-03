#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def moving_average(values, window: int):
    if window <= 1 or len(values) < window:
        return values

    smoothed = []
    running_sum = 0.0

    for i, v in enumerate(values):
        running_sum += v
        if i >= window:
            running_sum -= values[i - window]

        denom = min(i + 1, window)
        smoothed.append(running_sum / denom)

    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TensorBoard scalar metrics without launching TensorBoard."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="lightning_logs/version_13082144",
        help="TensorBoard log directory, for example lightning_logs/version_13082144",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="log_outputs",
        help="Output directory for plots and CSV files",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Moving average smoothing window. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional list of scalar tags to plot. If omitted, plot all scalar tags.",
    )

    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not logdir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {logdir}")

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])

    if not scalar_tags:
        print(f"No scalar metrics found in {logdir}")
        return

    if args.tags:
        selected_tags = [tag for tag in args.tags if tag in scalar_tags]
        missing_tags = [tag for tag in args.tags if tag not in scalar_tags]

        if missing_tags:
            print("Warning: these tags were not found:")
            for tag in missing_tags:
                print(f"  {tag}")
    else:
        selected_tags = scalar_tags

    print("Found scalar tags:")
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            print(f"  {tag}: last_step={last.step}, last_value={last.value}")

    print()
    print(f"Plotting {len(selected_tags)} tags to: {outdir}")

    summary_rows = []

    for tag in selected_tags:
        events = ea.Scalars(tag)

        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        smooth_values = moving_average(values, args.smooth)

        if not steps:
            continue

        safe_tag = safe_filename(tag)

        csv_path = outdir / f"{safe_tag}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "value", "smoothed_value", "wall_time"])
            for step, value, smooth_value, wall_time in zip(
                steps, values, smooth_values, wall_times
            ):
                writer.writerow([step, value, smooth_value, wall_time])

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label="raw", alpha=0.4)

        if args.smooth > 1:
            plt.plot(steps, smooth_values, label=f"moving avg window={args.smooth}")

        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.legend()
        plt.tight_layout()

        png_path = outdir / f"{safe_tag}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        summary_rows.append(
            {
                "tag": tag,
                "num_points": len(values),
                "first_step": steps[0],
                "last_step": steps[-1],
                "first_value": values[0],
                "last_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "csv": str(csv_path),
                "plot": str(png_path),
            }
        )

    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "tag",
            "num_points",
            "first_step",
            "last_step",
            "first_value",
            "last_value",
            "min_value",
            "max_value",
            "csv",
            "plot",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Done. Summary written to: {summary_csv}")
    print()
    print("Generated plots:")
    for row in summary_rows:
        print(f"  {row['tag']} -> {row['plot']}")


if __name__ == "__main__":
    main()