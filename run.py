#!/usr/bin/env python3
"""
Runs the entire pipeline in order, from raw data through final analysis
tables and figures, with one command:

    python run.py --config configs/config.yaml

Equivalent to running each of the following manually, in order:
    python -m src.data_loading
    python -m src.gap_filling
    python -m src.compute_pet
    python -m src.compute_sti
    python -m src.compute_sapei
    python -m src.compute_scdhi
    python -m src.classify_events
    python -m src.detect_heatwaves
    python -m src.merge_dataset
    python -m src.correlation_analysis
    python -m src.group_comparisons
    python -m src.pca_permanova
    python -m src.seasonal_analysis
    python -m src.viz.figure_timeseries
    python -m src.viz.figure_calendar_heatmap
    python -m src.viz.figure_event_comparison
    python -m src.viz.figure_pca_suite

Run with --skip-figures to run the analysis pipeline only.
"""

import argparse
import subprocess
import sys
import time

PIPELINE_STEPS = [
    "src.data_loading",
    "src.gap_filling",
    "src.compute_pet",
    "src.compute_sti",
    "src.compute_sapei",
    "src.compute_scdhi",
    "src.classify_events",
    "src.detect_heatwaves",
    "src.merge_dataset",
    "src.correlation_analysis",
    "src.group_comparisons",
    "src.pca_permanova",
    "src.seasonal_analysis",
    "src.trend_analysis",
    "src.seasonal_regression",
    "src.drought_episode_duration",
]

FIGURE_STEPS = [
    "src.viz.figure_timeseries",
    "src.viz.figure_calendar_heatmap",
    "src.viz.figure_event_counts",
    "src.viz.figure_event_comparison",
    "src.viz.figure_correlation_heatmap",
    "src.viz.figure_seasonal_correlation",
    "src.viz.figure_pca_2d",
    "src.viz.figure_pca_3d",
    "src.viz.figure_permanova_permdisp",
]


def run_step(module: str, config_path: str) -> float:
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", module, "--config", config_path])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nFAILED: {module} (exit code {result.returncode})")
        print("Stopping here -- downstream steps depend on this step's output.")
        sys.exit(result.returncode)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline, in order, with one command")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Run the analysis pipeline only, skip figure generation")
    args = parser.parse_args()

    steps = PIPELINE_STEPS + ([] if args.skip_figures else FIGURE_STEPS)
    overall_start = time.time()

    for i, module in enumerate(steps, start=1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(steps)}] {module}")
        print(f"{'='*70}")
        elapsed = run_step(module, args.config)
        print(f"-- done in {elapsed:.1f}s")

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"Pipeline complete: {len(steps)} steps, {total:.1f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
