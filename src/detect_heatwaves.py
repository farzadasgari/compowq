"""
Identify formal heatwave episodes: independent runs of >= min_duration_days
consecutive calendar days.

Run:
    python -m src.detect_heatwaves --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import load_config

DEFAULT_DURATIONS_DAYS = [3, 5, 7]


def find_episodes(df: pd.DataFrame, flag_col: str, min_duration_days: int) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    flag = df[flag_col].fillna(False).to_numpy()
    dates = df["date"].to_numpy()

    day_diff = np.empty(len(df), dtype=float)
    day_diff[0] = 1
    day_diff[1:] = (dates[1:] - dates[:-1]) / np.timedelta64(1, "D")

    episodes = []
    start_idx = None
    for i in range(len(df)):
        continuous = day_diff[i] == 1
        if flag[i]:
            if start_idx is None:
                start_idx = i
            elif not continuous:
                episodes.append((start_idx, i - 1))
                start_idx = i
        else:
            if start_idx is not None:
                episodes.append((start_idx, i - 1))
                start_idx = None
    if start_idx is not None:
        episodes.append((start_idx, len(df) - 1))

    records = []
    for start_idx, end_idx in episodes:
        duration = end_idx - start_idx + 1
        if duration >= min_duration_days:
            records.append(
                {
                    "start_date": df["date"].iloc[start_idx],
                    "end_date": df["date"].iloc[end_idx],
                    "duration_days": duration,
                }
            )
    return pd.DataFrame(records, columns=["start_date", "end_date", "duration_days"])


def sweep_durations(sti: pd.DataFrame, flag_col: str, durations: list) -> pd.DataFrame:
    frames = []
    for d in durations:
        eps = find_episodes(sti, flag_col, d)
        eps.insert(0, "min_duration_days", d)
        frames.append(eps)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["min_duration_days", "start_date",
                 "end_date", "duration_days"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep formal (sustained) heatwave episode detection across durations")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])

    sti_path = processed_dir / "sti_daily.csv"
    thresholds_path = processed_dir / "event_thresholds.csv"
    if not sti_path.exists():
        raise SystemExit(
            f"ERROR: {sti_path} not found. Run `python -m src.compute_sti --config {args.config}` first.")
    if not thresholds_path.exists():
        raise SystemExit(
            f"ERROR: {thresholds_path} not found. Run `python -m src.classify_events --config {args.config}` first.")

    sti = pd.read_csv(sti_path, parse_dates=["date"])
    thresholds = pd.read_csv(thresholds_path)
    row = thresholds[(thresholds["set"] == "primary") &
                     (thresholds["variable"] == "sti")]
    if row.empty:
        raise SystemExit(
            "ERROR: primary STI threshold not found in event_thresholds.csv")
    heat_thresh = float(row["threshold_value"].iloc[0])

    sti["is_heat_primary"] = sti["sti"] > heat_thresh

    durations = cfg.get("events", {}).get(
        "heatwave_durations_days", DEFAULT_DURATIONS_DAYS)
    print(
        f"Formal heatwave episodes (STI > {heat_thresh:.4f}, primary 90th percentile), swept across durations {durations}:\n")

    all_episodes = sweep_durations(sti, "is_heat_primary", durations)

    for d in durations:
        eps = all_episodes[all_episodes["min_duration_days"] == d]
        print(f"  >= {d} consecutive days: {len(eps)} episodes", end="")
        if len(eps):
            print(f", {int(eps['duration_days'].sum())} total heatwave-days, "
                  f"median {eps['duration_days'].median():.1f}d, max {int(eps['duration_days'].max())}d")
        else:
            print()

    out_path = processed_dir / "heatwave_episodes.csv"
    all_episodes.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
