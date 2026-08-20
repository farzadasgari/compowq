"""
Compute the daily-scale Standardized Temperature Index (STI).

Run:
    python -m src.compute_sti --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import load_config

TEMP_COL = "tmean"


def compute_sti(df: pd.DataFrame, temp_col: str = TEMP_COL) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby([df["date"].dt.month, df["date"].dt.day])[temp_col]
    mu = grp.transform("mean")
    sigma = grp.transform("std")
    n = grp.transform("count")

    df["sti"] = (df[temp_col] - mu) / sigma
    df["sti_calendar_day_mean"] = mu
    df["sti_calendar_day_std"] = sigma
    df["sti_calendar_day_n"] = n
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute daily STI from the filled climate record")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    filled_path = processed_dir / "climate_filled.csv"
    if not filled_path.exists():
        raise SystemExit(
            f"ERROR: {filled_path} not found. Run `python -m src.gap_filling "
            f"--config {args.config}` first (Step 2b)."
        )

    climate = pd.read_csv(filled_path, parse_dates=["date"])
    sti = compute_sti(climate, TEMP_COL)

    n_valid = sti["sti"].notna().sum()
    n_total = len(sti)
    print(
        f"STI computed for {n_valid}/{n_total} days "
        f"({climate['date'].min().date()} -> {climate['date'].max().date()})."
    )
    print(
        f"  NaN days (from the {TEMP_COL} gap that could not be filled): {n_total - n_valid}")
    print(f"  STI range: {sti['sti'].min():.3f} to {sti['sti'].max():.3f}")
    print(
        f"  smallest calendar-day sample size (likely Feb 29): {sti['sti_calendar_day_n'].min()} years")

    out_path = processed_dir / "sti_daily.csv"
    out_cols = ["date", TEMP_COL, "sti", "sti_calendar_day_mean",
                "sti_calendar_day_std", "sti_calendar_day_n"]
    sti[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    wq_path = processed_dir / "wq_clean.csv"
    if wq_path.exists():
        wq = pd.read_csv(wq_path, parse_dates=["date"])
        matched = wq.merge(sti[["date", "sti"]], on="date", how="left")
        n_matched = matched["sti"].notna().sum()
        print(
            f"\nSanity check -- exact-date match against {len(wq)} Sentinel-2 acquisitions:")
        print(f"  {n_matched}/{len(wq)} acquisition dates have a valid STI value.")
        if n_matched < len(wq):
            missing = matched.loc[matched["sti"].isna(
            ), "date"].dt.strftime("%Y-%m-%d").tolist()
            print(f"  missing: {missing}")


if __name__ == "__main__":
    main()
