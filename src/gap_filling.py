"""
Fill short gaps in the daily climate record produced by src.data_loading,
using a per-variable, config-driven strategy. All four fill strategies are
implemented (linear interpolation, calendar-day climatology, forward-fill,
and "none" / flag-only) so any of them can be selected per variable via
configs/config.yaml — for the main analysis, or later for a sensitivity
re-run. Nothing here hardcodes a single "correct" strategy.

Run:
    python -m src.gap_filling --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_loading import load_config

FILL_COLUMNS = ["precipitation", "tmean", "tmax", "tmin"]
VALID_METHODS = {"linear", "climatology", "ffill", "none"}


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    return df.reset_index()


def _fillable_mask(is_na: pd.Series, max_gap_days: int) -> pd.Series:
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("size")
    return is_na & (run_len <= max_gap_days)


def _apply_method(series: pd.Series, dates: pd.Series, method: str) -> pd.Series:
    if method == "linear":
        return series.interpolate(method="linear", limit_direction=None)
    if method == "ffill":
        return series.ffill()
    if method == "climatology":
        clim_mean = series.groupby(
            [dates.dt.month, dates.dt.day]).transform("mean")
        return series.fillna(clim_mean)
    if method == "none":
        return series.copy()
    raise ValueError(
        f"Unknown gap-filling method '{method}'. Valid: {sorted(VALID_METHODS)}")


def fill_column(df: pd.DataFrame, col: str, method: str, max_gap_days: int) -> pd.DataFrame:
    series = df[col]
    is_na = series.isna()
    fillable = _fillable_mask(is_na, max_gap_days)

    candidate = _apply_method(series, df["date"], method)
    out = series.copy()
    out[fillable] = candidate[fillable]

    df[col] = out
    df[f"{col}_gap_filled"] = is_na & out.notna()
    df[f"{col}_unfillable"] = is_na & out.isna()
    return df


def apply_gap_filling(df: pd.DataFrame, methods: dict, max_gap_days: int) -> pd.DataFrame:
    df = reindex_daily(df)
    for col in FILL_COLUMNS:
        method = methods.get(col, "none")
        if method not in VALID_METHODS:
            raise ValueError(
                f"Config gap_filling.methods.{col} = '{method}' is not one of {sorted(VALID_METHODS)}")
        df = fill_column(df, col, method, max_gap_days)
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"\nRows after reindexing to a complete daily calendar: {len(df)}")
    for col in FILL_COLUMNS:
        n_filled = df[f"{col}_gap_filled"].sum()
        n_unfillable = df[f"{col}_unfillable"].sum()
        print(
            f"  {col:<14} filled: {n_filled:>4}   still missing (gap > max_gap_days): {n_unfillable:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill gaps in the daily climate record")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    climate_clean_path = processed_dir / "climate_clean.csv"
    if not climate_clean_path.exists():
        raise SystemExit(
            f"ERROR: {climate_clean_path} not found. Run `python -m src.data_loading "
            f"--config {args.config}` first (Step 2)."
        )

    gf_cfg = cfg["gap_filling"]
    df = pd.read_csv(climate_clean_path, parse_dates=["date"])

    filled = apply_gap_filling(df, gf_cfg["methods"], gf_cfg["max_gap_days"])
    summarize(filled)

    out_path = processed_dir / "climate_filled.csv"
    filled.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
