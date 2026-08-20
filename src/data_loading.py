"""
Load and validate the raw water-quality and climate input
files described in configs/config.yaml. Writes lightly-cleaned copies
(parsed dates, sorted, numeric-coerced) to data/processed/ so downstream
steps (STI, SAPEI, SCDHI, event classification, merging, ...) all read the
same validated data instead of re-parsing the raw CSVs themselves.

Run:
    python -m src.data_loading --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

WQ_REQUIRED_COLUMNS = ["date", "tsm", "chl_a", "cdom"]
CLIMATE_REQUIRED_COLUMNS = [
    "date", "precipitation", "tmean", "tmax", "tmin"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _validate_columns(df: pd.DataFrame, required: list, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required column(s) {missing}. Found: {list(df.columns)}"
        )


def _validate_dates(df: pd.DataFrame, name: str) -> None:
    n_bad = df["date"].isna().sum()
    if n_bad:
        raise ValueError(f"{name}: {n_bad} row(s) have an unparseable Date.")
    dup = df["date"].duplicated()
    if dup.any():
        dup_dates = df.loc[dup, "date"].dt.strftime("%Y-%m-%d").tolist()
        preview = dup_dates[:10]
        suffix = " ..." if len(dup_dates) > 10 else ""
        raise ValueError(f"{name}: duplicate date value(s): {preview}{suffix}")


def load_wq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, WQ_REQUIRED_COLUMNS, "wq.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    _validate_dates(df, "wq.csv")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["tsm", "chl_a", "cdom"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_climate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, CLIMATE_REQUIRED_COLUMNS, "cl.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    _validate_dates(df, "cl.csv")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["precipitation", "tmean", "tmax", "tmin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_daily_continuity(df: pd.DataFrame) -> pd.DatetimeIndex:
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return full_range.difference(df["date"])


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print(f"  rows: {len(df)}")
    print(
        f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts):
        print(f"  missing values:\n{na_counts.to_string()}")
    else:
        print("  missing values: none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate wq.csv and cl.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wq_path = cfg["paths"]["raw_wq"]
    cl_path = cfg["paths"]["raw_climate"]
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not Path(wq_path).exists():
        sys.exit(f"ERROR: {wq_path} not found. See data/raw/README.md.")
    if not Path(cl_path).exists():
        sys.exit(
            f"ERROR: {cl_path} not found — this file is user-supplied and "
            f"gitignored. See data/raw/README.md."
        )

    wq = load_wq(wq_path)
    climate = load_climate(cl_path)

    summarize(wq, "wq.csv")
    summarize(climate, "cl.csv")

    missing_climate_dates = check_daily_continuity(climate)
    print(
        f"\ncl.csv: {len(missing_climate_dates)} missing calendar day(s) in "
        f"{climate['date'].min().date()} -> {climate['date'].max().date()} "
        f"(expected daily coverage)."
    )
    if len(missing_climate_dates):
        preview = [d.strftime("%Y-%m-%d") for d in missing_climate_dates[:10]]
        suffix = " ..." if len(missing_climate_dates) > 10 else ""
        print(f"  first missing dates: {preview}{suffix}")

    wq_out = processed_dir / "wq_clean.csv"
    cl_out = processed_dir / "climate_clean.csv"
    wq.to_csv(wq_out, index=False)
    climate.to_csv(cl_out, index=False)
    print(f"\nWrote {wq_out}")
    print(f"Wrote {cl_out}")


if __name__ == "__main__":
    main()
