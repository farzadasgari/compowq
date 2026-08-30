"""
Compute the daily-scale Standardized Antecedent Precipitation Evapotranspiration
Index (SAPEI) at four antecedent timescales (3, 6, 9, 12 "months").

Run:
    python -m src.compute_sapei --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func
from scipy.stats import norm

from src.data_loading import load_config

DEFAULT_WINDOW_DAYS = {3: 90, 6: 180, 9: 270, 12: 360}
DEFAULT_CLIMATOLOGY_WINDOW_DAYS = 15


def fit_loglogistic_lmoments(x: np.ndarray) -> tuple:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 10:
        raise ValueError(f"Too few samples ({n}) to fit a log-logistic distribution.")
    i = np.arange(1, n + 1)

    w0 = np.mean(x)
    w1 = np.mean(((n - i) / (n - 1)) * x)
    w2 = np.mean(((n - i) * (n - i - 1)) / ((n - 1) * (n - 2)) * x)

    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g1 = gamma_func(1 + 1 / beta)
    g2 = gamma_func(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / (g1 * g2)
    gamma_param = w0 - alpha * g1 * g2

    return alpha, beta, gamma_param


def loglogistic_cdf(x: np.ndarray, alpha: float, beta: float, gamma_param: float) -> np.ndarray:
    return 1.0 / (1.0 + (alpha / (x - gamma_param)) ** beta)


def compute_water_balance(df: pd.DataFrame) -> pd.Series:
    return df["precipitation"] - df["pet"]


def pseudo_doy(dates: pd.Series) -> np.ndarray:
    month = dates.dt.month.values
    day = dates.dt.day.values
    day = np.where((month == 2) & (day == 29), 28, day)
    ref = pd.to_datetime({"year": 2001, "month": month, "day": day})
    return ref.dt.dayofyear.values


def compute_sapei_for_window(
    df: pd.DataFrame, water_balance: pd.Series, window_days: int, climatology_window_days: int = DEFAULT_CLIMATOLOGY_WINDOW_DAYS
) -> pd.Series:
    accum = water_balance.rolling(window=window_days, min_periods=window_days).sum()
    doy = pseudo_doy(df["date"])
    accum_vals = accum.values
    valid = ~np.isnan(accum_vals)

    sapei_vals = np.full(len(df), np.nan)

    for target_doy in range(1, 366):
        dist = np.abs(doy - target_doy)
        dist = np.minimum(dist, 365 - dist)
        window_mask = (dist <= climatology_window_days) & valid
        target_mask = (doy == target_doy) & valid

        if target_mask.sum() == 0 or window_mask.sum() < 10:
            continue

        alpha, beta, gamma_param = fit_loglogistic_lmoments(accum_vals[window_mask])
        f = loglogistic_cdf(accum_vals[target_mask], alpha, beta, gamma_param)
        f = np.clip(f, 1e-10, 1 - 1e-10)
        sapei_vals[target_mask] = norm.ppf(f)

    return pd.Series(sapei_vals, index=df.index)


def compute_sapei(df: pd.DataFrame, window_days: dict, climatology_window_days: int = DEFAULT_CLIMATOLOGY_WINDOW_DAYS) -> pd.DataFrame:
    wb = compute_water_balance(df)
    out = df[["date"]].copy()
    for months, days in window_days.items():
        out[f"sapei_{months}m"] = compute_sapei_for_window(df, wb, days, climatology_window_days)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute daily SAPEI at 3/6/9/12-month windows")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    pet_path = processed_dir / "climate_with_pet.csv"
    if not pet_path.exists():
        raise SystemExit(
            f"ERROR: {pet_path} not found. Run `python -m src.compute_pet "
            f"--config {args.config}` first (Step 3.5)."
        )

    sapei_cfg = cfg.get("indices", {}).get("sapei", {})
    window_days = sapei_cfg.get("window_days", DEFAULT_WINDOW_DAYS)
    window_days = {int(k): int(v) for k, v in window_days.items()}
    climatology_window_days = int(sapei_cfg.get("climatology_window_days", DEFAULT_CLIMATOLOGY_WINDOW_DAYS))

    climate = pd.read_csv(pet_path, parse_dates=["date"])
    sapei = compute_sapei(climate, window_days, climatology_window_days)

    print(f"SAPEI computed {climate['date'].min().date()} -> {climate['date'].max().date()}")
    print(f"Climatology fit window: +/-{climatology_window_days} days (moving, leap-year-safe)")
    for months in sorted(window_days):
        col = f"sapei_{months}m"
        n_valid = sapei[col].notna().sum()
        print(
            f"  {col:<10} valid: {n_valid:>6}/{len(sapei)}   "
            f"range: {sapei[col].min():.3f} to {sapei[col].max():.3f}"
        )

    out_path = processed_dir / "sapei_daily.csv"
    sapei.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    wq_path = processed_dir / "wq_clean.csv"
    if wq_path.exists():
        wq = pd.read_csv(wq_path, parse_dates=["date"])
        matched = wq.merge(sapei, on="date", how="left")
        print(f"\nSanity check -- exact-date match against {len(wq)} Sentinel-2 acquisitions:")
        for months in sorted(window_days):
            col = f"sapei_{months}m"
            n_matched = matched[col].notna().sum()
            print(f"  {col:<10} {n_matched}/{len(wq)} acquisition dates matched")


if __name__ == "__main__":
    main()
