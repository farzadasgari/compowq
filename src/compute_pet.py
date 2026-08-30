import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import load_config

GSC = 0.0820  # MJ m^-2 min^-1, FAO-56 solar constant


def compute_ra(day_of_year: np.ndarray, latitude_rad: float) -> np.ndarray:
    j = day_of_year.astype(float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(latitude_rad) * np.tan(delta), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(latitude_rad) * np.sin(delta)
        + np.cos(latitude_rad) * np.cos(delta) * np.sin(ws)
    )
    return ra


def compute_pet(tmean: pd.Series, dates: pd.Series, latitude_deg: float) -> pd.Series:
    latitude_rad = np.deg2rad(latitude_deg)
    day_of_year = dates.dt.dayofyear.values
    ra = compute_ra(day_of_year, latitude_rad)
    pet = 0.408 * ra * (tmean.values + 5) / 100
    pet = np.maximum(pet, 0)
    return pd.Series(pet, index=tmean.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PET from tmean, date, and station latitude")
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

    station_cfg = cfg.get("station", {})
    if "latitude_deg" not in station_cfg:
        raise SystemExit("ERROR: configs/config.yaml is missing station.latitude_deg.")
    latitude_deg = float(station_cfg["latitude_deg"])

    df = pd.read_csv(filled_path, parse_dates=["date"])
    if "pet" in df.columns:
        df = df.rename(columns={"pet": "pet_raw_prior"})
    df["pet"] = compute_pet(df["tmean"], df["date"], latitude_deg)

    n_valid = df["pet"].notna().sum()
    print(f"PET computed for {n_valid}/{len(df)} days (station latitude {latitude_deg} deg N).")
    print(f"  PET range: {df['pet'].min():.3f} to {df['pet'].max():.3f}")
    if "pet_raw_prior" in df.columns:
        both = df[["pet", "pet_raw_prior"]].dropna()
        if len(both):
            corr = both["pet"].corr(both["pet_raw_prior"])
            print(f"  correlation with the prior (untrusted) pet column: {corr:.3f} (n={len(both)})")

    out_path = processed_dir / "climate_with_pet.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
