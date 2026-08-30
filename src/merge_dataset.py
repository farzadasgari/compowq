"""
Build the single, canonical analysis-ready table.

Run:
    python -m src.merge_dataset --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import load_config

REQUIRED_FILES = {
    "wq_clean.csv": "python -m src.data_loading",
    "climate_with_pet.csv": "python -m src.compute_pet",
    "sti_daily.csv": "python -m src.compute_sti",
    "sapei_daily.csv": "python -m src.compute_sapei",
    "scdhi_daily.csv": "python -m src.compute_scdhi",
    "events_daily.csv": "python -m src.classify_events",
    "heatwave_episodes.csv": "python -m src.detect_heatwaves",
}


def _require_files(processed_dir: Path, config_path: str) -> None:
    missing = [f for f in REQUIRED_FILES if not (processed_dir / f).exists()]
    if missing:
        lines = [f"  {f}  ->  run `{REQUIRED_FILES[f]} --config {config_path}`" for f in missing]
        raise SystemExit("ERROR: missing required input file(s):\n" + "\n".join(lines))


def _assert_full_match(merged: pd.DataFrame, cols: list, n_expected: int, step_name: str) -> None:
    n_missing = merged[cols].isna().any(axis=1).sum()
    if n_missing:
        bad_dates = merged.loc[merged[cols].isna().any(axis=1), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"{step_name}: {n_missing}/{n_expected} acquisition dates failed to match "
            f"(expected 300/300, as confirmed in every earlier step). "
            f"Unmatched dates: {bad_dates[:10]}{' ...' if len(bad_dates) > 10 else ''}"
        )


def build_full_daily_merged(climate: pd.DataFrame, sti: pd.DataFrame, sapei: pd.DataFrame, scdhi: pd.DataFrame) -> pd.DataFrame:
    daily = climate.copy()
    daily = daily.merge(sti[["date", "sti"]], on="date", how="left")
    sapei_cols = [c for c in sapei.columns if c.startswith("sapei_")]
    daily = daily.merge(sapei[["date"] + sapei_cols], on="date", how="left")
    scdhi_cols = [c for c in scdhi.columns if c.startswith("scdhi_")]
    daily = daily.merge(scdhi[["date"] + scdhi_cols], on="date", how="left")
    return daily


def add_heatwave_membership(df: pd.DataFrame, episodes: pd.DataFrame, durations: list) -> pd.DataFrame:
    df = df.copy()
    for d in durations:
        col = f"is_heatwave_{d}d"
        df[col] = False
        eps_d = episodes[episodes["min_duration_days"] == d]
        for _, ep in eps_d.iterrows():
            in_episode = (df["date"] >= ep["start_date"]) & (df["date"] <= ep["end_date"])
            df.loc[in_episode, col] = True
    return df


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "date": ("Sentinel-2 acquisition date", "YYYY-MM-DD"),
        "tsm": ("Total suspended matter (C2RCC retrieval)", "units TBC before submission"),
        "chl_a": ("Chlorophyll-a (C2RCC retrieval)", "units TBC before submission"),
        "cdom": ("CDOM absorption at 443 nm (C2RCC iop_agelb)", "units TBC before submission"),
        "precipitation": ("Daily precipitation, Zahak station", "mm"),
        "tmean": ("Daily mean temperature, gap-filled", "deg C"),
        "tmax": ("Daily max temperature, gap-filled", "deg C"),
        "tmin": ("Daily min temperature, gap-filled", "deg C"),
        "pet": ("Potential evapotranspiration, computed from tmean via Eqs. 2-6 (not the raw cl.csv column)", "mm/day"),
        "sti": ("Standardized Temperature Index (daily, calendar-day z-score)", "dimensionless (~N(0,1))"),
        "is_heat_primary": ("STI above the 90th-percentile (primary) threshold", "boolean"),
        "is_heat_secondary": ("STI above the 85th-percentile (secondary) threshold", "boolean"),
    }
    for w in ["3m", "6m", "9m", "12m"]:
        descriptions[f"sapei_{w}"] = (f"SAPEI at the {w} antecedent window", "dimensionless (~N(0,1))")
        descriptions[f"scdhi_{w}"] = (f"SCDHI (Frank copula, primary family) at the {w} SAPEI window", "dimensionless")
        descriptions[f"is_drought_{w}_primary"] = (f"SAPEI_{w} below the 10th-percentile (primary) threshold", "boolean")
        descriptions[f"is_drought_{w}_secondary"] = (f"SAPEI_{w} below the 15th-percentile (secondary) threshold", "boolean")
        descriptions[f"is_compound_{w}_primary"] = (f"SCDHI_{w} below the 10th-percentile (primary) threshold", "boolean")
        descriptions[f"is_compound_{w}_secondary"] = (f"SCDHI_{w} below the 12th-percentile (secondary) threshold", "boolean")
    for d in [3, 5, 7]:
        descriptions[f"is_heatwave_{d}d"] = (f"Acquisition falls within a formal >={d}-consecutive-day heatwave episode", "boolean")

    rows = []
    for col in df.columns:
        desc, unit = descriptions.get(col, ("", ""))
        rows.append({"column": col, "description": desc, "unit": unit})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge WQ, climate, indices, and event flags into one analysis table")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    _require_files(processed_dir, args.config)

    wq = pd.read_csv(processed_dir / "wq_clean.csv", parse_dates=["date"])
    n_expected = len(wq)

    climate = pd.read_csv(processed_dir / "climate_with_pet.csv", parse_dates=["date"])
    climate_cols = ["date", "precipitation", "tmean", "tmax", "tmin", "pet"]
    merged = wq.merge(climate[climate_cols], on="date", how="left")
    _assert_full_match(merged, ["precipitation", "tmean", "tmax", "tmin", "pet"], n_expected, "climate merge")

    sti = pd.read_csv(processed_dir / "sti_daily.csv", parse_dates=["date"])[["date", "sti"]]
    merged = merged.merge(sti, on="date", how="left")
    _assert_full_match(merged, ["sti"], n_expected, "STI merge")

    sapei = pd.read_csv(processed_dir / "sapei_daily.csv", parse_dates=["date"])
    sapei_cols = [c for c in sapei.columns if c.startswith("sapei_")]
    merged = merged.merge(sapei, on="date", how="left")
    _assert_full_match(merged, sapei_cols, n_expected, "SAPEI merge")

    scdhi = pd.read_csv(processed_dir / "scdhi_daily.csv", parse_dates=["date"])
    scdhi_cols = [c for c in scdhi.columns if c.startswith("scdhi_")]
    merged = merged.merge(scdhi, on="date", how="left")
    _assert_full_match(merged, scdhi_cols, n_expected, "SCDHI merge")

    events = pd.read_csv(processed_dir / "events_daily.csv", parse_dates=["date"])
    flag_cols = [c for c in events.columns if c.startswith("is_")]
    merged = merged.merge(events[["date"] + flag_cols], on="date", how="left")
    _assert_full_match(merged, flag_cols, n_expected, "event classification merge")

    episodes = pd.read_csv(processed_dir / "heatwave_episodes.csv", parse_dates=["start_date", "end_date"])
    durations = cfg.get("events", {}).get("heatwave_durations_days", [3, 5, 7])
    merged = add_heatwave_membership(merged, episodes, durations)

    print(f"Merged analysis dataset: {len(merged)} rows (expected {n_expected}), {len(merged.columns)} columns")
    print(f"Date range: {merged['date'].min().date()} -> {merged['date'].max().date()}")
    print(f"Missing values anywhere in the final table: {int(merged.isna().sum().sum())}")
    for d in durations:
        col = f"is_heatwave_{d}d"
        print(f"Acquisitions within a >= {d}-day heatwave episode: {int(merged[col].sum())}/{n_expected}")

    out_path = processed_dir / "analysis_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    dictionary = build_data_dictionary(merged)
    dict_path = processed_dir / "data_dictionary.csv"
    dictionary.to_csv(dict_path, index=False)
    print(f"Wrote {dict_path}")

    full_daily = build_full_daily_merged(climate, sti, sapei, scdhi)
    full_daily_path = processed_dir / "full_daily_merged.csv"
    full_daily.to_csv(full_daily_path, index=False)
    print(f"Wrote {full_daily_path} ({len(full_daily)} days)")


if __name__ == "__main__":
    main()
