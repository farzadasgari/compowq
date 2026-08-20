"""
Classify each day (and separately, each Sentinel-2 acquisition date) as
heat / drought / compound-dry-hot using percentile thresholds on the full
daily STI, SAPEI, and SCDHI series.

Run:
    python -m src.classify_events --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import load_config


def compute_threshold(series: pd.Series, pct: float) -> float:
    return float(series.dropna().quantile(pct / 100))


def classify(df: pd.DataFrame, thresholds_cfg: dict, sapei_windows: list, scdhi_windows: list) -> tuple:
    threshold_records = []

    for set_name, t in thresholds_cfg.items():
        heat_thresh = compute_threshold(df["sti"], t["heat_pct"])
        df[f"is_heat_{set_name}"] = df["sti"] > heat_thresh
        threshold_records.append(
            {"set": set_name, "variable": "sti", "window": None,
                "percentile": t["heat_pct"], "threshold_value": heat_thresh}
        )

        for w in sapei_windows:
            col = f"sapei_{w}"
            thresh = compute_threshold(df[col], t["drought_pct"])
            df[f"is_drought_{w}_{set_name}"] = df[col] < thresh
            threshold_records.append(
                {"set": set_name, "variable": "sapei", "window": w,
                    "percentile": t["drought_pct"], "threshold_value": thresh}
            )

        for w in scdhi_windows:
            col = f"scdhi_{w}"
            thresh = compute_threshold(df[col], t["compound_pct"])
            df[f"is_compound_{w}_{set_name}"] = df[col] < thresh
            threshold_records.append(
                {"set": set_name, "variable": "scdhi", "window": w,
                    "percentile": t["compound_pct"], "threshold_value": thresh}
            )

    return df, threshold_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify heat/drought/compound events per day and per acquisition date")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])

    sti_path = processed_dir / "sti_daily.csv"
    sapei_path = processed_dir / "sapei_daily.csv"
    scdhi_path = processed_dir / "scdhi_daily.csv"
    for p, step in [(sti_path, "compute_sti"), (sapei_path, "compute_sapei"), (scdhi_path, "compute_scdhi")]:
        if not p.exists():
            raise SystemExit(
                f"ERROR: {p} not found. Run `python -m src.{step} --config {args.config}` first.")

    sti = pd.read_csv(sti_path, parse_dates=["date"])[["date", "sti"]]
    sapei = pd.read_csv(sapei_path, parse_dates=["date"])
    scdhi = pd.read_csv(scdhi_path, parse_dates=["date"])

    df = sti.merge(sapei, on="date", how="inner").merge(
        scdhi, on="date", how="inner")

    sapei_windows = sorted(c.replace("sapei_", "")
                           for c in sapei.columns if c.startswith("sapei_"))
    scdhi_windows = sorted(c.replace("scdhi_", "")
                           for c in scdhi.columns if c.startswith("scdhi_"))

    df, threshold_records = classify(
        df, cfg["thresholds"], sapei_windows, scdhi_windows)

    print(
        f"Event classification computed over {len(df)} days ({df['date'].min().date()} -> {df['date'].max().date()})\n")
    thresholds_df = pd.DataFrame(threshold_records)
    print(thresholds_df.to_string(index=False))

    flag_cols = [c for c in df.columns if c.startswith("is_")]
    print("\nDay counts per flag (full daily record):")
    for col in flag_cols:
        print(f"  {col:<28} {int(df[col].sum())} days")

    out_path = processed_dir / "events_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    thresholds_path = processed_dir / "event_thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"Wrote {thresholds_path}")

    wq_path = processed_dir / "wq_clean.csv"
    if wq_path.exists():
        wq = pd.read_csv(wq_path, parse_dates=["date"])
        matched = wq.merge(df, on="date", how="left")
        matched_out = processed_dir / "events_matched.csv"
        matched.to_csv(matched_out, index=False)
        print(f"Wrote {matched_out}")

        print(
            f"\nAcquisition-date counts per flag (n={len(wq)} Sentinel-2 dates):")
        for col in flag_cols:
            print(f"  {col:<28} {int(matched[col].sum())} acquisitions")


if __name__ == "__main__":
    main()
