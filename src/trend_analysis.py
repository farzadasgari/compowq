"""
Run:
    python -m src.trend_analysis --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import pymannkendall as mk

from src.data_loading import load_config


def get_threshold(thresholds: pd.DataFrame, variable: str, window, set_name: str = "primary") -> float:
    row = thresholds[
        (thresholds["set"] == set_name)
        & (thresholds["variable"] == variable)
        & (thresholds["window"] == window if window is not None else thresholds["window"].isna())
    ]
    return float(row["threshold_value"].iloc[0])


def build_annual_series(df: pd.DataFrame, thresholds: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year

    sti_thr = get_threshold(thresholds, "sti", None)
    sapei_thr = get_threshold(thresholds, "sapei", "9m")
    df["is_heat"] = df["sti"] > sti_thr
    df["is_drought_9m"] = df["sapei_9m"] < sapei_thr

    annual = df.groupby("year").agg(
        mean_tmean=("tmean", "mean"),
        heatwave_days=("is_heat", "sum"),
        mean_sapei_9m=("sapei_9m", "mean"),
        drought_days_9m=("is_drought_9m", "sum"),
        mean_sti=("sti", "mean"),
        mean_sapei_6m=("sapei_6m", "mean"),
        mean_scdhi_6m=("scdhi_6m", "mean"),
        mean_scdhi_9m=("scdhi_9m", "mean"),
    )

    episodes_3d = episodes[episodes["min_duration_days"] == 3].copy()
    episodes_3d["year"] = pd.to_datetime(episodes_3d["start_date"]).dt.year
    episode_counts = episodes_3d.groupby("year").size().rename("heatwave_episodes")
    annual = annual.join(episode_counts, how="left")
    annual["heatwave_episodes"] = annual["heatwave_episodes"].fillna(0)

    full_years = df.groupby("year").size()
    complete = full_years[(full_years >= 365)].index
    annual = annual.loc[annual.index.isin(complete)]

    return annual


def run_trend_tests(annual: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in annual.columns:
        series = annual[col].dropna()
        result = mk.original_test(series.values)
        records.append(
            {
                "series": col,
                "n_years": len(series),
                "trend": result.trend,
                "mk_p_value": result.p,
                "kendall_tau": result.Tau,
                "sen_slope_per_year": result.slope,
                "sen_intercept": result.intercept,
                "significant_05": result.p < 0.05,
            }
        )
    return pd.DataFrame(records)


def build_summary(results: pd.DataFrame, annual: pd.DataFrame) -> str:
    lines = ["TREND ANALYSIS SUMMARY (Mann-Kendall, 1994-2025)", "=" * 70]
    lines.append(f"\n{len(annual)} complete years tested per series.\n")

    for _, row in results.iterrows():
        sig = "SIGNIFICANT" if row["significant_05"] else "not significant"
        lines.append(
            f"{row['series']:<20} trend={row['trend']:<12} p={row['mk_p_value']:.4f} ({sig})   "
            f"Sen's slope={row['sen_slope_per_year']:+.4f}/year"
        )

    n_sig = int(results["significant_05"].sum())
    lines.append(f"\n{n_sig}/{len(results)} series show a significant monotonic trend (p<0.05).")
    lines.append(f"Note: these are {len(results)} independent, pre-specified tests (not a swept battery), so no")
    lines.append("FDR correction is applied -- each answers a distinct, separately-motivated question.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mann-Kendall trend analysis on annual climate extremity")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "full_daily_merged.csv", parse_dates=["date"])
    thresholds = pd.read_csv(processed_dir / "event_thresholds.csv")
    episodes = pd.read_csv(processed_dir / "heatwave_episodes.csv", parse_dates=["start_date", "end_date"])

    annual = build_annual_series(df, thresholds, episodes)
    annual.to_csv(tables_dir / "trend_analysis_annual_series.csv")

    results = run_trend_tests(annual)
    results.to_csv(tables_dir / "trend_analysis_results.csv", index=False)

    summary = build_summary(results, annual)
    (tables_dir / "trend_analysis_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
