"""
Stratifies the same group-comparison battery from src.group_comparisons by
season (Northern Hemisphere: Winter=Dec-Jan-Feb, Spring=Mar-Apr-May,
Summer=Jun-Jul-Aug, Autumn=Sep-Oct-Nov).

Run:
    python -m src.seasonal_analysis --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.correlation_analysis import benjamini_hochberg, effect_size_label
from src.data_loading import load_config
from src.group_comparisons import build_event_variants, mannwhitney_group_test

WQ_COLS = ["tsm", "chl_a", "cdom"]
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
MIN_N_EVENT_SEASONAL = 8


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["date"].dt.month.map(SEASON_MAP)
    return df


def run_seasonal_comparisons(df: pd.DataFrame, wq_cols: list) -> pd.DataFrame:
    variants = build_event_variants(df.columns)
    records = []

    for season in SEASON_ORDER:
        season_df = df[df["season"] == season]
        for v in variants:
            for y in wq_cols:
                result = mannwhitney_group_test(season_df, y, v["flag_col"])
                records.append({"season": season, "y": y, **v,
                               **result, "n_season_total": len(season_df)})

    result_df = pd.DataFrame(records)

    too_small = result_df["n_event"] < MIN_N_EVENT_SEASONAL
    result_df.loc[too_small, ["p_raw", "effect_size_r"]] = pd.NA

    result_df["p_fdr"] = pd.NA
    for season in SEASON_ORDER:
        mask = (result_df["season"] == season) & result_df["p_raw"].notna()
        if mask.sum() == 0:
            continue
        result_df.loc[mask, "p_fdr"] = benjamini_hochberg(
            result_df.loc[mask, "p_raw"].values)

    result_df["significant_fdr_05"] = result_df["p_fdr"] < 0.05
    result_df["effect_size_label"] = result_df["effect_size_r"].apply(
        lambda r: effect_size_label(r) if pd.notna(r) else ""
    )
    result_df["too_small_to_test"] = too_small
    return result_df


def build_summary(long_df: pd.DataFrame) -> str:
    lines = [
        "SEASONAL GROUP COMPARISON SUMMARY (exploratory -- see module docstring)", "=" * 70]

    for season in SEASON_ORDER:
        season_df = long_df[long_df["season"] == season]
        n_total = season_df["n_season_total"].iloc[0]
        n_tested = (~season_df["too_small_to_test"]).sum()
        n_skipped = season_df["too_small_to_test"].sum()
        n_sig = int(season_df["significant_fdr_05"].sum())

        lines.append(f"\n{season} (n={n_total} acquisitions):")
        lines.append(
            f"  {n_tested} tests run, {n_skipped} skipped (n_event < {MIN_N_EVENT_SEASONAL})")
        lines.append(
            f"  {n_sig} significant after within-season FDR correction (q < 0.05)")

        if n_sig:
            sig = season_df[season_df["significant_fdr_05"]
                            ].sort_values("p_fdr")
            for _, row in sig.iterrows():
                direction = "higher" if row["effect_size_r"] > 0 else "lower"
                lines.append(
                    f"    {row['y']} {direction} during {row['event_type']} ({row['variant']}): "
                    f"r={row['effect_size_r']:.3f} ({row['effect_size_label']}), q={row['p_fdr']:.4f}, "
                    f"n_event={row['n_event']}, n_non_event={row['n_non_event']}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seasonal stratification of the group-comparison battery")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "analysis_dataset.csv"
    if not dataset_path.exists():
        raise SystemExit(
            f"ERROR: {dataset_path} not found. Run `python -m src.merge_dataset --config {args.config}` first.")

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = add_season(df)

    long_df = run_seasonal_comparisons(df, WQ_COLS)
    long_path = tables_dir / "seasonal_group_comparison_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    summary = build_summary(long_df)
    summary_path = tables_dir / "seasonal_group_comparison_summary.txt"
    summary_path.write_text(summary)
    print(f"Wrote {summary_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
