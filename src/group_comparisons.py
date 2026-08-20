"""
Run:
    python -m src.group_comparisons --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.correlation_analysis import benjamini_hochberg, effect_size_label
from src.data_loading import load_config

WQ_COLS = ["tsm", "chl_a", "cdom"]


def build_event_variants(columns: list) -> list:
    variants = []
    for col in columns:
        if col.startswith("is_heatwave_"):
            variants.append({"event_type": "heatwave", "flag_col": col,
                            "variant": col.replace("is_heatwave_", "")})
        elif col.startswith("is_heat_"):
            variants.append({"event_type": "heat", "flag_col": col,
                            "variant": col.replace("is_heat_", "")})
        elif col.startswith("is_drought_"):
            variants.append({"event_type": "drought", "flag_col": col,
                            "variant": col.replace("is_drought_", "")})
        elif col.startswith("is_compound_"):
            variants.append({"event_type": "compound", "flag_col": col,
                            "variant": col.replace("is_compound_", "")})
    return variants


def mannwhitney_group_test(df: pd.DataFrame, value_col: str, flag_col: str) -> dict:
    event = df.loc[df[flag_col] == True, value_col].dropna()  # noqa: E712
    non_event = df.loc[df[flag_col] == False, value_col].dropna()  # noqa: E712
    n1, n0 = len(event), len(non_event)

    if n1 < 3 or n0 < 3:
        return {
            "n_event": n1, "n_non_event": n0,
            "median_event": np.nan, "median_non_event": np.nan,
            "U": np.nan, "p_raw": np.nan, "effect_size_r": np.nan,
        }

    U, p = stats.mannwhitneyu(event, non_event, alternative="two-sided")
    r = (2 * U) / (n1 * n0) - 1
    return {
        "n_event": n1, "n_non_event": n0,
        "median_event": float(event.median()), "median_non_event": float(non_event.median()),
        "U": float(U), "p_raw": float(p), "effect_size_r": float(r),
    }


def run_all_comparisons(df: pd.DataFrame, wq_cols: list) -> pd.DataFrame:
    variants = build_event_variants(df.columns)
    records = []
    for v in variants:
        for y in wq_cols:
            result = mannwhitney_group_test(df, y, v["flag_col"])
            records.append({"y": y, **v, **result})

    result_df = pd.DataFrame(records)
    valid = result_df["p_raw"].notna()
    result_df["p_fdr"] = np.nan
    result_df.loc[valid, "p_fdr"] = benjamini_hochberg(
        result_df.loc[valid, "p_raw"].values)
    result_df["significant_fdr_05"] = result_df["p_fdr"] < 0.05
    result_df["effect_size_label"] = result_df["effect_size_r"].apply(
        effect_size_label)
    return result_df


def robustness_table(long_df: pd.DataFrame, event_type: str, wq_cols: list) -> pd.DataFrame:
    sub = long_df[long_df["event_type"] == event_type]

    def fmt(row):
        if pd.isna(row["effect_size_r"]):
            return "n/a"
        star = "*" if row["significant_fdr_05"] else ""
        return f"{row['effect_size_r']:.3f}{star}"

    sub = sub.copy()
    sub["cell"] = sub.apply(fmt, axis=1)
    wide = sub.pivot(index="y", columns="variant", values="cell")
    return wide.reindex(index=wq_cols)


def build_summary(long_df: pd.DataFrame) -> str:
    lines = [
        "GROUP COMPARISON SUMMARY (event-day vs non-event-day, Mann-Whitney U)", "=" * 70]

    n_tests = len(long_df)
    n_variants = long_df[["event_type", "variant"]].drop_duplicates().shape[0]
    n_sig = int(long_df["significant_fdr_05"].sum())
    lines.append(
        f"\n{n_tests} total tests ({n_variants} event-definition variants x {len(WQ_COLS)} WQ variables).")
    lines.append(
        f"{n_sig} significant after Benjamini-Hochberg FDR correction (q < 0.05).")

    if n_sig == 0:
        lines.append(
            "\nNo event-definition variant showed a significant group difference for any WQ")
        lines.append(
            "variable, at any window/duration/threshold-set tested. This would mean the")
        lines.append(
            "group-level question also comes up empty, not just the correlation approach.")
    else:
        lines.append("\nSignificant results (FDR q < 0.05), sorted by q:")
        sig = long_df[long_df["significant_fdr_05"]].sort_values("p_fdr")
        for _, row in sig.iterrows():
            direction = "higher" if row["effect_size_r"] > 0 else "lower"
            lines.append(
                f"  {row['y']} is {direction} during {row['event_type']} ({row['variant']}): "
                f"median_event={row['median_event']:.3f} vs median_non_event={row['median_non_event']:.3f}, "
                f"r={row['effect_size_r']:.3f} ({row['effect_size_label']}), q={row['p_fdr']:.4f}, "
                f"n_event={row['n_event']}, n_non_event={row['n_non_event']}"
            )

        lines.append(
            "\nRobustness check -- for each significant (y, event_type) pair, how many of the")
        lines.append(
            "variants tested for that event_type were ALSO significant (same y):")
        for (y, event_type), _ in sig.groupby(["y", "event_type"]):
            family = long_df[(long_df["y"] == y) & (
                long_df["event_type"] == event_type)]
            n_variant_total = len(family)
            n_variant_sig = int(family["significant_fdr_05"].sum())
            same_direction = family.loc[family["significant_fdr_05"], "effect_size_r"].apply(
                np.sign)
            consistent = same_direction.nunique() <= 1 if len(same_direction) else True
            lines.append(
                f"  {y} x {event_type}: {n_variant_sig}/{n_variant_total} variants significant, "
                f"direction {'consistent' if consistent else 'INCONSISTENT -- check sign flips'}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group comparisons: event-day vs non-event-day WQ differences")
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
    long_df = run_all_comparisons(df, WQ_COLS)

    long_path = tables_dir / "group_comparison_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} tests)")

    for event_type in ["heat", "heatwave", "drought", "compound"]:
        wide = robustness_table(long_df, event_type, WQ_COLS)
        out_path = tables_dir / f"group_comparison_{event_type}.csv"
        wide.to_csv(out_path)
        print(f"Wrote {out_path}")

    summary = build_summary(long_df)
    summary_path = tables_dir / "group_comparison_summary.txt"
    summary_path.write_text(summary)
    print(f"Wrote {summary_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
