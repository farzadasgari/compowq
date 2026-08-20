"""
Comprehensive pairwise correlation analysis between water quality parameters
(TSM, Chl-a, CDOM) and the full set of climate/index predictors.

Run:
    python -m src.correlation_analysis --config configs/config.yaml
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loading import load_config

WQ_COLS = ["tsm", "chl_a", "cdom"]
PREDICTOR_COLS = [
    "precipitation", "tmean", "tmax", "tmin", "pet",
    "sti", "sapei_3m", "sapei_6m", "sapei_9m", "sapei_12m",
    "scdhi_3m", "scdhi_6m", "scdhi_9m", "scdhi_12m",
]
INDEX_COLS = [
    "sti", "sapei_3m", "sapei_6m", "sapei_9m", "sapei_12m",
    "scdhi_3m", "scdhi_6m", "scdhi_9m", "scdhi_12m",
]
METHODS = {"pearson": stats.pearsonr,
           "spearman": stats.spearmanr, "kendall": stats.kendalltau}


def effect_size_label(r: float) -> str:
    if pd.isna(r):
        return ""
    a = abs(r)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "strong"
    return "very strong"


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def compute_pairwise(df: pd.DataFrame, x_cols: list, y_cols: list, methods: dict) -> pd.DataFrame:
    records = []
    for method_name, method_fn in methods.items():
        for y in y_cols:
            for x in x_cols:
                sub = df[[x, y]].dropna()
                if len(sub) < 3:
                    r, p = np.nan, np.nan
                else:
                    r, p = method_fn(sub[x], sub[y])
                records.append(
                    {"y": y, "x": x, "method": method_name, "r": r, "p_raw": p, "n": len(sub)})
    result = pd.DataFrame(records)

    result["p_fdr"] = np.nan
    for method_name in methods:
        mask = (result["method"] == method_name) & result["p_raw"].notna()
        result.loc[mask, "p_fdr"] = benjamini_hochberg(
            result.loc[mask, "p_raw"].values)

    result["significant_fdr_05"] = result["p_fdr"] < 0.05
    result["effect_size"] = result["r"].apply(effect_size_label)
    return result


def to_wide_matrix(long_df: pd.DataFrame, method: str, value_col: str, x_cols: list, y_cols: list) -> pd.DataFrame:
    sub = long_df[long_df["method"] == method]
    wide = sub.pivot(index="y", columns="x", values=value_col)
    return wide.reindex(index=y_cols, columns=x_cols)


def annotate_with_stars(r_matrix: pd.DataFrame, p_matrix: pd.DataFrame) -> pd.DataFrame:
    def fmt(r, p):
        if pd.isna(r):
            return ""
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"{r:.3f}{stars}"

    return pd.DataFrame(
        {col: [fmt(r_matrix.loc[idx, col], p_matrix.loc[idx, col])
               for idx in r_matrix.index] for col in r_matrix.columns},
        index=r_matrix.index,
    )


def compute_symmetric_matrix(df: pd.DataFrame, cols: list) -> tuple:
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))),
                         index=cols, columns=cols)
    for a, b in combinations(cols, 2):
        sub = df[[a, b]].dropna()
        r, p = stats.pearsonr(sub[a], sub[b])
        r_mat.loc[a, b] = r_mat.loc[b, a] = r
        p_mat.loc[a, b] = p_mat.loc[b, a] = p
    return r_mat, p_mat


def build_summary(long_df: pd.DataFrame, multicollinearity: pd.DataFrame) -> str:
    lines = ["CORRELATION ANALYSIS SUMMARY", "=" * 60]

    n_tests = len(long_df)
    n_sig = int(long_df["significant_fdr_05"].sum())
    lines.append(
        f"\n{n_tests} total tests run (3 WQ vars x {len(PREDICTOR_COLS)} predictors x {len(METHODS)} methods).")
    lines.append(
        f"{n_sig} significant after Benjamini-Hochberg FDR correction (q < 0.05).")

    if n_sig == 0:
        lines.append(
            "\nNo WQ-predictor pair reached significance after FDR correction, at any of the "
            "three correlation methods. This is consistent with the manuscript's existing "
            "statement that the correlation analysis was inconclusive."
        )
    else:
        lines.append("\nSignificant pairs (FDR q < 0.05):")
        sig = long_df[long_df["significant_fdr_05"]].sort_values("p_fdr")
        for _, row in sig.iterrows():
            lines.append(
                f"  {row['y']} ~ {row['x']} ({row['method']}): r={row['r']:.3f}, "
                f"q={row['p_fdr']:.4f}, n={row['n']}, {row['effect_size']}"
            )

    pearson_df = long_df[long_df["method"] == "pearson"].copy()
    pearson_df["abs_r"] = pearson_df["r"].abs()
    top = pearson_df.sort_values("abs_r", ascending=False).head(5)
    lines.append(
        "\nStrongest correlations regardless of significance (top 5 by |r|, Pearson):")
    for _, row in top.iterrows():
        lines.append(
            f"  {row['y']} ~ {row['x']}: r={row['r']:.3f} (p_raw={row['p_raw']:.4f}, {row['effect_size']})")

    lines.append(
        "\nPredictor multicollinearity (Pearson |r| among the 9 index variables):")
    upper = multicollinearity.where(
        np.triu(np.ones(multicollinearity.shape), k=1).astype(bool))
    strong_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and abs(val) >= 0.7:
                strong_pairs.append((idx, col, val))
    strong_pairs.sort(key=lambda t: -abs(t[2]))
    if strong_pairs:
        lines.append(
            f"  {len(strong_pairs)} pair(s) with |r| >= 0.7 -- meaningful redundancy among indices:")
        for a, b, v in strong_pairs[:15]:
            lines.append(f"    {a} ~ {b}: r={v:.3f}")
        lines.append(
            "  This redundancy is exactly what PCA is designed to handle.")
    else:
        lines.append("  No pair reached |r| >= 0.7.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive WQ-predictor correlation analysis")
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

    long_df = compute_pairwise(df, PREDICTOR_COLS, WQ_COLS, METHODS)
    long_path = tables_dir / "correlation_wq_vs_predictors_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} tests)")

    for method in METHODS:
        r_mat = to_wide_matrix(long_df, method, "r", PREDICTOR_COLS, WQ_COLS)
        p_mat = to_wide_matrix(long_df, method, "p_fdr",
                               PREDICTOR_COLS, WQ_COLS)
        annotated = annotate_with_stars(r_mat, p_mat)

        r_path = tables_dir / f"correlation_matrix_{method}_r.csv"
        p_path = tables_dir / f"correlation_matrix_{method}_p_fdr.csv"
        annotated_path = tables_dir / \
            f"correlation_matrix_{method}_annotated.csv"
        r_mat.to_csv(r_path)
        p_mat.to_csv(p_path)
        annotated.to_csv(annotated_path)
        print(f"Wrote {r_path}, {p_path}, {annotated_path}")

    multicollinearity_r, _ = compute_symmetric_matrix(df, INDEX_COLS)
    multi_path = tables_dir / "correlation_predictor_multicollinearity.csv"
    multicollinearity_r.to_csv(multi_path)
    print(f"Wrote {multi_path}")

    wq_r, _ = compute_symmetric_matrix(df, WQ_COLS)
    wq_path = tables_dir / "correlation_wq_intercorrelation.csv"
    wq_r.to_csv(wq_path)
    print(f"Wrote {wq_path}")

    summary = build_summary(long_df, multicollinearity_r)
    summary_path = tables_dir / "correlation_summary.txt"
    summary_path.write_text(summary)
    print(f"Wrote {summary_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
