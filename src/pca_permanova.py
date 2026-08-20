"""
Two PCAs, and the actual group-difference test the paper is built around.

Run:
    python -m src.pca_permanova --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.distance import DistanceMatrix, permanova, permdisp

from src.correlation_analysis import benjamini_hochberg
from src.data_loading import load_config
from src.group_comparisons import build_event_variants

OPTICAL_COLS = ["tsm", "chl_a", "cdom"]
INTEGRATED_COLS = OPTICAL_COLS + [
    "sti", "sapei_3m", "sapei_6m", "sapei_9m", "sapei_12m",
    "scdhi_3m", "scdhi_6m", "scdhi_9m", "scdhi_12m",
]


def run_pca(df: pd.DataFrame, cols: list, id_col: str = "date") -> tuple:
    X = StandardScaler().fit_transform(df[cols].values)
    n_components = min(len(cols), 5)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    score_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    scores_df.insert(0, id_col, df[id_col].values)

    loadings_df = pd.DataFrame(
        pca.components_.T, index=cols, columns=score_cols)

    variance_df = pd.DataFrame(
        {
            "PC": score_cols,
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance_explained": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return scores_df, loadings_df, variance_df


def build_optical_distance_matrix(df: pd.DataFrame, id_col: str = "date") -> DistanceMatrix:
    X = StandardScaler().fit_transform(df[OPTICAL_COLS].values)
    dist = squareform(pdist(X, metric="euclidean"))
    ids = df[id_col].astype(str).values
    return DistanceMatrix(dist, ids=ids)


def run_permanova_permdisp(distmat: DistanceMatrix, grouping: pd.Series, permutations: int = 999, seed: int = 0) -> dict:
    grouping = np.asarray(grouping)
    labels = np.where(grouping, "event", "non_event")

    n_event = int((labels == "event").sum())
    n_non_event = int((labels == "non_event").sum())
    if n_event < 3 or n_non_event < 3:
        return {
            "n_event": n_event, "n_non_event": n_non_event,
            "permanova_F": np.nan, "permanova_p": np.nan,
            "permdisp_F": np.nan, "permdisp_p": np.nan,
        }

    pn = permanova(distmat, labels, permutations=permutations, seed=seed)
    pd_res = permdisp(distmat, labels, permutations=permutations, seed=seed)

    return {
        "n_event": n_event, "n_non_event": n_non_event,
        "permanova_F": float(pn["test statistic"]), "permanova_p": float(pn["p-value"]),
        "permdisp_F": float(pd_res["test statistic"]), "permdisp_p": float(pd_res["p-value"]),
    }


def run_all_permanova(df: pd.DataFrame, distmat: DistanceMatrix, id_col: str = "date") -> pd.DataFrame:
    variants = build_event_variants(df.columns)

    records = []
    for v in variants:
        # array-like, same row order as distmat.ids
        grouping = df[v["flag_col"]].values
        result = run_permanova_permdisp(distmat, grouping)
        records.append({**v, **result})

    result_df = pd.DataFrame(records)

    for col_prefix in ["permanova", "permdisp"]:
        p_col, q_col = f"{col_prefix}_p", f"{col_prefix}_q"
        valid = result_df[p_col].notna()
        result_df[q_col] = np.nan
        result_df.loc[valid, q_col] = benjamini_hochberg(
            result_df.loc[valid, p_col].values)
        result_df[f"{col_prefix}_significant_fdr_05"] = result_df[q_col] < 0.05

    return result_df


def build_summary(long_df: pd.DataFrame, optical_variance: pd.DataFrame) -> str:
    lines = ["PCA + PERMANOVA/PERMDISP SUMMARY", "=" * 70]

    lines.append("\nOptical-only PCA (tsm, chl_a, cdom), variance explained:")
    for _, row in optical_variance.iterrows():
        lines.append(
            f"  {row['PC']}: {row['variance_explained']*100:.1f}% (cumulative {row['cumulative_variance_explained']*100:.1f}%)")

    n_permanova_sig = int(long_df["permanova_significant_fdr_05"].sum())
    n_permdisp_sig = int(long_df["permdisp_significant_fdr_05"].sum())
    lines.append(
        f"\n{len(long_df)} event-definition variants tested against the optical-only distance matrix.")
    lines.append(
        f"PERMANOVA (centroid difference): {n_permanova_sig} significant after FDR correction (q < 0.05).")
    lines.append(
        f"PERMDISP (dispersion difference): {n_permdisp_sig} significant after FDR correction (q < 0.05).")

    if n_permanova_sig:
        lines.append("\nPERMANOVA significant results (sorted by q):")
        sig = long_df[long_df["permanova_significant_fdr_05"]
                      ].sort_values("permanova_q")
        for _, row in sig.iterrows():
            lines.append(
                f"  {row['event_type']} ({row['variant']}): pseudo-F={row['permanova_F']:.3f}, "
                f"q={row['permanova_q']:.4f}, n_event={row['n_event']}, n_non_event={row['n_non_event']}"
            )
    else:
        lines.append(
            "\nNo PERMANOVA test survived FDR correction -- optical-space centroids do not")
        lines.append(
            "differ significantly between event and non-event groups, at any window/duration")
        lines.append(
            "tested, consistent with the univariate group comparisons.")

    if n_permdisp_sig:
        lines.append(
            "\nPERMDISP significant results (dispersion differs -- interpret any PERMANOVA")
        lines.append(
            "result for these variants cautiously, since PERMANOVA assumes equal dispersion):")
        sig = long_df[long_df["permdisp_significant_fdr_05"]
                      ].sort_values("permdisp_q")
        for _, row in sig.iterrows():
            lines.append(
                f"  {row['event_type']} ({row['variant']}): q={row['permdisp_q']:.4f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optical-only PCA + PERMANOVA/PERMDISP against event groups; integrated PCA (descriptive)")
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

    # Optical-only PCA
    opt_scores, opt_loadings, opt_variance = run_pca(df, OPTICAL_COLS)
    opt_scores.to_csv(tables_dir / "pca_optical_scores.csv", index=False)
    opt_loadings.to_csv(tables_dir / "pca_optical_loadings.csv")
    opt_variance.to_csv(tables_dir / "pca_optical_variance.csv", index=False)
    print("Optical-only PCA:")
    print(opt_variance.to_string(index=False))
    print()

    # Integrated PCA
    int_scores, int_loadings, int_variance = run_pca(df, INTEGRATED_COLS)
    int_scores.to_csv(tables_dir / "pca_integrated_scores.csv", index=False)
    int_loadings.to_csv(tables_dir / "pca_integrated_loadings.csv")
    int_variance.to_csv(
        tables_dir / "pca_integrated_variance.csv", index=False)
    print("Integrated PCA (optical + climate/index, descriptive only):")
    print(int_variance.head(3).to_string(index=False))
    print()

    # PERMANOVA / PERMDISP on the optical-only distance matrix
    distmat = build_optical_distance_matrix(df)
    long_df = run_all_permanova(df, distmat)
    long_path = tables_dir / "permanova_permdisp_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} variants x permanova+permdisp)")

    summary = build_summary(long_df, opt_variance)
    summary_path = tables_dir / "permanova_permdisp_summary.txt"
    summary_path.write_text(summary)
    print(f"Wrote {summary_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
