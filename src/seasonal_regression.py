"""
Formalizes the seasonal pattern found by src.seasonal_analysis as ONE
pre-specified regression model per WQ variable, rather than a stratified
battery of subgroup tests:

Run:
    python -m src.seasonal_regression --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from src.data_loading import load_config
from src.seasonal_analysis import SEASON_MAP

WQ_COLS = ["tsm", "chl_a", "cdom"]
PREDICTOR = "sapei_9m"


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["date"].dt.month.map(SEASON_MAP)
    df["season"] = pd.Categorical(df["season"], categories=["Summer", "Winter", "Spring", "Autumn"])
    return df


def fit_interaction_model(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    formula = f"{y_col} ~ {PREDICTOR} * C(season)"
    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    rows = []
    for term in model.params.index:
        rows.append(
            {
                "y": y_col,
                "term": term,
                "coef": model.params[term],
                "std_err": model.bse[term],
                "p_value": model.pvalues[term],
                "significant_05": model.pvalues[term] < 0.05,
            }
        )
    rows.append({"y": y_col, "term": "R-squared", "coef": model.rsquared, "std_err": None, "p_value": None, "significant_05": None})
    rows.append({"y": y_col, "term": "n_obs", "coef": int(model.nobs), "std_err": None, "p_value": None, "significant_05": None})
    return pd.DataFrame(rows)


def build_summary(all_results: pd.DataFrame) -> str:
    lines = ["SEASONAL INTERACTION REGRESSION: y ~ sapei_9m * C(season)", "=" * 70]
    lines.append("Reference season: Summer (showed no effect in exploratory analysis)")
    lines.append("Robust (HC3) standard errors throughout.\n")

    for y in WQ_COLS:
        sub = all_results[all_results["y"] == y]
        lines.append(f"\n{y.upper()}")
        lines.append("-" * 40)
        for _, row in sub.iterrows():
            if row["term"] in ("R-squared", "n_obs"):
                lines.append(f"  {row['term']}: {row['coef']:.4f}" if row["term"] == "R-squared" else f"  {row['term']}: {int(row['coef'])}")
                continue
            sig = " *" if row["significant_05"] else ""
            lines.append(f"  {row['term']:<35} coef={row['coef']:+.4f}  se={row['std_err']:.4f}  p={row['p_value']:.4f}{sig}")

        interaction_rows = sub[sub["term"].str.contains(":", na=False)]
        n_sig_interaction = int(interaction_rows["significant_05"].sum())
        lines.append(f"  -> {n_sig_interaction}/{len(interaction_rows)} sapei_9m x season interaction terms significant")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal interaction regression (single pre-specified model per WQ variable)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "analysis_dataset.csv"
    if not dataset_path.exists():
        raise SystemExit(f"ERROR: {dataset_path} not found. Run `python -m src.merge_dataset --config {args.config}` first.")

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = add_season(df)

    all_results = pd.concat([fit_interaction_model(df, y) for y in WQ_COLS], ignore_index=True)
    all_results.to_csv(tables_dir / "seasonal_regression_results.csv", index=False)

    summary = build_summary(all_results)
    (tables_dir / "seasonal_regression_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
