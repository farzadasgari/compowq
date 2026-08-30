import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loading import load_config
from src.seasonal_analysis import SEASON_MAP, SEASON_ORDER
from src.viz.style import add_panel_letter, apply_style, load_viz_config, save_figure

VARIABLES = {"tsm": "TSM", "chl_a": "Chl-a", "cdom": "CDOM", "sti": "STI", "sapei_9m": "SAPEI-9", "scdhi_9m": "SCDHI-9"}


def plot_triangular_heatmap(ax, corr: pd.DataFrame, base_font: int, annot_size: int) -> None:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white", cbar=False,
        annot_kws={"size": annot_size}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=base_font - 4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=base_font - 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: seasonal correlation heatmaps, 2x2")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    BASE_FONT = viz_cfg["base_font"]

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])
    df["season"] = df["date"].dt.month.map(SEASON_MAP)
    cols = list(VARIABLES.keys())

    apply_style(viz_cfg)
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()

    export_frames = []
    for ax, season, letter in zip(axes, SEASON_ORDER, ["A", "B", "C", "D"]):
        sub = df[df["season"] == season]
        corr = sub[cols].corr(method="pearson")
        corr.index = corr.columns = list(VARIABLES.values())

        plot_triangular_heatmap(ax, corr, BASE_FONT, annot_size=BASE_FONT - 5)
        add_panel_letter(ax, letter, x=-0.15, y=1.05, base_font=BASE_FONT)

        export = corr.copy()
        export.insert(0, "season", season)
        export.insert(1, "n", len(sub))
        export_frames.append(export)

    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.03, aspect=30)
    cbar.set_label("Pearson correlation (r)", fontsize=BASE_FONT - 1)

    out_path = figures_dir / "figure_seasonal_correlation"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    pd.concat(export_frames).to_csv(figures_dir / "figure_seasonal_correlation_matrices.csv")

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {figures_dir / 'figure_seasonal_correlation_matrices.csv'}")


if __name__ == "__main__":
    main()
