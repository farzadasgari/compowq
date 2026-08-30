import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loading import load_config
from src.viz.style import apply_style, load_viz_config, save_figure

VARIABLES = {
    "tsm": "TSM",
    "chl_a": "Chl-a",
    "cdom": "CDOM",
    "precipitation": "Precipitation",
    "tmean": "Tmean",
    "sti": "STI",
    "sapei_3m": "SAPEI-3",
    "sapei_6m": "SAPEI-6",
    "sapei_9m": "SAPEI-9",
    "sapei_12m": "SAPEI-12",
    "scdhi_3m": "SCDHI-3",
    "scdhi_6m": "SCDHI-6",
    "scdhi_9m": "SCDHI-9",
    "scdhi_12m": "SCDHI-12",
}


def plot_triangular_heatmap(ax, corr: pd.DataFrame, base_font: int, annot_size: int) -> None:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white", cbar=False,
        annot_kws={"size": annot_size}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=base_font - 4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=base_font - 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: comprehensive triangular Pearson correlation heatmap")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    BASE_FONT = viz_cfg["base_font"]

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "analysis_dataset.csv")
    cols = list(VARIABLES.keys())
    corr = df[cols].corr(method="pearson")
    corr.index = corr.columns = list(VARIABLES.values())

    apply_style(viz_cfg)
    fig, ax = plt.subplots(figsize=(13, 11))
    plot_triangular_heatmap(ax, corr, BASE_FONT, annot_size=BASE_FONT - 9)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pearson correlation (r)", fontsize=BASE_FONT - 2)

    plt.tight_layout()
    out_path = figures_dir / "figure_correlation_heatmap"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    corr.to_csv(figures_dir / "figure_correlation_heatmap_matrix.csv")
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
