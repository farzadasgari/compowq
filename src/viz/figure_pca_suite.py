import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.data_loading import load_config
from src.viz.style import add_panel_letter, apply_style, load_viz_config, save_figure, style_ax

WINDOW = "6m"
LOADING_COLORS = {"tsm": "#8B4513", "chl_a": "#2E8B57", "cdom": "#DAA520"}
LOADING_LABELS = {"tsm": "TSM", "chl_a": "Chl-a", "cdom": "CDOM"}
EVENT_COLORS = {"Normal": "#55A868", "Heat": "crimson", "Drought": "deepskyblue", "Compound": "purple"}


def build_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Heat"] = df["is_heat_primary"]
    df["Drought"] = df[f"is_drought_{WINDOW}_primary"]
    df["Compound"] = df[f"is_compound_{WINDOW}_primary"]
    df["Normal"] = ~(df["Heat"] | df["Drought"] | df["Compound"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: optical-only PCA suite (scree, biplot, event-state scores)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    BASE_FONT = viz_cfg["base_font"]
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])
    dataset = build_event_flags(dataset)

    scores = pd.read_csv(tables_dir / "pca_optical_scores.csv", parse_dates=["date"])
    loadings = pd.read_csv(tables_dir / "pca_optical_loadings.csv", index_col=0)
    variance = pd.read_csv(tables_dir / "pca_optical_variance.csv")
    permanova = pd.read_csv(tables_dir / "permanova_permdisp_long.csv")

    merged = scores.merge(dataset[["date", "Normal", "Heat", "Drought", "Compound"]], on="date")

    pc1_var = variance.loc[variance["PC"] == "PC1", "variance_explained"].iloc[0] * 100
    pc2_var = variance.loc[variance["PC"] == "PC2", "variance_explained"].iloc[0] * 100

    drought_row = permanova[(permanova["event_type"] == "drought") & (permanova["variant"] == f"{WINDOW}_primary")].iloc[0]
    compound_row = permanova[(permanova["event_type"] == "compound") & (permanova["variant"] == f"{WINDOW}_primary")].iloc[0]

    apply_style(viz_cfg)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    ax = axes[0]
    pcs = np.arange(1, len(variance) + 1)
    pct = variance["variance_explained"].values * 100
    ax.plot(pcs, pct, "-o", color="black", linewidth=2.5, markersize=10, markerfacecolor="crimson", markeredgecolor="black")
    for i, v in enumerate(pct):
        ax.text(pcs[i] + .1, v + 5, f"{v:.1f}%", ha="center", fontsize=BASE_FONT - 4)
    ax.set_xticks(pcs)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    style_ax(ax)
    add_panel_letter(ax, "A")

    ax = axes[1]
    ax.scatter(merged["PC1"], merged["PC2"], s=60, alpha=0.25, color="lightgray", edgecolors="none")
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    scale = 3.5
    legend_handles = []
    for var in loadings.index:
        x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        ax.arrow(0, 0, x, y, color=LOADING_COLORS[var], linewidth=2.5, head_width=0.10,
                  head_length=0.14, length_includes_head=True)
        legend_handles.append(Line2D([0], [0], color=LOADING_COLORS[var], lw=3, label=LOADING_LABELS[var]))
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    style_ax(ax)
    add_panel_letter(ax, "B")

    ax = axes[2]
    for event in ["Normal", "Heat", "Drought", "Compound"]:
        subset = merged[merged[event]]
        ax.scatter(subset["PC1"], subset["PC2"], s=55, alpha=0.35, color=EVENT_COLORS[event],
                   edgecolors="none", zorder=2)

    legend_handles = []
    for event in ["Normal", "Heat", "Drought", "Compound"]:
        subset = merged[merged[event]]
        cx, cy = subset["PC1"].mean(), subset["PC2"].mean()
        ax.scatter(cx, cy, s=420, marker="*", color=EVENT_COLORS[event], edgecolors="black",
                  linewidth=1.2, zorder=5)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=EVENT_COLORS[event],
                   markersize=11, label=event)
        )
    legend_handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="0.4",
                                  markeredgecolor="black", markersize=16, label="Group centroid"))
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=BASE_FONT - 4)
    style_ax(ax)
    add_panel_letter(ax, "C")
    ax.text(
        0.98, 0.02,
        f"PERMANOVA (optical space), corrected SAPEI/SCDHI:\n"
        f"drought {WINDOW}: q={drought_row['permanova_q']:.3f} (n.s.)\n"
        f"compound {WINDOW}: q={compound_row['permanova_q']:.3f} (n.s.)\n"
        f"closest overall: drought 6m_strict, q=0.060",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=BASE_FONT - 5, color="0.25",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.85),
    )

    plt.tight_layout()
    out_path = figures_dir / "figure_pca_suite"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
