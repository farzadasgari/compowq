import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import load_config
from src.viz.style import apply_style, load_viz_config, save_figure, style_ax

EVENT_TYPE_ORDER = ["heat", "heatwave", "drought", "compound"]


def variant_label(row: pd.Series) -> str:
    if row["event_type"] == "heat":
        return f"Heat ({row['variant']})"
    if row["event_type"] == "heatwave":
        return f"Heatwave episode ({row['variant']})"
    prefix = "Drought" if row["event_type"] == "drought" else "Compound"
    return f"{prefix} {row['variant']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: comprehensive PERMANOVA/PERMDISP results, all variants")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]
    EVENT_COLORS = {"heat": COLORS["heat"], "heatwave": "darkorange", "drought": COLORS["drought"], "compound": COLORS["compound"]}

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "permanova_permdisp_long.csv")
    df["event_type"] = pd.Categorical(df["event_type"], categories=EVENT_TYPE_ORDER, ordered=True)
    df = df.sort_values(["event_type", "variant"]).reset_index(drop=True)
    df["label"] = df.apply(variant_label, axis=1)
    df["color"] = df["event_type"].map(EVENT_COLORS)

    y = np.arange(len(df))[::-1]  # top-to-bottom reading order

    apply_style(viz_cfg)
    fig, axes = plt.subplots(1, 2, figsize=(22, 12), sharey=True)

    panels = [
        ("permanova_F", "PERMANOVA pseudo-F", False),
        # ("permanova_q", "PERMANOVA q-value", True),
        ("permdisp_F", "PERMDISP pseudo-F", False),
        # ("permdisp_q", "PERMDISP q-value", True),
    ]

    for ax, (col, title, is_q) in zip(axes, panels):
        ax.hlines(y, 0, df[col], color=df["color"], alpha=0.8, linewidth=5, zorder=2)
        ax.scatter(df[col], y, color=df["color"], s=120, edgecolors="black", linewidth=1.5, zorder=3)
        if is_q:
            ax.axvline(0.05, color="black", ls="--", lw=1.8, zorder=1)
            ax.text(0.065, len(df) * 0.5, "q=0.05", rotation=90, ha="left", va="center", fontsize=BASE_FONT - 4, fontweight="bold")
            ax.set_xlim(0, 1.05)
        ax.set_xlabel(title)
        style_ax(ax)
        ax.grid(axis="x", alpha=0.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"], fontsize=BASE_FONT - 6)

    legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=EVENT_COLORS[e], markersize=13, label=e.capitalize()) for e in EVENT_TYPE_ORDER]
    fig.legend(handles=legend_handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=BASE_FONT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = figures_dir / "figure_permanova_permdisp"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    export_cols = ["event_type", "variant", "n_event", "n_non_event", "permanova_F", "permanova_q",
                   "permanova_significant_fdr_05", "permdisp_F", "permdisp_q", "permdisp_significant_fdr_05"]
    df[export_cols].to_csv(figures_dir / "figure_permanova_permdisp_export.csv", index=False)

    n_sig_permanova = int(df["permanova_significant_fdr_05"].sum())
    n_sig_permdisp = int(df["permdisp_significant_fdr_05"].sum())
    print(f"Wrote {out_path}.png / .pdf")
    print(f"PERMANOVA significant: {n_sig_permanova}/30, PERMDISP significant: {n_sig_permdisp}/30")


if __name__ == "__main__":
    main()
