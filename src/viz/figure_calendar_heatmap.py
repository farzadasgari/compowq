import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.compute_sapei import pseudo_doy
from src.data_loading import load_config
from src.viz.style import add_panel_letter, apply_style, load_viz_config, save_figure

WINDOW = "9m"
MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: year x day-of-year calendar heatmap")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    BASE_FONT = viz_cfg["base_font"]

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "full_daily_merged.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = pseudo_doy(df["date"])

    panels = [
        ("sti", "STI", "A", "seismic"),
        (f"sapei_{WINDOW}", f"SAPEI ({WINDOW})", "B", "seismic_r"),
        (f"scdhi_{WINDOW}", f"SCDHI ({WINDOW})", "C", "seismic_r"),
    ]

    apply_style(viz_cfg)
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    for ax, (col, label, letter, cmap) in zip(axes, panels):
        heatmap_df = df.pivot_table(index="year", columns="doy", values=col, aggfunc="mean").reindex(columns=range(1, 366))

        im = ax.imshow(heatmap_df.values, aspect="auto", cmap=cmap, vmin=-3, vmax=3)

        years = heatmap_df.index.values
        yticks = np.arange(0, len(years), 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(years[yticks])
        ax.set_ylabel("Time (Year)")

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(label, rotation=90, labelpad=8)

        add_panel_letter(ax, letter, x=0.005, y=1.07, base_font=BASE_FONT)

    axes[-1].set_xticks(MONTH_STARTS)
    axes[-1].set_xticklabels(MONTH_LABELS, rotation=0)
    axes[-1].set_xlabel("Time (Month)")

    plt.tight_layout()
    out_path = figures_dir / "figure_calendar_heatmap"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    summary_rows = []
    for col, label, _, _ in panels:
        summary_rows.append({
            "series": label, "n_days": int(df[col].notna().sum()),
            "min": df[col].min(), "max": df[col].max(), "mean": df[col].mean(), "std": df[col].std(),
            "year_range": f"{df['year'].min()}-{df['year'].max()}",
        })
    pd.DataFrame(summary_rows).to_csv(figures_dir / "figure_calendar_heatmap_summary.csv", index=False)

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {figures_dir / 'figure_calendar_heatmap_summary.csv'}")


if __name__ == "__main__":
    main()
