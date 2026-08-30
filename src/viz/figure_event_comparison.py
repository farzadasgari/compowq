import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import load_config
from src.viz.style import apply_style, load_viz_config, save_figure, style_ax

WINDOW = "9m"
VARS = [("tsm", "TSM", "g/m$^3$"), ("cdom", "CDOM", "m$^{-1}$"), ("chl_a", "Chl-a", "mg/m$^3$")]
EVENT_LABELS = ["All", "Normal", "Heat", "Drought", "Compound"]
HATCHES = {"All": "\\\\", "Normal": "", "Heat": "///", "Drought": "xxx", "Compound": "..."}

STAT_STYLE = {
    "Mean": {"color": "#2C3E50", "marker": "o", "ls": "-"},
    "STD": {"color": "#2980B9", "marker": "^", "ls": "--"},
    "Skewness": {"color": "#C0392B", "marker": "s", "ls": ":"},
}


def build_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["All"] = True
    df["Heat"] = df["is_heat_primary"]
    df["Drought"] = df[f"is_drought_{WINDOW}_primary"]
    df["Compound"] = df[f"is_compound_{WINDOW}_primary"]
    df["Normal"] = ~(df["Heat"] | df["Drought"] | df["Compound"])
    return df


def compute_stats(data: pd.DataFrame, var: str) -> dict:
    s = data[var].dropna()
    return {"count": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(), "skew": s.skew()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: event-conditioned WQ statistics")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]
    EVENT_COLORS = {"All": COLORS["all"], "Normal": COLORS["normal"], "Heat": COLORS["heat"],
                     "Drought": COLORS["drought"], "Compound": COLORS["compound"]}

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])
    df = build_event_flags(df)

    log_rows = []
    for var, _, _ in VARS:
        for event in EVENT_LABELS:
            stats = compute_stats(df[df[event]], var)
            log_rows.append({"variable": var, "event": event, **stats})

    apply_style(viz_cfg)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    for i, (var, var_label, unit) in enumerate(VARS):
        ax = axes[i, 0]
        x = np.arange(len(EVENT_LABELS))
        means, stds, skews = [], [], []
        for event in EVENT_LABELS:
            data = df.loc[df[event], var].dropna()
            means.append(data.mean())
            stds.append(data.std())
            skews.append(data.skew())

        for label, values in [("Mean", means), ("STD", stds), ("Skewness", skews)]:
            st = STAT_STYLE[label]
            ax.plot(x, values, marker=st["marker"], ls=st["ls"], color=st["color"],
                    linewidth=4.2, markersize=15, markeredgecolor="black", markeredgewidth=0.8, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(EVENT_LABELS, fontsize=BASE_FONT - 3, rotation=90)
        ax.set_ylabel(f"{var_label} ({unit})")
        ax.grid(alpha=0.2)
        style_ax(ax)
        ax.text(0.02, 1.1, chr(65 + 2 * i), transform=ax.transAxes, fontsize=BASE_FONT + 3, fontweight="bold", va="top")
        if i == 0:
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=BASE_FONT - 2)
        ax = axes[i, 1]
        values = [df.loc[df[event], var].dropna() for event in EVENT_LABELS]
        bp = ax.boxplot(values, patch_artist=True, widths=0.6, showfliers=False, zorder=3)
        for patch, event in zip(bp["boxes"], EVENT_LABELS):
            patch.set_facecolor(EVENT_COLORS[event])
            patch.set_edgecolor("black")
            patch.set_linewidth(1.1)
            patch.set_hatch(HATCHES[event])
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2.5)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set_color("#444444")
                line.set_linewidth(1.6)

        for j, event in enumerate(EVENT_LABELS, start=1):
            box_data = values[j - 1]
            if len(box_data):
                med = np.median(box_data)
                q3 = np.percentile(box_data, 75)
                ax.text(
                    j, q3, f"{med:.2f}", ha="left", va="bottom", fontsize=BASE_FONT - 5,
                    fontweight="bold", color="black", zorder=6,
                    # bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.65),
                )

        ax.set_xticks(range(1, len(EVENT_LABELS) + 1))
        ax.set_xticklabels(EVENT_LABELS, fontsize=BASE_FONT - 3, rotation=90)
        ax.set_ylabel(f"{var_label} ({unit})")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.set_axisbelow(True)
        style_ax(ax)
        ax.text(0.02, 1.1, chr(65 + 2 * i + 1), transform=ax.transAxes, fontsize=BASE_FONT + 3, fontweight="bold", va="top")

    plt.tight_layout()
    out_path = figures_dir / "figure_event_comparison"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    log_path = figures_dir / "figure_event_comparison_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
