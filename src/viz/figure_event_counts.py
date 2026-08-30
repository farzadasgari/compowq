import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import load_config
from src.seasonal_analysis import SEASON_MAP, SEASON_ORDER
from src.viz.style import apply_style, load_viz_config, save_figure, style_ax

WINDOW = "9m"
CATEGORIES = ["Normal", "Heat", "Drought", "Compound"]
HATCHES = {"Normal": "", "Heat": "///", "Drought": "xxx", "Compound": "..."}


def build_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Heat"] = df["is_heat_primary"]
    df["Drought"] = df[f"is_drought_{WINDOW}_primary"]
    df["Compound"] = df[f"is_compound_{WINDOW}_primary"]
    df["Normal"] = ~(df["Heat"] | df["Drought"] | df["Compound"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: event frequency, overall and seasonal")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]
    CAT_COLORS = {"Normal": COLORS["normal"], "Heat": COLORS["heat"], "Drought": COLORS["drought"], "Compound": COLORS["compound"]}

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(processed_dir / "events_daily.csv", parse_dates=["date"])
    daily = build_categories(daily)

    acq = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])
    acq = build_categories(acq)
    acq["season"] = acq["date"].dt.month.map(SEASON_MAP)

    n_daily = len(daily)
    n_acq = len(acq)
    overall_counts = {cat: int(daily[cat].sum()) for cat in CATEGORIES}
    seasonal_counts = {
        season: {cat: int(acq.loc[acq["season"] == season, cat].sum()) for cat in CATEGORIES} for season in SEASON_ORDER
    }

    apply_style(viz_cfg)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.1, 2]})

    ax = axes[0]
    x = np.arange(len(CATEGORIES))
    pct = [100 * overall_counts[c] / n_daily for c in CATEGORIES]
    bars = ax.bar(
        x, pct, color=[CAT_COLORS[c] for c in CATEGORIES], edgecolor="black", linewidth=2.0,
        hatch=[HATCHES[c] for c in CATEGORIES], width=0.65,
    )
    for b, cat in zip(bars, CATEGORIES):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + max(pct) * 0.02,
            f"{overall_counts[cat]:,}\n({b.get_height():.1f}%)",
            ha="center", va="bottom", fontsize=BASE_FONT - 5, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=BASE_FONT - 4, rotation=90, ha="center")
    ax.set_ylabel("Share of days (%)")
    ax.set_ylim(0, max(pct) * 1.28)
    ax.set_title(f"Full climatology\n(n = {n_daily:,} days, 1994\u20132025)", fontsize=BASE_FONT - 1)
    style_ax(ax)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(len(SEASON_ORDER))
    width = 0.2
    for i, cat in enumerate(CATEGORIES):
        values = [seasonal_counts[s][cat] for s in SEASON_ORDER]
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset, values, width=width, color=CAT_COLORS[cat], edgecolor="black",
            linewidth=2.0, hatch=HATCHES[cat], label=cat, zorder=3,
        )
        for rect, v in zip(bars, values):
            if v > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6,
                    f"{v}", ha="center", va="bottom", fontsize=BASE_FONT - 6, fontweight="bold",
                )

    season_n = {s: int((acq["season"] == s).sum()) for s in SEASON_ORDER}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={season_n[s]})" for s in SEASON_ORDER], fontsize=BASE_FONT - 2)
    ax.set_ylabel("Number of acquisitions")
    ax.set_title(f"Sentinel-2 acquisitions by season\n(n = {n_acq} total)", fontsize=BASE_FONT - 1)
    style_ax(ax)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLORS[c], edgecolor="black", hatch=HATCHES[c], linewidth=2.0, label=c)
        for c in CATEGORIES
    ]
    fig.legend(
        handles=legend_handles, frameon=False, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=BASE_FONT - 1,
    )

    plt.tight_layout()
    out_path = figures_dir / "figure_event_counts"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    log_path = figures_dir / "figure_event_counts_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Full daily record: n={n_daily}\n")
        for cat in CATEGORIES:
            f.write(f"  {cat}: {overall_counts[cat]} ({100*overall_counts[cat]/n_daily:.2f}%)\n")
        f.write(f"\nSentinel-2 acquisitions: n={n_acq}\n")
        for season in SEASON_ORDER:
            f.write(f"\n{season} (n={season_n[season]}):\n")
            for cat in CATEGORIES:
                f.write(f"  {cat}: {seasonal_counts[season][cat]}\n")

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
