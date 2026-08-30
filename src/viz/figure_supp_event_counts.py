import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import load_config
from src.seasonal_analysis import SEASON_MAP, SEASON_ORDER
from src.viz.figure_supp_sti_contamination import compute_mixed_scdhi
from src.viz.style import apply_style, load_viz_config, save_figure, style_ax

WINDOW = "6m"
CATEGORIES = ["Normal", "Heat", "Drought", "Compound"]
HATCHES = {"Normal": "", "Heat": "///", "Drought": "xxx", "Compound": "..."}


def build_supp_categories(daily_full: pd.DataFrame) -> pd.DataFrame:
    df = daily_full.copy()
    heat_thr = df["old_sti"].quantile(0.90)
    compound_thr = df["scdhi_6m"].quantile(0.10)

    df["Heat"] = df["old_sti"] > heat_thr
    df["Drought"] = df["is_drought_6m_primary"]
    df["Compound"] = df["scdhi_6m"] < compound_thr
    df["Normal"] = ~(df["Heat"] | df["Drought"] | df["Compound"])
    return df, heat_thr, compound_thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Supp Data version of the event counts figure")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]
    CAT_COLORS = {"Normal": COLORS["normal"], "Heat": COLORS["heat"], "Drought": COLORS["drought"], "Compound": COLORS["compound"]}

    processed_dir = Path(cfg["paths"]["processed_dir"])
    raw_dir = Path(cfg["paths"]["raw_wq"]).parent
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    old_ref_path = raw_dir / "cl_old_reference.csv"
    if not old_ref_path.exists():
        raise SystemExit(f"ERROR: {old_ref_path} not found. Supp Data requires this reference file (gitignored, place manually).")

    old = pd.read_csv(old_ref_path, parse_dates=["Date"])
    old_sti = old[["Date", "STI"]].rename(columns={"Date": "date", "STI": "old_sti"})

    sapei = pd.read_csv(processed_dir / "sapei_daily.csv", parse_dates=["date"])
    mixed_scdhi = compute_mixed_scdhi(old_sti, sapei, WINDOW)

    events = pd.read_csv(processed_dir / "events_daily.csv", parse_dates=["date"])
    daily = old_sti.merge(events[["date", f"is_drought_{WINDOW}_primary"]], on="date").merge(mixed_scdhi, on="date")
    daily = daily.rename(columns={f"is_drought_{WINDOW}_primary": "is_drought_6m_primary"})
    daily, heat_thr, compound_thr = build_supp_categories(daily)

    wq = pd.read_csv(processed_dir / "wq_clean.csv", parse_dates=["date"])
    acq = wq.merge(daily[["date"] + CATEGORIES], on="date", how="inner")
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
    bars = ax.bar(x, pct, color=[CAT_COLORS[c] for c in CATEGORIES], edgecolor="black", linewidth=1.1,
                  hatch=[HATCHES[c] for c in CATEGORIES], width=0.65)
    for b, cat in zip(bars, CATEGORIES):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(pct) * 0.02,
                f"{overall_counts[cat]:,}\n({b.get_height():.1f}%)", ha="center", va="bottom",
                fontsize=BASE_FONT - 5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=BASE_FONT - 4, rotation=20, ha="right")
    ax.set_ylabel("Share of days (%)")
    ax.set_ylim(0, max(pct) * 1.28)
    ax.set_title(f"Full climatology -- SUPP DATA\n(n = {n_daily:,} days, 1994\u20132025)", fontsize=BASE_FONT - 1, color="0.25")
    style_ax(ax)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.arange(len(SEASON_ORDER))
    width = 0.2
    for i, cat in enumerate(CATEGORIES):
        values = [seasonal_counts[s][cat] for s in SEASON_ORDER]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width=width, color=CAT_COLORS[cat], edgecolor="black",
                      linewidth=1.0, hatch=HATCHES[cat], label=cat, zorder=3)
        for rect, v in zip(bars, values):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6, f"{v}",
                        ha="center", va="bottom", fontsize=BASE_FONT - 6, fontweight="bold")

    season_n = {s: int((acq["season"] == s).sum()) for s in SEASON_ORDER}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={season_n[s]})" for s in SEASON_ORDER], fontsize=BASE_FONT - 2)
    ax.set_ylabel("Number of acquisitions")
    ax.set_title(f"Sentinel-2 acquisitions by season -- SUPP DATA\n(n = {n_acq} total)", fontsize=BASE_FONT - 1, color="0.25")
    style_ax(ax)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLORS[c], edgecolor="black", hatch=HATCHES[c], linewidth=1.0, label=c)
        for c in CATEGORIES
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=BASE_FONT - 1)
    fig.text(0.5, 1.115, "SUPP DATA: Heat from buggy (non-deseasonalized) STI; Drought unchanged; Compound from SCDHI(buggy STI x our SAPEI)",
              ha="center", fontsize=BASE_FONT - 6, style="italic", color="0.4")

    plt.tight_layout()
    out_path = figures_dir / "figure_supp_event_counts"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    log_path = figures_dir / "figure_supp_event_counts_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"SUPP DATA event counts (Heat: buggy STI, threshold={heat_thr:.4f}; Compound: mixed SCDHI, threshold={compound_thr:.4f})\n\n")
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
