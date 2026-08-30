import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.data_loading import load_config
from src.viz.style import add_panel_letter, apply_style, load_viz_config, save_figure, style_ax

WINDOW = "9m"
TREND_SERIES = {"sti": "mean_sti", "sapei_9m": "mean_sapei_9m", "scdhi_9m": "mean_scdhi_9m"}


def get_threshold(thresholds: pd.DataFrame, variable: str, window, set_name: str = "primary") -> float:
    row = thresholds[
        (thresholds["set"] == set_name)
        & (thresholds["variable"] == variable)
        & (thresholds["window"] == window if window is not None else thresholds["window"].isna())
    ]
    return float(row["threshold_value"].iloc[0])


def event_shade(ax, dates, mask, color) -> None:
    ymin, ymax = ax.get_ylim()
    ax.fill_between(dates, ymin, ymax, where=mask.values, color=color, alpha=0.30, linewidth=0)
    ax.set_ylim(ymin, ymax)


def threshold_line(ax, dates, value, color) -> None:
    ax.axhline(value, ls="--", color=color, lw=1.5)
    ax.annotate(
        f"{value:.2f}", xy=(1.0, value), xycoords=("axes fraction", "data"),
        xytext=(0, 7), textcoords="offset points",
        ha="right", va="bottom", fontsize=19, fontweight="bold", color="0.3", clip_on=False,
    )


def trend_line(ax, dates, annual: pd.Series, trend_row: pd.Series, color: str) -> None:
    n_years = len(annual)
    y_start = trend_row["sen_intercept"]
    y_end = trend_row["sen_slope_per_year"] * (n_years - 1) + trend_row["sen_intercept"]
    x_start, x_end = dates.min(), dates.max()
    sig = "significant" if trend_row["significant_05"] else "n.s."
    ax.plot(
        [x_start, x_end], [y_start, y_end], color=color, lw=2.5, ls="-", alpha=0.9, zorder=5,
        label=f"trend: {trend_row['sen_slope_per_year']:+.4f}/yr ({sig})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure: full daily time series with event shading and trend lines")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]

    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_dir / "full_daily_merged.csv", parse_dates=["date"])
    thresholds = pd.read_csv(processed_dir / "event_thresholds.csv")

    trend_path = tables_dir / "trend_analysis_results.csv"
    if not trend_path.exists():
        raise SystemExit(f"ERROR: {trend_path} not found. Run `python -m src.trend_analysis --config {args.config}` first.")
    trend_results = pd.read_csv(trend_path).set_index("series")
    annual_series = pd.read_csv(tables_dir / "trend_analysis_annual_series.csv", index_col=0)

    sti_thr = get_threshold(thresholds, "sti", None)
    sapei_thr = get_threshold(thresholds, "sapei", WINDOW)
    scdhi_thr = get_threshold(thresholds, "scdhi", WINDOW)

    heat = df["sti"] > sti_thr
    drought = df[f"sapei_{WINDOW}"] < sapei_thr
    compound = df[f"scdhi_{WINDOW}"] < scdhi_thr

    apply_style(viz_cfg)
    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True)

    axes[0].plot(df["date"], df["tmean"], color=COLORS["tmean"], lw=1.4)
    axes[0].set_ylabel("Tmean (°C)")
    style_ax(axes[0])
    add_panel_letter(axes[0], "A", base_font=BASE_FONT)

    axes[1].plot(df["date"], df["precipitation"], color=COLORS["precip"], lw=1.2)
    axes[1].set_ylabel("Precipitation (mm)")
    style_ax(axes[1])
    add_panel_letter(axes[1], "B", base_font=BASE_FONT)

    axes[2].plot(df["date"], df["sti"], color=COLORS["sti"], lw=1.6)
    event_shade(axes[2], df["date"], heat, COLORS["heat"])
    threshold_line(axes[2], df["date"], sti_thr, COLORS["threshold"])
    trend_line(axes[2], df["date"], annual_series["mean_sti"], trend_results.loc["mean_sti"], COLORS["trend"])
    axes[2].set_ylabel("STI")
    style_ax(axes[2])
    add_panel_letter(axes[2], "C", base_font=BASE_FONT)

    axes[3].plot(df["date"], df[f"sapei_{WINDOW}"], color=COLORS["sapei"], lw=1.6)
    event_shade(axes[3], df["date"], drought, COLORS["drought"])
    threshold_line(axes[3], df["date"], sapei_thr, COLORS["threshold"])
    trend_line(axes[3], df["date"], annual_series["mean_sapei_9m"], trend_results.loc["mean_sapei_9m"], COLORS["trend"])
    axes[3].set_ylabel(f"SAPEI ({WINDOW})")
    style_ax(axes[3])
    add_panel_letter(axes[3], "D", base_font=BASE_FONT)

    axes[4].plot(df["date"], df[f"scdhi_{WINDOW}"], color=COLORS["scdhi"], lw=1.6)
    event_shade(axes[4], df["date"], compound, COLORS["compound"])
    threshold_line(axes[4], df["date"], scdhi_thr, COLORS["threshold"])
    trend_line(axes[4], df["date"], annual_series["mean_scdhi_9m"], trend_results.loc["mean_scdhi_9m"], COLORS["trend"])
    axes[4].set_ylabel(f"SCDHI ({WINDOW})")
    axes[4].set_xlabel("Time (Year)")
    style_ax(axes[4])
    add_panel_letter(axes[4], "E", base_font=BASE_FONT)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    event_legend_handles = [
        Line2D([0], [0], color=COLORS["heat"], lw=6, alpha=0.5, label="Heat"),
        Line2D([0], [0], color=COLORS["drought"], lw=6, alpha=0.5, label="Drought"),
        Line2D([0], [0], color=COLORS["compound"], lw=6, alpha=0.5, label="Compound"),
        Line2D([0], [0], color=COLORS["threshold"], lw=1.5, ls="--", label="Threshold"),
        Line2D([0], [0], color=COLORS["trend"], lw=2.5, label="Trend (Sen's slope)"),
    ]
    fig.legend(
        handles=event_legend_handles, frameon=False, ncol=5,
        loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=BASE_FONT - 2,
    )

    plt.tight_layout()
    out_path = figures_dir / "figure_timeseries"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    log_path = figures_dir / "figure_timeseries_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset size: {len(df)} days\n")
        f.write(f"Time range: {df['date'].min().date()} -> {df['date'].max().date()}\n\n")
        f.write("---- Thresholds (primary set) ----\n")
        f.write(f"STI 90th percentile (heat): {sti_thr:.4f}\n")
        f.write(f"SAPEI_{WINDOW} 10th percentile (drought): {sapei_thr:.4f}\n")
        f.write(f"SCDHI_{WINDOW} 10th percentile (compound): {scdhi_thr:.4f}\n\n")
        f.write("---- Event day counts ----\n")
        f.write(f"Heat days: {int(heat.sum())} ({100*heat.mean():.2f}%)\n")
        f.write(f"Drought days: {int(drought.sum())} ({100*drought.mean():.2f}%)\n")
        f.write(f"Compound days: {int(compound.sum())} ({100*compound.mean():.2f}%)\n\n")
        f.write("---- Trends (Mann-Kendall / Sen's slope, from src.trend_analysis) ----\n")
        for col, series_name in TREND_SERIES.items():
            row = trend_results.loc[series_name]
            f.write(f"{series_name}: {row['sen_slope_per_year']:+.4f}/year, p={row['mk_p_value']:.4f}, significant={row['significant_05']}\n")

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
