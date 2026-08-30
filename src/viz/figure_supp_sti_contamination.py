import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from src.compute_sapei import pseudo_doy
from src.copulas import FAMILIES
from src.data_loading import load_config
from src.viz.style import add_panel_letter, apply_style, load_viz_config, save_figure

WINDOW = "6m"
MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def to_uniform_empirical(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, method="average")
    return ranks / (len(x) + 1)


def compute_mixed_scdhi(old_sti: pd.DataFrame, sapei: pd.DataFrame, window: str) -> pd.DataFrame:
    merged = sapei.merge(old_sti, on="date", how="inner")
    sapei_col = f"sapei_{window}"
    mask = merged[sapei_col].notna() & merged["old_sti"].notna()

    u_all = norm.cdf(merged[sapei_col])
    v_all = norm.cdf(merged["old_sti"])
    u = np.clip(u_all[mask.values], 1e-6, 1 - 1e-6)
    v = np.clip(v_all[mask.values], 1e-6, 1 - 1e-6)

    params = FAMILIES["frank"]["fit"](u, v)
    c = FAMILIES["frank"]["cdf"](u, v, params)
    p = np.clip(u - c, 1e-10, 1 - 1e-10)
    q = to_uniform_empirical(p)

    scdhi = pd.Series(index=merged.index, dtype=float)
    scdhi[mask] = norm.ppf(q)

    out = merged[["date"]].copy()
    out[f"scdhi_{window}"] = scdhi
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplementary figure: STI contamination pathway into SCDHI")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    BASE_FONT = viz_cfg["base_font"]

    processed_dir = Path(cfg["paths"]["processed_dir"])
    raw_dir = Path(cfg["paths"]["raw_wq"]).parent
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    old_ref_path = raw_dir / "cl_old_reference.csv"
    if not old_ref_path.exists():
        raise SystemExit(
            f"ERROR: {old_ref_path} not found. This is the old/untrusted reference file "
            f"used for the contamination-pathway supplementary figure -- place it manually "
            f"(gitignored, same as cl.csv/wq.csv)."
        )

    old = pd.read_csv(old_ref_path, parse_dates=["Date"])
    old_sti = old[["Date", "STI"]].rename(columns={"Date": "date", "STI": "old_sti"})

    sapei = pd.read_csv(processed_dir / "sapei_daily.csv", parse_dates=["date"])

    mixed_scdhi = compute_mixed_scdhi(old_sti, sapei, WINDOW)

    df = old_sti.merge(sapei, on="date").merge(mixed_scdhi, on="date")
    df["year"] = df["date"].dt.year
    df["doy"] = pseudo_doy(df["date"])

    panels = [
        ("old_sti", "STI (old reference file)", "A", "seismic"),
        (f"sapei_{WINDOW}", f"SAPEI ({WINDOW}, our pipeline)", "B", "seismic_r"),
        (f"scdhi_{WINDOW}", f"SCDHI ({WINDOW}, old STI x our SAPEI)", "C", "seismic_r"),
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
        ax.set_ylabel("Year")

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(label, rotation=90, labelpad=8)
        add_panel_letter(ax, letter, x=0.005, y=1.07, base_font=BASE_FONT)

    axes[-1].set_xticks(MONTH_STARTS)
    axes[-1].set_xticklabels(MONTH_LABELS, rotation=0)
    axes[-1].set_xlabel("Month")

    fig.text(
        0.5, -0.015,
        "Panel C is computed fresh from panels A and B (Frank copula), not taken from either source file.\n"
        "Note SCDHI inherits STI's seasonal contamination (compare band position to Panel A) despite a fully clean SAPEI input.",
        ha="center", fontsize=BASE_FONT - 6, style="italic", color="0.3",
    )

    plt.tight_layout()
    out_path = figures_dir / "figure_supp_sti_contamination"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
