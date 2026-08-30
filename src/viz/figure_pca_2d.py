import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats

from src.data_loading import load_config
from src.viz.style import apply_style, load_viz_config, save_figure, style_ax

WINDOW = "9m"
EVENT_ORDER = ["Normal", "Heat", "Drought", "Compound"]
LOADING_LABELS = {"tsm": "TSM", "chl_a": "Chl-a", "cdom": "CDOM"}
LOADING_COLOR_KEYS = {"tsm": "tsm", "chl_a": "chl_a", "cdom": "cdom"}


def build_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Heat"] = df["is_heat_primary"]
    df["Drought"] = df[f"is_drought_{WINDOW}_primary"]
    df["Compound"] = df[f"is_compound_{WINDOW}_primary"]
    df["Normal"] = ~(df["Heat"] | df["Drought"] | df["Compound"])
    return df


def confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str, confidence: float = 0.95,
                        facecolor: str = None, edgecolor: str = None, **kwargs) -> tuple:
    if len(x) < 4:
        return None
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    chi2_val = stats.chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(np.clip(eigvals, 0, None) * chi2_val)
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle = np.degrees(angle_rad)
    cx, cy = x.mean(), y.mean()
    ellipse = Ellipse(
        (cx, cy), width, height, angle=angle,
        facecolor=facecolor if facecolor is not None else color,
        edgecolor=edgecolor if edgecolor is not None else color,
        **kwargs,
    )
    ax.add_patch(ellipse)

    a, b = width / 2, height / 2
    half_x = np.sqrt((a * np.cos(angle_rad)) ** 2 + (b * np.sin(angle_rad)) ** 2)
    half_y = np.sqrt((a * np.sin(angle_rad)) ** 2 + (b * np.cos(angle_rad)) ** 2)
    return (cx - half_x, cx + half_x, cy - half_y, cy + half_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fancy 2D PCA figure with confidence ellipses and loadings")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = load_viz_config(args.config)
    COLORS = viz_cfg["colors"]
    BASE_FONT = viz_cfg["base_font"]
    EVENT_COLORS = {"Normal": COLORS["normal"], "Heat": COLORS["heat"], "Drought": COLORS["drought"], "Compound": COLORS["compound"]}

    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(processed_dir / "analysis_dataset.csv", parse_dates=["date"])
    dataset = build_event_flags(dataset)

    scores = pd.read_csv(tables_dir / "pca_optical_scores.csv", parse_dates=["date"])
    loadings = pd.read_csv(tables_dir / "pca_optical_loadings.csv", index_col=0)
    variance = pd.read_csv(tables_dir / "pca_optical_variance.csv")

    merged = scores.merge(dataset[["date"] + EVENT_ORDER], on="date")
    pc1_var = variance.loc[variance["PC"] == "PC1", "variance_explained"].iloc[0] * 100
    pc2_var = variance.loc[variance["PC"] == "PC2", "variance_explained"].iloc[0] * 100

    apply_style(viz_cfg)
    fig, ax = plt.subplots(figsize=(11, 10))

    ellipse_bounds = []
    centroid_rows = []
    for event in EVENT_ORDER:
        subset = merged[merged[event]]
        color = EVENT_COLORS[event]
        ax.scatter(subset["PC1"], subset["PC2"], s=140, alpha=0.65, color=color, edgecolors="black", linewidth=1.0, zorder=3, label=event)
        confidence_ellipse(ax, subset["PC1"].values, subset["PC2"].values, color=color, alpha=0.12, zorder=1, linewidth=0)
        bounds = confidence_ellipse(ax, subset["PC1"].values, subset["PC2"].values, color=color, facecolor="none", edgecolor=color, alpha=0.9, linewidth=2.2, zorder=2)
        if bounds is not None:
            ellipse_bounds.append(bounds)
        cx, cy = subset["PC1"].mean(), subset["PC2"].mean()
        ax.scatter(cx, cy, s=480, marker="*", color=color, edgecolors="black", linewidth=1.3, zorder=6)
        centroid_rows.append({"event": event, "n": len(subset), "PC1_centroid": cx, "PC2_centroid": cy,
                               "PC1_std": subset["PC1"].std(), "PC2_std": subset["PC2"].std()})

    data_range = max(merged["PC1"].abs().max(), merged["PC2"].abs().max())
    max_loading = loadings[["PC1", "PC2"]].abs().values.max()
    scale = (data_range * 0.65) / max_loading
    for var in loadings.index:
        x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        vec_color = COLORS[LOADING_COLOR_KEYS[var]]
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=vec_color, lw=2.6, mutation_scale=22), zorder=7,
        )
        ax.text(x * 1.12, y * 1.12, LOADING_LABELS[var], fontsize=BASE_FONT, fontweight="bold",
                ha="center", va="center", color=vec_color, zorder=8)

    ax.axhline(0, color="gray", lw=0.8, alpha=0.5, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    style_ax(ax)
    ax.grid(alpha=0.15)

    all_x = list(merged["PC1"]) + [loadings.loc[v, "PC1"] * scale * 1.25 for v in loadings.index]
    all_y = list(merged["PC2"]) + [loadings.loc[v, "PC2"] * scale * 1.25 for v in loadings.index]
    for xmin, xmax, ymin, ymax in ellipse_bounds:
        all_x.extend([xmin, xmax])
        all_y.extend([ymin, ymax])
    pad_x = (max(all_x) - min(all_x)) * 0.08
    pad_y = (max(all_y) - min(all_y)) * 0.08
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=BASE_FONT - 1, markerscale=1.3)

    plt.tight_layout()
    out_path = figures_dir / "figure_pca_2d"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    export = pd.DataFrame(centroid_rows)
    export.to_csv(figures_dir / "figure_pca_2d_centroids.csv", index=False)
    loadings.to_csv(figures_dir / "figure_pca_2d_loadings.csv")

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {figures_dir / 'figure_pca_2d_centroids.csv'}")


if __name__ == "__main__":
    main()
