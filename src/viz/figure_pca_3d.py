import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from src.data_loading import load_config
from src.viz.style import apply_style, load_viz_config, save_figure

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fancy 3D PCA figure with convex hulls and loading vectors")
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
    pc_var = {row["PC"]: row["variance_explained"] * 100 for _, row in variance.iterrows()}

    apply_style(viz_cfg)
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection="3d")

    centroids = {}
    hull_volumes = {}
    for event in EVENT_ORDER:
        subset = merged[merged[event]]
        pts = subset[["PC1", "PC2", "PC3"]].values
        color = EVENT_COLORS[event]

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, alpha=0.4, color=color, edgecolors="none", zorder=2)

        if event != "Normal" and len(pts) >= 4:
            hull = ConvexHull(pts)
            hull_volumes[event] = hull.volume
            faces = [pts[simplex] for simplex in hull.simplices]
            poly = Poly3DCollection(faces, facecolor=color, edgecolor=color, linewidth=0.6, alpha=0.12)
            ax.add_collection3d(poly)
        elif len(pts) >= 4:
            hull_volumes[event] = ConvexHull(pts).volume

        cx, cy, cz = pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()
        centroids[event] = (cx, cy, cz)
        ax.scatter([cx], [cy], [cz], s=400, marker="*", color=color, edgecolors="black", linewidth=1.2, zorder=10)

    for i, e1 in enumerate(EVENT_ORDER):
        for e2 in EVENT_ORDER[i + 1:]:
            p1, p2 = centroids[e1], centroids[e2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black", alpha=0.25, lw=1.0, ls="--", zorder=5)

    robust_range = np.percentile(np.abs(merged[["PC1", "PC2", "PC3"]].values), 95)
    max_loading = loadings[["PC1", "PC2", "PC3"]].abs().values.max()
    scale = (robust_range * 0.7) / max_loading
    for var in loadings.index:
        x, y, z = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale, loadings.loc[var, "PC3"] * scale
        vec_color = COLORS[LOADING_COLOR_KEYS[var]]
        ax.quiver(0, 0, 0, x, y, z, color=vec_color, linewidth=3.2, arrow_length_ratio=0.30, zorder=11)
        ax.text(x * 1.15, y * 1.15, z * 1.15, LOADING_LABELS[var], fontsize=BASE_FONT, fontweight="bold", color=vec_color, zorder=12)

    for axis, col in zip(["x", "y", "z"], ["PC1", "PC2", "PC3"]):
        lo, hi = merged[col].quantile(0.02), merged[col].quantile(0.98)
        pad = (hi - lo) * 0.35
        getattr(ax, f"set_{axis}lim")(lo - pad, hi + pad)

    ax.set_xlabel(f"PC1 ({pc_var['PC1']:.1f}%)", labelpad=16)
    ax.set_ylabel(f"PC2 ({pc_var['PC2']:.1f}%)", labelpad=16)
    ax.text2D(1.10, 0.5, f"PC3 ({pc_var['PC3']:.1f}%)", transform=ax.transAxes,
              rotation=90, va="center", ha="center", fontsize=BASE_FONT)
    ax.tick_params(axis="z", pad=8)
    ax.view_init(elev=28, azim=-40)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=EVENT_COLORS[e], markersize=11, label=e) for e in EVENT_ORDER]
    legend_handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=16, label="Centroid"))
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=BASE_FONT - 3)

    fig.subplots_adjust(left=0.02, right=0.72, top=0.95, bottom=0.05)
    out_path = figures_dir / "figure_pca_3d"
    save_figure(fig, str(out_path), dpi=viz_cfg["dpi"])
    plt.close(fig)

    export_rows = []
    for event in EVENT_ORDER:
        n = int(merged[event].sum())
        cx, cy, cz = centroids[event]
        export_rows.append({
            "event": event, "n": n, "PC1_centroid": cx, "PC2_centroid": cy, "PC3_centroid": cz,
            "convex_hull_volume": hull_volumes.get(event, None),
        })
    pd.DataFrame(export_rows).to_csv(figures_dir / "figure_pca_3d_centroids_and_hull_volumes.csv", index=False)

    print(f"Wrote {out_path}.png / .pdf")
    print(f"Wrote {figures_dir / 'figure_pca_3d_centroids_and_hull_volumes.csv'}")


if __name__ == "__main__":
    main()
