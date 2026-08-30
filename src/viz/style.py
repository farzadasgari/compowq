import matplotlib.pyplot as plt
import yaml

DEFAULT_BASE_FONT = 22
DEFAULT_DPI = 600
DEFAULT_COLORS = {
    "sti": "crimson",
    "sapei": "deepskyblue",
    "scdhi": "purple",
    "heat": "crimson",
    "drought": "deepskyblue",
    "compound": "purple",
    "normal": "0.65",
    "tmean": "firebrick",
    "precip": "steelblue",
    "threshold": "0.5",
    "trend": "black",
    "tsm": "#8B4513",
    "chl_a": "#2E8B57",
    "cdom": "#DAA520",
}

CMAP_DIVERGING = "coolwarm"

BASE_FONT = DEFAULT_BASE_FONT
COLORS = DEFAULT_COLORS


def load_viz_config(config_path: str = "configs/config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        viz_cfg = cfg.get("viz", {}) or {}
    except FileNotFoundError:
        viz_cfg = {}

    return {
        "base_font": viz_cfg.get("base_font", DEFAULT_BASE_FONT),
        "dpi": viz_cfg.get("dpi", DEFAULT_DPI),
        "colors": {**DEFAULT_COLORS, **viz_cfg.get("colors", {})},
    }


def apply_style(viz_cfg: dict = None) -> None:
    viz_cfg = viz_cfg or load_viz_config()
    base_font = viz_cfg["base_font"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "DejaVu Serif", "serif"],
            "font.size": base_font,
            "axes.labelsize": base_font,
            "axes.titlesize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
            "legend.fontsize": base_font - 2,
            "axes.linewidth": 1.4,
            "savefig.dpi": viz_cfg["dpi"],
        }
    )


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_letter(ax, letter: str, x: float = 0.01, y: float = 0.96, base_font: int = None) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=(base_font or BASE_FONT) + 2,
        fontweight="bold",
        va="top",
    )


def save_figure(fig, path_no_ext: str, dpi: int = None) -> None:
    d = dpi or DEFAULT_DPI
    fig.savefig(f"{path_no_ext}.png", dpi=d, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")
