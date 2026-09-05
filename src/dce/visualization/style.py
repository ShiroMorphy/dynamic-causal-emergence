"""
Nature / Science Publication Visual Styling Guidelines and Palettes.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_nature_style() -> None:
    """Configure Matplotlib parameters for Nature / Science journal figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10.0,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Colorblind-safe scientific palettes
COLORS = {
    "primary_dark": "#1B365D",      # Deep Navy
    "primary_blue": "#0072B2",      # Blue
    "accent_orange": "#D55E00",     # Vermilion / Orange
    "accent_green": "#009E73",      # Bluish Green
    "accent_yellow": "#E69F00",     # Amber
    "accent_purple": "#CC79A7",     # Reddish Purple
    "neutral_grey": "#7F7F7F",      # Neutral Grey
    "light_grey": "#E0E0E0",        # Light Grid Grey
}
