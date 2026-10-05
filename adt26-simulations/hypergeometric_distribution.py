from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom

FIGSIZE = (6, 4)
OUTPUT_DIR = Path(__file__).resolve().parent
DPI = 400
DEFAULT_PALETTE = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]


def _resolve_palette(
    num_colors: int,
    palette: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    colors = list(palette) if palette is not None else list(DEFAULT_PALETTE)
    if not colors:
        raise ValueError("palette must contain at least one color.")
    return [colors[index % len(colors)] for index in range(num_colors)]

def plot_hypergeometric_survival_function(
    m: int,
    l_values: list[int] | tuple[int, ...],
    k: int,
    *,
    ax: plt.Axes | None = None,
    output_path: str | Path | None = None,
    palette: tuple[str, ...] | list[str] | None = None,
) -> plt.Axes:
    """Plot the survival function for each Hypergeometric(m, l, k) in ``l_values``."""

    if m <= 0:
        raise ValueError("m must be positive.")
    if not 0 <= k <= m:
        raise ValueError("k must satisfy 0 <= k <= m.")
    if not l_values:
        raise ValueError("l_values must contain at least one value.")

    for l in l_values:
        if not 0 <= l <= m:
            raise ValueError("each l must satisfy 0 <= l <= m.")

    lower = min(max(0, k - (m - l)) for l in l_values)
    upper = max(min(k, l) for l in l_values)
    x = np.arange(lower, upper + 1)

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure
    colors = _resolve_palette(len(l_values), palette)

    ax.axhline(0.5, color="gray", linestyle="-", linewidth=1.0, alpha=0.75)

    for l, color in zip(l_values, colors):
        survival = hypergeom.sf(x, m, l, k)
        ax.plot(
            x,
            survival,
            marker="o",
            linewidth=1.5,
            color=color,
            label=fr"$\ell={l}$",
        )

    ax.set_xlabel(f"$j$ (number of successes in sample)")
    ax.set_ylabel(f"$\\Pr[S > j]$")
    ax.set_title(f"Survival function of $S \\sim Hypergeometric(m={m}, k={k})$")
    ax.set_xticks(x)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()

    save_path = Path(output_path) if output_path is not None else OUTPUT_DIR / "plots/hypergeometric_survival_function.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    if created_figure:
        plt.close(fig)

    return ax


def plot_threshold_survival_over_k(
    m: int,
    l_values: list[int] | tuple[int, ...],
    k_values: list[int] | tuple[int, ...] | range,
    *,
    ax: plt.Axes | None = None,
    output_path: str | Path | None = None,
    palette: tuple[str, ...] | list[str] | None = None,
) -> plt.Axes:
    """Plot Pr[S > k//2] as k varies, with one curve for each l in ``l_values``."""

    if m <= 0:
        raise ValueError("m must be positive.")
    if not l_values:
        raise ValueError("l_values must contain at least one value.")
    if not k_values:
        raise ValueError("k_values must contain at least one value.")

    for l in l_values:
        if not 0 <= l <= m:
            raise ValueError("each l must satisfy 0 <= l <= m.")
    for k in k_values:
        if not 0 <= k <= m:
            raise ValueError("each k must satisfy 0 <= k <= m.")

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure
    colors = _resolve_palette(len(l_values), palette)

    ax.axhline(0.5, color="gray", linestyle="-", linewidth=2.0, alpha=0.75)

    for l, color in zip(l_values, colors):
        # denominator = l - (m/2)
        # if denominator != 0:
        #     threshold = m / denominator
        #     ax.axvline(
        #         threshold,
        #         color=DEFAULT_PALETTE[2], 
        #         linewidth=2, 
        #         linestyle=":", 
        #         alpha=0.8
        #     )
        #     x_offset = 0.01 * (max(k_values) - min(k_values))
        #     ax.text(
        #         threshold+x_offset,
        #         0.03,
        #         rf"$\left(\frac{{\ell}}{{m}} - \frac{{1}}{{2}}\right)^{{-1}}$",
        #         rotation=0,
        #         color=DEFAULT_PALETTE[0],
        #         va="bottom",
        #         ha="left",
        #         fontsize=14,
        #         alpha=0.9,
        #     )

        survival = [hypergeom.sf(k // 2, m, l, k) for k in k_values]
        ax.plot(
            k_values,
            survival,
            marker="o",
            linewidth=1.5,
            color=color,
            label=fr"$\ell={l}$",
        )

    ax.set_xlabel(f"$k$")
    ax.set_ylabel(rf"Initial accuracy per agent")
    ax.set_title(rf"$m={m}$") # initial accuracy
    ax.set_xticks(k_values[::2])
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()

    save_path = Path(output_path) if output_path is not None else OUTPUT_DIR / "plots/threshold_survival_over_k.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", transparent=True)

    if created_figure:
        plt.close(fig)

    return ax


if __name__ == "__main__":
    # plot_hypergeometric_survival_function(
    #     m=100, 
    #     l_values=[60, 70, 75], 
    #     k=10
    # )

    m = 100
    plot_threshold_survival_over_k(
        m=m,
        l_values=[m//2 + 2],
        k_values=list(range(1, m, 4)) + [99],
        palette=DEFAULT_PALETTE
    )
