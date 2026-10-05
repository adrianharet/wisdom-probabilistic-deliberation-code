from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

OUTPUT_DIR = Path(__file__).resolve().parent
FIG_SIZE = (6, 4)
CUSTOM_PALETTE = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]



def plot_binomial_survival_over_p(
    n_values: Sequence[int],
    *,
    threshold: int | None = None,
    threshold_fn: Callable[[int], int] | None = None,
    p_values: Sequence[float] | None = None,
    palette: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    output_path: str | Path | None = None,
) -> plt.Axes:
    """Plot ``Pr[X_n > t_n]`` against ``p`` for ``X_n ~ Binomial(n, p)``.

    By default, the threshold is the strict-majority cutoff ``floor(n / 2)``.
    Use ``threshold`` for a fixed cutoff across all curves or ``threshold_fn`` for
    an ``n``-dependent cutoff.
    """

    if not n_values:
        raise ValueError("n_values must contain at least one value.")
    if threshold is not None and threshold_fn is not None:
        raise ValueError("Provide at most one of threshold and threshold_fn.")

    validated_n_values = []
    for n in n_values:
        if n <= 0:
            raise ValueError("Each n must be positive.")
        validated_n_values.append(int(n))

    if p_values is None:
        p_grid = np.linspace(0.0, 1.0, 501)
    else:
        p_grid = np.asarray(p_values, dtype=float)
        if p_grid.ndim != 1 or p_grid.size == 0:
            raise ValueError("p_values must be a non-empty one-dimensional sequence.")
        if np.any((p_grid < 0.0) | (p_grid > 1.0)):
            raise ValueError("Each p must lie in [0, 1].")

    if threshold_fn is None:
        if threshold is None:
            threshold_fn = lambda n: n // 2
        else:
            threshold_fn = lambda _n: threshold

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.figure

    ax.axhline(0.5, color="gray", linestyle="-", linewidth=1.5, alpha=0.5)
    colors = _resolve_line_colors(len(validated_n_values), palette)

    for n, color in zip(validated_n_values, colors, strict=True):
        cutoff = int(threshold_fn(n))
        if not 0 <= cutoff <= n:
            raise ValueError(f"Threshold for n={n} must satisfy 0 <= threshold <= n.")
        survival = binom.sf(cutoff, n, p_grid)
        ax.plot(
            p_grid,
            survival,
            linewidth=2.0,
            color=color,
            label=fr"$n={n}$",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\Pr[\mathsf{Vote}(n) = a]$")
    # ax.set_title(r"Binomial survival function as a function of $p$")
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(loc="lower left")

    save_path = (
        Path(output_path)
        if output_path is not None
        else OUTPUT_DIR / "binomial_survival_over_p.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=True)

    if created_figure:
        plt.close(fig)

    return ax


def plot_binomial_survival_over_n(
    n_values: Sequence[int],
    p_values: Sequence[float],
    *,
    threshold: int | None = None,
    threshold_fn: Callable[[int], int] | None = None,
    palette: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    output_path: str | Path | None = None,
) -> plt.Axes:
    """Plot ``Pr[X_n > t_n]`` against ``n`` with one curve for each value of ``p``."""

    if not n_values:
        raise ValueError("n_values must contain at least one value.")
    if not p_values:
        raise ValueError("p_values must contain at least one value.")
    if threshold is not None and threshold_fn is not None:
        raise ValueError("Provide at most one of threshold and threshold_fn.")

    validated_n_values = []
    for n in n_values:
        if n <= 0:
            raise ValueError("Each n must be positive.")
        validated_n_values.append(int(n))

    validated_p_values = []
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Each p must lie in [0, 1].")
        validated_p_values.append(float(p))

    if threshold_fn is None:
        if threshold is None:
            threshold_fn = lambda n: n // 2
        else:
            threshold_fn = lambda _n: threshold

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.figure

    thresholds = [int(threshold_fn(n)) for n in validated_n_values]
    for n, cutoff in zip(validated_n_values, thresholds, strict=True):
        if not 0 <= cutoff <= n:
            raise ValueError(f"Threshold for n={n} must satisfy 0 <= threshold <= n.")
    colors = _resolve_line_colors(len(validated_p_values), palette)

    ax.axhline(
        1, 
        color="gray", 
        linestyle="-", 
        linewidth=1.5, 
        alpha=1
    )

    for p, color in zip(validated_p_values, colors, strict=True):
        survival = [
            binom.sf(cutoff, n, p)
            for n, cutoff in zip(validated_n_values, thresholds, strict=True)
        ]
        ax.plot(
            validated_n_values,
            survival,
            marker="o",
            linewidth=2.0,
            color=color,
            label=fr"$p={p:g}$",
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\Pr[Vote(n) = a]$")
    # ax.set_title(r"Binomial survival function as a function of $n$")
    ax.set_xticks(validated_n_values)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(loc="lower left")

    save_path = (
        Path(output_path)
        if output_path is not None
        else OUTPUT_DIR / f"binomial_survival_over_n_{len(p_values)}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path, 
        dpi=500, 
        bbox_inches="tight", 
        transparent=True
        )

    if created_figure:
        plt.close(fig)

    return ax


def _resolve_line_colors(curve_count: int, palette: Sequence[str] | None) -> list[str | None]:
    """Return one color per curve, or ``None`` to use Matplotlib defaults."""

    if palette is None:
        return [None] * curve_count
    if len(palette) < curve_count:
        raise ValueError("palette must have at least as many colors as plotted curves.")
    return list(palette[:curve_count])


if __name__ == "__main__":
    plot_binomial_survival_over_p(
        n_values=[1, 3, 5, 11, 25],
        palette=CUSTOM_PALETTE,
        )


    p_values_list = [
        [0.7],
        [0.7, 0.6],
        [0.7, 0.6, 0.3]
    ]

    for p_values in p_values_list:
        plot_binomial_survival_over_n(
        n_values=range(1, 32, 2),
        p_values=p_values,
        palette=CUSTOM_PALETTE,
        )
