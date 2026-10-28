"""Convergence figures. Bands use the sample standard deviation (ddof=1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

try:
    matplotlib.use("Agg", force=True)
except TypeError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sis6116.validation import ValidationError


def convergence_array(histories: list[list[int]]) -> np.ndarray:
    if len(histories) < 2:
        raise ValidationError(
            "sample standard deviation requires at least 2 runs"
        )
    width = len(histories[0])
    if width < 1 or any(len(history) != width for history in histories):
        raise ValidationError("convergence histories must share one length")
    array = np.asarray(histories, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValidationError("convergence histories must be finite")
    return array


def mean_and_sample_std(histories: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    array = convergence_array(histories)
    return array.mean(axis=0), array.std(axis=0, ddof=1)


def plot_mean_std(
    histories: list[list[int]],
    path: Path,
    *,
    nn_city0: float | None = None,
    nn_best: float | None = None,
) -> plt.Figure:
    mean, std = mean_and_sample_std(histories)
    iterations = np.arange(mean.shape[0])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(iterations, mean, color="C0", label="mean best-so-far")
    ax.fill_between(
        iterations,
        mean - std,
        mean + std,
        color="C0",
        alpha=0.25,
        label="sample std (ddof=1)",
    )
    if nn_city0 is not None:
        ax.axhline(nn_city0, color="C1", linestyle="--", label="NN city 0")
    if nn_best is not None:
        ax.axhline(
            nn_best,
            color="C2",
            linestyle=":",
            label="NN best of all starts",
        )
    ax.set_xlabel("Iteration (0 = initial incumbent)")
    ax.set_ylabel("Best-so-far tour cost")
    ax.set_title("MMAS mean convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_overlay(
    histories: list[list[int]],
    seeds: list[int],
    path: Path,
) -> plt.Figure:
    if len(histories) != len(seeds):
        raise ValidationError("overlay requires one seed per history")
    if not histories:
        raise ValidationError("overlay requires at least one run")
    width = len(histories[0])
    if any(len(history) != width for history in histories):
        raise ValidationError("convergence histories must share one length")
    iterations = np.arange(width)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for seed, history in zip(seeds, histories):
        ax.plot(iterations, history, linewidth=1.0, label=f"seed {seed}")
    ax.set_xlabel("Iteration (0 = initial incumbent)")
    ax.set_ylabel("Best-so-far tour cost")
    ax.set_title("MMAS best-so-far by run")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
