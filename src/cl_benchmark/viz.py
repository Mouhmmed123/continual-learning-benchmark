"""Visualization utilities for continual learning benchmark metrics."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_matrix(
    matrix: np.ndarray,
    save_path: str | None = None,
    title: str = "Continual Learning Evaluation Matrix $R$",
    cmap: str = "Blues",
) -> plt.Figure:
    """Render a heatmap of the evaluation matrix R with cell annotations.

    Args:
        matrix: 2D NumPy array `[T, T]` with test accuracies in `[0.0, 1.0]`.
        save_path: Optional file path to save the rendered figure to disk.
        title: Title string for the plot.
        cmap: Matplotlib colormap name.

    Returns:
        Matplotlib Figure object.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    num_tasks = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, num_tasks * 1.2), max(5, num_tasks * 1.0)))
    im = ax.imshow(matrix * 100.0, cmap=cmap, vmin=0.0, vmax=100.0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", rotation=270, labelpad=15, fontweight="bold")

    # Ticks and Labels
    task_labels = [f"Task {i}" for i in range(num_tasks)]
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_tasks))
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticklabels(task_labels, fontsize=10)

    ax.set_xlabel("Evaluated Task ($j$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Trained Task ($i$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Cell text annotations
    threshold = 50.0
    for i in range(num_tasks):
        for j in range(num_tasks):
            val = matrix[i, j] * 100.0
            if i >= j or val > 0:
                text_color = "white" if val > threshold else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Evaluation matrix plot saved to: '{save_path}'")

    return fig


def plot_accuracy_trajectories(
    task_accuracies: Mapping[str, Sequence[float]] | np.ndarray,
    save_path: str | None = None,
    title: str = "Task Accuracy Trajectories Across Stream",
) -> plt.Figure:
    """Plot multi-task accuracy lines over the continual learning stream.

    Args:
        task_accuracies: Task-name-to-accuracy mapping or 2D R matrix `[T, T]`.
        save_path: Optional file path to save the rendered figure to disk.
        title: Title string for the plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    if isinstance(task_accuracies, np.ndarray):
        matrix = task_accuracies
        num_tasks = matrix.shape[0]
        for j in range(num_tasks):
            train_steps = list(range(j, num_tasks))
            accs = [matrix[i, j] * 100.0 for i in train_steps]
            ax.plot(
                train_steps,
                accs,
                marker="o",
                linewidth=1.8,
                label=f"Task {j}",
            )
        # Average accuracy curve
        avg_accs = [np.mean(matrix[i, : i + 1]) * 100.0 for i in range(num_tasks)]
        ax.plot(
            range(num_tasks),
            avg_accs,
            "k--",
            linewidth=2.2,
            label="Average Accuracy ($A_k$)",
        )
        ax.set_xticks(range(num_tasks))
        ax.set_xticklabels([f"Task {i}" for i in range(num_tasks)])
    else:
        for label, accs in task_accuracies.items():
            steps = list(range(len(accs)))
            values = [v * 100.0 if max(accs) <= 1.0 else v for v in accs]
            ax.plot(steps, values, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Trained Task Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Accuracy trajectories plot saved to: '{save_path}'")

    return fig
