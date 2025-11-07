"""Visualization utilities for training metrics."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_history(history: dict[str, list], save_dir: Path) -> None:
    """Plot and save loss history chart.

    Args:
        history: Dictionary with 'iters', 'train_loss', 'val_loss' keys
        save_dir: Directory to save the chart
    """
    if len(history["iters"]) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history["iters"], history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["iters"], history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = save_dir / "loss_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Loss chart saved to {chart_path}")
