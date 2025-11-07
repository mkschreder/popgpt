"""Tests for visualization utilities."""

from pathlib import Path

import pytest

from popgpt.utils.visualization import plot_loss_history


class TestPlotLossHistory:
    """Tests for loss history plotting."""

    def test_creates_chart_file(self, tmp_path: Path) -> None:
        """plot_loss_history creates PNG file."""
        # Arrange
        history = {
            "iters": [0, 100, 200],
            "train_loss": [2.5, 2.0, 1.5],
            "val_loss": [2.6, 2.1, 1.6],
        }

        # Act
        plot_loss_history(history, tmp_path)

        # Assert
        chart_path = tmp_path / "loss_chart.png"
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0

    def test_handles_empty_history(self, tmp_path: Path) -> None:
        """plot_loss_history handles empty history gracefully."""
        # Arrange
        empty_history = {"iters": [], "train_loss": [], "val_loss": []}

        # Act
        plot_loss_history(empty_history, tmp_path)

        # Assert - should not create file for empty history
        chart_path = tmp_path / "loss_chart.png"
        assert not chart_path.exists()

    def test_handles_single_point(self, tmp_path: Path) -> None:
        """plot_loss_history works with single data point."""
        # Arrange
        history = {
            "iters": [100],
            "train_loss": [2.0],
            "val_loss": [2.1],
        }

        # Act
        plot_loss_history(history, tmp_path)

        # Assert
        chart_path = tmp_path / "loss_chart.png"
        assert chart_path.exists()

    def test_handles_many_points(self, tmp_path: Path) -> None:
        """plot_loss_history works with many data points."""
        # Arrange
        n_points = 100
        history = {
            "iters": list(range(0, n_points * 100, 100)),
            "train_loss": [2.0 - i * 0.01 for i in range(n_points)],
            "val_loss": [2.1 - i * 0.01 for i in range(n_points)],
        }

        # Act
        plot_loss_history(history, tmp_path)

        # Assert
        chart_path = tmp_path / "loss_chart.png"
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0

    @pytest.mark.parametrize(
        "history",
        [
            {"iters": [0, 100], "train_loss": [3.0, 2.0], "val_loss": [3.1, 2.1]},
            {"iters": [0, 50, 100], "train_loss": [2.5, 2.0, 1.5], "val_loss": [2.6, 2.1, 1.6]},
        ],
        ids=["two_points", "three_points"],
    )
    def test_various_history_sizes(self, history: dict, tmp_path: Path) -> None:
        """plot_loss_history handles various history sizes."""
        plot_loss_history(history, tmp_path)

        chart_path = tmp_path / "loss_chart.png"
        assert chart_path.exists()
