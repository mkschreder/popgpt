"""Tests for checkpoint manager."""

import pickle
from pathlib import Path

import pytest
import torch

from popgpt.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """CheckpointManager creates out_dir on initialization."""
        out_dir = tmp_path / "checkpoints"
        assert not out_dir.exists()
        
        manager = CheckpointManager(out_dir)
        
        assert out_dir.exists()
        assert manager.out_dir == out_dir

    def test_checkpoint_path_is_correct(self, tmp_path: Path) -> None:
        """CheckpointManager sets correct checkpoint path."""
        out_dir = tmp_path / "out"
        manager = CheckpointManager(out_dir)
        
        assert manager.checkpoint_path == out_dir / "ckpt.pt"

    def test_exists_returns_false_for_new_manager(self, tmp_path: Path) -> None:
        """exists() returns False when no checkpoint exists."""
        manager = CheckpointManager(tmp_path / "new")
        assert manager.exists() is False

    def test_exists_returns_true_after_save(self, tmp_path: Path) -> None:
        """exists() returns True after checkpoint is saved."""
        # Arrange
        manager = CheckpointManager(tmp_path / "out")
        model_state = {"layer.weight": torch.randn(2, 2)}
        
        # Act
        manager.save(
            model_state=model_state,
            optimizer_state={},
            config={"model_args": {}},
            iter_num=100,
            best_val_loss=1.5,
        )
        
        # Assert
        assert manager.exists() is True

    def test_save_creates_checkpoint_file(self, tmp_path: Path) -> None:
        """save() creates checkpoint file with correct data."""
        # Arrange
        manager = CheckpointManager(tmp_path / "out")
        model_state = {"weight": torch.tensor([1.0, 2.0])}
        optimizer_state = {"lr": 0.001}
        config = {"model_args": {"n_layer": 6}}
        
        # Act
        manager.save(
            model_state=model_state,
            optimizer_state=optimizer_state,
            config=config,
            iter_num=500,
            best_val_loss=2.3,
        )
        
        # Assert
        checkpoint = torch.load(manager.checkpoint_path, map_location="cpu")
        assert "model" in checkpoint
        assert "optimizer" in checkpoint
        assert checkpoint["iter_num"] == 500
        assert checkpoint["best_val_loss"] == 2.3
        assert torch.equal(checkpoint["model"]["weight"], torch.tensor([1.0, 2.0]))

    def test_load_returns_saved_checkpoint(self, tmp_path: Path) -> None:
        """load() returns previously saved checkpoint."""
        # Arrange
        manager = CheckpointManager(tmp_path / "out")
        original_data = {
            "model": {"param": torch.randn(3, 3)},
            "optimizer": {"lr": 0.001},
            "model_args": {"n_layer": 4},
            "iter_num": 200,
            "best_val_loss": 1.8,
            "config": {},
        }
        torch.save(original_data, manager.checkpoint_path)
        
        # Act
        loaded = manager.load()
        
        # Assert
        assert loaded["iter_num"] == 200
        assert loaded["best_val_loss"] == 1.8
        assert "model" in loaded
        assert "optimizer" in loaded

    def test_load_raises_when_checkpoint_missing(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError when checkpoint doesn't exist."""
        manager = CheckpointManager(tmp_path / "empty")
        
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            manager.load()

    def test_save_loss_history_creates_pickle(self, tmp_path: Path) -> None:
        """save_loss_history() creates pickle file with history."""
        # Arrange
        manager = CheckpointManager(tmp_path / "out")
        history = {
            "iters": [0, 100, 200],
            "train_loss": [2.5, 2.0, 1.5],
            "val_loss": [2.6, 2.1, 1.6],
        }
        
        # Act
        manager.save_loss_history(history)
        
        # Assert
        assert manager.history_path.exists()
        with open(manager.history_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded["iters"] == [0, 100, 200]
        assert loaded["train_loss"] == [2.5, 2.0, 1.5]

    def test_load_loss_history_returns_saved_data(self, tmp_path: Path) -> None:
        """load_loss_history() returns previously saved history."""
        # Arrange
        manager = CheckpointManager(tmp_path / "out")
        history = {"iters": [100], "train_loss": [1.5], "val_loss": [1.6]}
        manager.save_loss_history(history)
        
        # Act
        loaded = manager.load_loss_history()
        
        # Assert
        assert loaded == history

    def test_load_loss_history_returns_empty_when_missing(self, tmp_path: Path) -> None:
        """load_loss_history() returns empty dict when file doesn't exist."""
        manager = CheckpointManager(tmp_path / "out")
        
        result = manager.load_loss_history()
        
        assert result == {"iters": [], "train_loss": [], "val_loss": []}

