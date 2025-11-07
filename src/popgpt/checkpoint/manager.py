"""Checkpoint management for model persistence."""

import pickle
from pathlib import Path
from typing import Any, Union

import torch


class CheckpointManager:
    """Manages model checkpoints and loss history.

    Implements the CheckpointProtocol.
    """

    def __init__(self, out_dir: Union[Path, str]) -> None:
        """Initialize checkpoint manager.

        Args:
            out_dir: Directory for saving checkpoints (Path or str)
        """
        self.out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        self.checkpoint_path = self.out_dir / "ckpt.pt"
        self.history_path = self.out_dir / "loss_history.pkl"

        # Ensure directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        config: dict[str, Any],
        iter_num: int,
        best_val_loss: float,
    ) -> None:
        """Save checkpoint.

        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            config: Training configuration dict
            iter_num: Current iteration number
            best_val_loss: Best validation loss so far
        """
        checkpoint = {
            "model": model_state,
            "optimizer": optimizer_state,
            "model_args": config.get("model_args", {}),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        print(f"Saving checkpoint to {self.out_dir}")
        torch.save(checkpoint, self.checkpoint_path)

    def load(self) -> dict[str, Any]:
        """Load checkpoint.

        Returns:
            Dictionary with checkpoint data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        return checkpoint

    def exists(self) -> bool:
        """Check if checkpoint exists.

        Returns:
            True if checkpoint file exists
        """
        return self.checkpoint_path.exists()

    def save_loss_history(self, history: dict[str, list]) -> None:
        """Save loss history to pickle file.

        Args:
            history: Dictionary with 'iters', 'train_loss', 'val_loss' keys
        """
        with open(self.history_path, "wb") as f:
            pickle.dump(history, f)

    def load_loss_history(self) -> dict[str, list]:
        """Load loss history from pickle file.

        Returns:
            Dictionary with loss history, or empty dict if not found
        """
        if self.history_path.exists():
            with open(self.history_path, "rb") as f:
                return pickle.load(f)
        return {"iters": [], "train_loss": [], "val_loss": []}
