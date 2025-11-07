"""Protocol definitions for dependency inversion and interface segregation."""

from typing import Protocol, runtime_checkable
import torch
from pathlib import Path


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loading implementations."""

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data.

        Args:
            split: 'train' or 'val'

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        ...


@runtime_checkable
class CheckpointProtocol(Protocol):
    """Protocol for checkpoint operations."""

    def save(
        self,
        model_state: dict,
        optimizer_state: dict,
        config: dict,
        iter_num: int,
        best_val_loss: float,
    ) -> None:
        """Save checkpoint."""
        ...

    def load(self) -> dict:
        """Load checkpoint.

        Returns:
            Dictionary with checkpoint data
        """
        ...

    def exists(self) -> bool:
        """Check if checkpoint exists."""
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for model evaluation."""

    def estimate_loss(self) -> dict[str, float]:
        """Estimate loss on train and validation sets.

        Returns:
            Dictionary with 'train' and 'val' keys
        """
        ...


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    """Protocol for learning rate scheduling."""

    def get_lr(self, iteration: int) -> float:
        """Get learning rate for given iteration."""
        ...
