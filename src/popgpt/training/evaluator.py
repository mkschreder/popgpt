"""Model evaluation utilities."""

from typing import Any

import torch

from ..core.protocols import DataLoaderProtocol


class Evaluator:
    """Evaluates model performance on train and validation sets.

    Implements the EvaluatorProtocol.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoaderProtocol,
        eval_iters: int,
        ctx: Any,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: The model to evaluate
            data_loader: Data loader for getting batches
            eval_iters: Number of iterations for evaluation
            ctx: Autocast context for mixed precision
        """
        self.model = model
        self.data_loader = data_loader
        self.eval_iters = eval_iters
        self.ctx = ctx

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate loss on train and validation sets.

        Returns:
            Dictionary with 'train' and 'val' loss values
        """
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.data_loader.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        self.model.train()
        return out
