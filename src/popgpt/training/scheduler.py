"""Learning rate scheduling."""

import math


class CosineDecayWithWarmup:
    """Cosine learning rate decay with linear warmup.

    Implements the LRSchedulerProtocol.
    """

    def __init__(
        self,
        learning_rate: float,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        decay_enabled: bool = True,
    ) -> None:
        """Initialize learning rate scheduler.

        Args:
            learning_rate: Maximum learning rate
            warmup_iters: Number of warmup iterations
            lr_decay_iters: Number of decay iterations
            min_lr: Minimum learning rate
            decay_enabled: Whether to enable decay
        """
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.decay_enabled = decay_enabled

    def get_lr(self, iteration: int) -> float:
        """Get learning rate for given iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Learning rate for this iteration
        """
        if not self.decay_enabled:
            return self.learning_rate

        # Linear warmup
        if iteration < self.warmup_iters:
            return self.learning_rate * (iteration + 1) / (self.warmup_iters + 1)

        # Return min LR after decay period
        if iteration > self.lr_decay_iters:
            return self.min_lr

        # Cosine decay
        decay_ratio = (iteration - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
