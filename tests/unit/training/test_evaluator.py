"""Tests for model evaluator."""

from contextlib import nullcontext

import pytest
import torch

from popgpt.training import Evaluator


class FakeDataLoader:
    """Fake data loader for testing."""

    def __init__(self, vocab_size: int = 10) -> None:
        self.vocab_size = vocab_size

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake batch data."""
        batch_size, block_size = 2, 8
        x = torch.randint(0, self.vocab_size, (batch_size, block_size))
        y = torch.randint(0, self.vocab_size, (batch_size, block_size))
        return x, y


class FakeModel(torch.nn.Module):
    """Fake model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.training = True

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake logits and loss."""
        self.call_count += 1
        batch_size, seq_len = x.shape
        vocab_size = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        loss = torch.tensor(1.5 + torch.rand(1).item() * 0.1)  # ~1.5-1.6
        return logits, loss

    def eval(self) -> "FakeModel":
        """Set to eval mode."""
        self.training = False
        return self

    def train(self) -> "FakeModel":
        """Set to train mode."""
        self.training = True
        return self


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_initializes_with_dependencies(self) -> None:
        """Evaluator initializes with required dependencies."""
        # Arrange
        model = FakeModel()
        data_loader = FakeDataLoader()
        ctx = nullcontext()

        # Act
        evaluator = Evaluator(model, data_loader, eval_iters=10, ctx=ctx)

        # Assert
        assert evaluator.model is model
        assert evaluator.data_loader is data_loader
        assert evaluator.eval_iters == 10

    def test_estimate_loss_returns_both_splits(self) -> None:
        """estimate_loss returns losses for train and val splits."""
        # Arrange
        model = FakeModel()
        data_loader = FakeDataLoader()
        evaluator = Evaluator(model, data_loader, eval_iters=5, ctx=nullcontext())

        # Act
        result = evaluator.estimate_loss()

        # Assert
        assert "train" in result
        assert "val" in result
        assert isinstance(result["train"], float)
        assert isinstance(result["val"], float)

    def test_estimate_loss_sets_model_to_eval_mode(self) -> None:
        """estimate_loss sets model to eval mode during evaluation."""
        # Arrange
        model = FakeModel()
        model.train()  # Start in train mode
        data_loader = FakeDataLoader()
        evaluator = Evaluator(model, data_loader, eval_iters=3, ctx=nullcontext())

        # Act
        evaluator.estimate_loss()

        # Assert - model should be back in train mode after
        assert model.training is True

    def test_estimate_loss_averages_multiple_iterations(self) -> None:
        """estimate_loss averages loss over multiple iterations."""
        # Arrange
        model = FakeModel()
        data_loader = FakeDataLoader()
        eval_iters = 10
        evaluator = Evaluator(model, data_loader, eval_iters=eval_iters, ctx=nullcontext())

        # Act
        result = evaluator.estimate_loss()

        # Assert - should call model eval_iters times per split
        expected_calls = eval_iters * 2  # train + val
        assert model.call_count == expected_calls

    def test_estimate_loss_with_different_eval_iters(self) -> None:
        """estimate_loss respects eval_iters parameter."""
        # Arrange
        model = FakeModel()
        data_loader = FakeDataLoader()

        # Act & Assert - different eval_iters values
        for eval_iters in [1, 5, 20]:
            model.call_count = 0
            evaluator = Evaluator(model, data_loader, eval_iters=eval_iters, ctx=nullcontext())
            evaluator.estimate_loss()
            assert model.call_count == eval_iters * 2

    def test_estimate_loss_returns_reasonable_values(self) -> None:
        """estimate_loss returns losses in reasonable range."""
        # Arrange
        model = FakeModel()
        data_loader = FakeDataLoader()
        evaluator = Evaluator(model, data_loader, eval_iters=10, ctx=nullcontext())

        # Act
        result = evaluator.estimate_loss()

        # Assert - losses should be positive and in expected range
        assert result["train"] > 0
        assert result["val"] > 0
        assert 1.0 < result["train"] < 2.0  # Based on FakeModel's range
        assert 1.0 < result["val"] < 2.0

    @pytest.mark.parametrize("eval_iters", [1, 3, 10, 50], ids=["one", "few", "normal", "many"])
    def test_estimate_loss_with_various_iterations(self, eval_iters: int) -> None:
        """estimate_loss works correctly with different iteration counts."""
        model = FakeModel()
        data_loader = FakeDataLoader()
        evaluator = Evaluator(model, data_loader, eval_iters=eval_iters, ctx=nullcontext())

        result = evaluator.estimate_loss()

        assert "train" in result
        assert "val" in result
        assert result["train"] > 0
        assert result["val"] > 0
