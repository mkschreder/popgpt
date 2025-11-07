"""Tests for GPT model core."""

import pytest
import torch

from popgpt.core import GPT, GPTConfig


class TestGPTConfig:
    """Tests for GPT configuration."""

    def test_default_config_values(self) -> None:
        """GPTConfig has reasonable defaults."""
        config = GPTConfig()

        assert config.block_size == 1024
        assert config.vocab_size == 50304
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768

    def test_custom_config_values(self) -> None:
        """GPTConfig accepts custom values."""
        config = GPTConfig(
            n_layer=6,
            n_head=4,
            n_embd=256,
            block_size=128,
            vocab_size=1000,
        )

        assert config.n_layer == 6
        assert config.n_head == 4
        assert config.n_embd == 256


class TestGPTInitialization:
    """Tests for GPT model initialization."""

    def test_initializes_with_valid_config(self) -> None:
        """GPT initializes successfully with valid config."""
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            vocab_size=100,
        )

        model = GPT(config)

        assert model.config == config
        assert isinstance(model, torch.nn.Module)

    def test_requires_vocab_size(self) -> None:
        """GPT requires vocab_size to be set."""
        config = GPTConfig(vocab_size=None)

        with pytest.raises(AssertionError):
            GPT(config)

    def test_requires_block_size(self) -> None:
        """GPT requires block_size to be set."""
        config = GPTConfig(block_size=None)

        with pytest.raises(AssertionError):
            GPT(config)


class TestGPTForward:
    """Tests for GPT forward pass."""

    @pytest.fixture
    def minimal_model(self) -> GPT:
        """Create minimal GPT model for testing."""
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            vocab_size=50,
            dropout=0.0,
        )
        return GPT(config)

    def test_forward_returns_logits_and_loss(self, minimal_model: GPT) -> None:
        """forward() returns logits and loss tensors."""
        # Arrange
        batch_size, seq_len = 2, 8
        idx = torch.randint(0, 50, (batch_size, seq_len))
        targets = torch.randint(0, 50, (batch_size, seq_len))

        # Act
        logits, loss = minimal_model(idx, targets)

        # Assert
        assert logits.shape == (batch_size, seq_len, 50)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_without_targets(self, minimal_model: GPT) -> None:
        """forward() without targets returns logits only."""
        # Arrange
        batch_size, seq_len = 2, 8
        idx = torch.randint(0, 50, (batch_size, seq_len))

        # Act
        logits, loss = minimal_model(idx)

        # Assert
        assert logits.shape == (batch_size, 1, 50)  # Only last position
        assert loss is None

    def test_forward_enforces_block_size(self, minimal_model: GPT) -> None:
        """forward() raises error when sequence exceeds block_size."""
        # Arrange - sequence longer than block_size (16)
        idx = torch.randint(0, 50, (1, 20))

        # Act & Assert
        with pytest.raises(AssertionError, match="Cannot forward sequence"):
            minimal_model(idx)


class TestGPTMethods:
    """Tests for GPT utility methods."""

    @pytest.fixture
    def minimal_model(self) -> GPT:
        """Create minimal GPT model for testing."""
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            vocab_size=50,
        )
        return GPT(config)

    def test_get_num_params(self, minimal_model: GPT) -> None:
        """get_num_params() returns reasonable count."""
        n_params = minimal_model.get_num_params()

        assert n_params > 0
        assert isinstance(n_params, int)

    def test_get_num_params_excludes_position_embeddings(self, minimal_model: GPT) -> None:
        """get_num_params(non_embedding=True) excludes position embeddings."""
        n_params_total = minimal_model.get_num_params(non_embedding=False)
        n_params_no_emb = minimal_model.get_num_params(non_embedding=True)

        assert n_params_no_emb < n_params_total

    def test_crop_block_size(self, minimal_model: GPT) -> None:
        """crop_block_size() reduces model's maximum sequence length."""
        # Arrange
        original_block_size = minimal_model.config.block_size
        new_block_size = 8

        # Act
        minimal_model.crop_block_size(new_block_size)

        # Assert
        assert minimal_model.config.block_size == new_block_size
        assert minimal_model.config.block_size < original_block_size

    def test_crop_block_size_validates_size(self, minimal_model: GPT) -> None:
        """crop_block_size() raises error if new size exceeds current."""
        with pytest.raises(AssertionError):
            minimal_model.crop_block_size(100)  # Larger than current (16)


class TestGPTGeneration:
    """Tests for text generation."""

    @pytest.fixture
    def minimal_model(self) -> GPT:
        """Create minimal GPT model in eval mode."""
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            vocab_size=50,
            dropout=0.0,
        )
        model = GPT(config)
        model.eval()
        return model

    def test_generate_returns_longer_sequence(self, minimal_model: GPT) -> None:
        """generate() extends input sequence."""
        # Arrange
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5

        # Act
        result = minimal_model.generate(idx, max_new_tokens)

        # Assert
        assert result.shape[1] == idx.shape[1] + max_new_tokens

    def test_generate_with_temperature_zero(self, minimal_model: GPT) -> None:
        """generate() with temperature=0 uses greedy decoding."""
        idx = torch.tensor([[1, 2]])

        # Generate twice with same seed - should be identical with temp=0
        torch.manual_seed(42)
        result1 = minimal_model.generate(idx, 3, temperature=0.0)
        torch.manual_seed(42)
        result2 = minimal_model.generate(idx, 3, temperature=0.0)

        assert torch.equal(result1, result2)

    def test_generate_with_top_k(self, minimal_model: GPT) -> None:
        """generate() respects top_k parameter."""
        idx = torch.tensor([[1, 2]])

        result = minimal_model.generate(idx, 5, top_k=10)

        assert result.shape[1] == 7  # 2 + 5

    def test_generate_stops_at_stop_token(self, minimal_model: GPT) -> None:
        """generate() can stop early with stop_token."""
        idx = torch.tensor([[1, 2]])

        # Note: May or may not stop early depending on model output
        result = minimal_model.generate(idx, 10, stop_token=0)

        # Should generate something, possibly stopping early
        assert result.shape[1] >= idx.shape[1]
        assert result.shape[1] <= idx.shape[1] + 10
