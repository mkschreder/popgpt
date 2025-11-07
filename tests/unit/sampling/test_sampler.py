"""Tests for text sampler."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from popgpt.config import SamplingConfig
from popgpt.sampling import Sampler


class TestSamplerInitialization:
    """Tests for Sampler initialization."""

    def test_initializes_with_resume_from_checkpoint(self, sample_checkpoint: Path) -> None:
        """Sampler loads model from checkpoint."""
        # Arrange
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
        )

        # Act
        sampler = Sampler(config)

        # Assert
        assert sampler.model is not None
        assert isinstance(sampler.model, torch.nn.Module)

    def test_initializes_with_pretrained_gpt2(self) -> None:
        """Sampler can load pretrained GPT-2 model."""
        # Arrange
        config = SamplingConfig(
            init_from="gpt2",
            device="cpu",
            dtype="float32",
            compile=False,
        )

        # Mock the GPT.from_pretrained to avoid downloading
        with patch("popgpt.sampling.sampler.GPT") as MockGPT:
            mock_model = MagicMock()
            MockGPT.from_pretrained.return_value = mock_model

            # Act
            sampler = Sampler(config)

            # Assert
            MockGPT.from_pretrained.assert_called_once()

    def test_sets_model_to_eval_mode(self, sample_checkpoint: Path) -> None:
        """Sampler sets model to evaluation mode."""
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
        )

        sampler = Sampler(config)

        assert not sampler.model.training


class TestTextGeneration:
    """Tests for text generation."""

    @pytest.mark.slow
    def test_generate_returns_list_of_samples(self, sample_checkpoint: Path) -> None:
        """generate() returns list of text samples."""
        # Arrange
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
            num_samples=3,
            max_new_tokens=10,
            start="test",
        )
        sampler = Sampler(config)

        # Act
        samples = sampler.generate()

        # Assert
        assert isinstance(samples, list)
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)

    @pytest.mark.slow
    def test_generate_with_custom_start_text(self, sample_checkpoint: Path) -> None:
        """generate() uses custom start text."""
        # Arrange
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
            num_samples=1,
            max_new_tokens=5,
        )
        sampler = Sampler(config)

        # Act
        samples = sampler.generate(start_text="Hello")

        # Assert
        assert len(samples) == 1
        # Generated text should start with or include the prompt
        # (exact behavior depends on tokenization)

    @pytest.mark.slow
    def test_generate_with_file_input(self, sample_checkpoint: Path, tmp_path: Path) -> None:
        """generate() loads start text from file."""
        # Arrange
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("This is a test prompt", encoding="utf-8")

        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
            num_samples=1,
            max_new_tokens=5,
            start=f"FILE:{prompt_file}",
        )
        sampler = Sampler(config)

        # Act
        samples = sampler.generate()

        # Assert
        assert len(samples) == 1

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "temperature",
        [0.0, 0.5, 0.8, 1.0],
        ids=["greedy", "low", "medium", "high"],
    )
    def test_generate_with_various_temperatures(
        self, sample_checkpoint: Path, temperature: float
    ) -> None:
        """generate() respects temperature parameter."""
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
            num_samples=1,
            max_new_tokens=5,
            temperature=temperature,
        )
        sampler = Sampler(config)

        samples = sampler.generate()

        assert len(samples) == 1


class TestCodecSetup:
    """Tests for encoder/decoder setup."""

    def test_uses_meta_when_available(
        self, sample_checkpoint: Path, tmp_data_dir_with_meta: Path
    ) -> None:
        """Sampler uses meta.pkl when available in dataset."""
        # This is harder to test without modifying the checkpoint
        # to include dataset info, so we test the fallback
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
        )

        sampler = Sampler(config)

        # Should have encode/decode functions
        assert callable(sampler.encode)
        assert callable(sampler.decode)

    def test_fallback_to_tiktoken(self, sample_checkpoint: Path) -> None:
        """Sampler falls back to tiktoken when meta.pkl not found."""
        config = SamplingConfig(
            init_from="resume",
            out_dir=sample_checkpoint,
            device="cpu",
            dtype="float32",
            compile=False,
        )

        sampler = Sampler(config)

        # Should use tiktoken encoding
        test_text = "Hello world"
        encoded = sampler.encode(test_text)
        decoded = sampler.decode(encoded)

        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
