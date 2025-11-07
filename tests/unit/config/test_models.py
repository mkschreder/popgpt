"""Tests for configuration Pydantic models."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from popgpt.config.models import (
    TrainingConfig,
    SamplingConfig,
    IOConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
)


class TestIOConfig:
    """Tests for IO configuration."""

    def test_default_values_are_valid(self) -> None:
        """IOConfig initializes with valid defaults."""
        config = IOConfig()
        assert config.out_dir == Path("out")
        assert config.eval_interval == 2000
        assert config.init_from == "scratch"

    def test_accepts_custom_values(self) -> None:
        """IOConfig accepts custom values."""
        config = IOConfig(
            out_dir=Path("custom-out"),
            eval_interval=500,
            init_from="resume",
        )
        assert config.out_dir == Path("custom-out")
        assert config.eval_interval == 500

    @pytest.mark.parametrize(
        "eval_interval,should_fail",
        [(0, True), (1, False), (1000, False), (-1, True)],
        ids=["zero", "one", "large", "negative"],
    )
    def test_eval_interval_validation(self, eval_interval: int, should_fail: bool) -> None:
        """eval_interval must be >= 1."""
        if should_fail:
            with pytest.raises(ValidationError):
                IOConfig(eval_interval=eval_interval)
        else:
            config = IOConfig(eval_interval=eval_interval)
            assert config.eval_interval == eval_interval


class TestDataConfig:
    """Tests for data configuration."""

    def test_default_values_are_valid(self) -> None:
        """DataConfig initializes with valid defaults."""
        config = DataConfig()
        assert config.dataset == "shakespeare_char"
        assert config.batch_size == 12
        assert config.block_size == 1024
        assert config.mask_before_token is None

    def test_masking_configuration(self) -> None:
        """DataConfig handles masking options correctly."""
        config = DataConfig(
            mask_before_token="=",
            mask_per_line=True,
            align_to_lines=True,
        )
        assert config.mask_before_token == "="
        assert config.mask_per_line is True
        assert config.align_to_lines is True


class TestModelConfig:
    """Tests for model architecture configuration."""

    def test_default_gpt2_like_architecture(self) -> None:
        """ModelConfig defaults to GPT-2 like architecture."""
        config = ModelConfig()
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.d_head == 64  # Calculated from n_embd / n_head

    def test_d_head_calculation_from_n_head(self) -> None:
        """d_head is calculated when n_head is provided."""
        config = ModelConfig(n_layer=6, n_head=8, n_embd=512)
        assert config.n_head == 8
        assert config.d_head == 64
        assert config.n_head * config.d_head == config.n_embd

    def test_n_head_calculation_from_d_head(self) -> None:
        """n_head is calculated when d_head is provided."""
        config = ModelConfig(n_layer=16, d_head=128, n_embd=1024)
        assert config.n_head == 8
        assert config.d_head == 128
        assert config.n_head * config.d_head == config.n_embd

    def test_both_n_head_and_d_head_consistent(self) -> None:
        """Both n_head and d_head can be specified if consistent."""
        config = ModelConfig(n_layer=6, n_head=6, d_head=64, n_embd=384)
        assert config.n_head == 6
        assert config.d_head == 64
        assert config.n_head * config.d_head == config.n_embd

    def test_inconsistent_n_head_and_d_head_fails(self) -> None:
        """Inconsistent n_head and d_head raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(n_layer=6, n_head=8, d_head=64, n_embd=384)

    def test_non_divisible_n_embd_by_d_head_fails(self) -> None:
        """n_embd not divisible by d_head raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(n_layer=6, d_head=100, n_embd=384)

    def test_non_divisible_n_embd_by_n_head_fails(self) -> None:
        """n_embd not divisible by n_head raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(n_layer=6, n_head=7, n_embd=384)

    def test_d_model_instead_of_n_embd(self) -> None:
        """d_model can be used instead of n_embd."""
        config = ModelConfig(n_layer=6, d_model=512, d_head=64)
        assert config.d_model == 512
        assert config.n_embd == 512
        assert config.d_head == 64
        assert config.n_head == 8

    def test_d_model_and_n_embd_same_value(self) -> None:
        """Both d_model and n_embd can be specified if they match."""
        config = ModelConfig(n_layer=6, d_model=384, n_embd=384, d_head=64)
        assert config.d_model == 384
        assert config.n_embd == 384
        assert config.d_head == 64
        assert config.n_head == 6

    def test_d_model_and_n_embd_mismatch_fails(self) -> None:
        """Mismatched d_model and n_embd raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(n_layer=6, d_model=512, n_embd=384, d_head=64)

    @pytest.mark.parametrize(
        "dropout",
        [0.0, 0.1, 0.5, 1.0],
        ids=["zero", "small", "medium", "one"],
    )
    def test_dropout_validation(self, dropout: float) -> None:
        """dropout must be between 0.0 and 1.0."""
        config = ModelConfig(dropout=dropout)
        assert config.dropout == dropout

    def test_dropout_out_of_range_fails(self) -> None:
        """dropout outside [0,1] raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.5)
        with pytest.raises(ValidationError):
            ModelConfig(dropout=-0.1)


class TestOptimizerConfig:
    """Tests for optimizer configuration."""

    def test_default_adamw_settings(self) -> None:
        """OptimizerConfig has reasonable AdamW defaults."""
        config = OptimizerConfig()
        assert config.learning_rate == 6e-4
        assert config.beta1 == 0.9
        assert config.beta2 == 0.95
        assert config.weight_decay == 1e-1

    def test_betas_within_valid_range(self) -> None:
        """beta1 and beta2 must be in [0, 1]."""
        with pytest.raises(ValidationError):
            OptimizerConfig(beta1=1.5)
        with pytest.raises(ValidationError):
            OptimizerConfig(beta2=-0.1)


class TestTrainingConfig:
    """Tests for complete training configuration."""

    def test_initializes_with_defaults(self) -> None:
        """TrainingConfig can be created with all defaults."""
        config = TrainingConfig()
        assert config.io.out_dir == Path("out")
        assert config.data.batch_size == 12
        assert config.model.n_layer == 12

    def test_nested_structure(self, sample_config_dict: dict) -> None:
        """TrainingConfig properly nests sub-configurations."""
        config = TrainingConfig(**sample_config_dict)
        assert config.io.out_dir == Path("out-test")
        assert config.data.batch_size == 4
        assert config.model.n_embd == 32
        assert config.optimizer.learning_rate == 1e-3

    def test_get_data_dir_with_default(self) -> None:
        """get_data_dir returns data/dataset when data_dir not set."""
        config = TrainingConfig(data={"dataset": "shakespeare"})
        assert config.get_data_dir() == Path("data/shakespeare")

    def test_get_data_dir_with_override(self) -> None:
        """get_data_dir returns data_dir when explicitly set."""
        config = TrainingConfig(data={"dataset": "shakespeare", "data_dir": "/custom/path"})
        assert config.get_data_dir() == Path("/custom/path")

    def test_resume_from_checkpoint_path(self) -> None:
        """IOConfig supports resume_from for checkpoint initialization."""
        config = TrainingConfig(io={"resume_from": "/path/to/checkpoint", "weights_only": True})
        assert config.io.resume_from == Path("/path/to/checkpoint")
        assert config.io.weights_only is True


class TestSamplingConfig:
    """Tests for sampling configuration."""

    def test_default_sampling_settings(self) -> None:
        """SamplingConfig has reasonable defaults."""
        config = SamplingConfig()
        assert config.init_from == "resume"
        assert config.num_samples == 10
        assert config.temperature == 0.8
        assert config.top_k == 200

    def test_greedy_decoding_with_zero_temperature(self) -> None:
        """temperature=0.0 enables greedy decoding."""
        config = SamplingConfig(temperature=0.0)
        assert config.temperature == 0.0

    @pytest.mark.parametrize(
        "start",
        ["\n", "Hello world", "FILE:prompt.txt"],
        ids=["newline", "text", "file"],
    )
    def test_start_text_options(self, start: str) -> None:
        """start supports various input formats."""
        config = SamplingConfig(start=start)
        assert config.start == start

    def test_start_accepts_single_string(self) -> None:
        """start accepts a single string (backward compatible)."""
        config = SamplingConfig(start="test_prompt=")
        assert isinstance(config.start, str)
        assert config.start == "test_prompt="

    def test_start_accepts_list_of_strings(self) -> None:
        """start accepts a list of strings."""
        start_list = ["prompt1=", "prompt2=", "prompt3="]
        config = SamplingConfig(start=start_list)
        assert isinstance(config.start, list)
        assert config.start == start_list
        assert len(config.start) == 3

    def test_num_samples_applies_per_start_sequence(self) -> None:
        """num_samples applies to each start sequence in the list."""
        start_list = ["2+2=", "3+3=", "4+4="]
        config = SamplingConfig(start=start_list, num_samples=5)
        # With 3 start sequences and num_samples=5, we expect 5 samples per sequence
        assert len(config.start) == 3
        assert config.num_samples == 5
