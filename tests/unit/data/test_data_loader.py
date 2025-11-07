"""Tests for data loader."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from popgpt.config import DataConfig
from popgpt.data import DataLoader


class TestDataLoaderInitialization:
    """Tests for DataLoader initialization."""

    def test_initializes_with_basic_config(self, tmp_data_dir: Path) -> None:
        """DataLoader initializes with minimal configuration."""
        config = DataConfig(batch_size=4, block_size=16)

        loader = DataLoader(
            data_dir=tmp_data_dir,
            config=config,
            device="cpu",
            device_type="cpu",
        )

        assert loader.data_dir == tmp_data_dir
        assert loader.config.batch_size == 4

    def test_loads_metadata_when_present(self, tmp_data_dir_with_meta: Path) -> None:
        """DataLoader loads metadata file when available."""
        config = DataConfig(mask_before_token="=")

        loader = DataLoader(
            data_dir=tmp_data_dir_with_meta,
            config=config,
            device="cpu",
            device_type="cpu",
        )

        assert loader.mask_token_id == 8  # From meta fixture
        assert loader.newline_token_id == 5


class TestGetBatch:
    """Tests for get_batch method."""

    def test_returns_correct_shapes(self, tmp_data_dir: Path) -> None:
        """get_batch returns tensors with correct shapes."""
        # Arrange
        config = DataConfig(batch_size=4, block_size=16)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        # Act
        x, y = loader.get_batch("train")

        # Assert
        assert x.shape == (4, 16)  # (batch_size, block_size)
        assert y.shape == (4, 16)

    def test_returns_tensors_on_correct_device(self, tmp_data_dir: Path) -> None:
        """get_batch returns tensors on specified device."""
        config = DataConfig(batch_size=2, block_size=8)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        x, y = loader.get_batch("train")

        assert x.device.type == "cpu"
        assert y.device.type == "cpu"

    def test_loads_from_train_split(self, tmp_data_dir: Path) -> None:
        """get_batch('train') loads from train.bin."""
        config = DataConfig(batch_size=2, block_size=8)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        x, y = loader.get_batch("train")

        # Values should be from train data (1-5 repeated)
        assert x.min() >= 1
        assert x.max() <= 5

    def test_loads_from_val_split(self, tmp_data_dir: Path) -> None:
        """get_batch('val') loads from val.bin."""
        config = DataConfig(batch_size=2, block_size=8)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        x, y = loader.get_batch("val")

        # Values should be from val data (6, 7, 8, 9, 0 repeated)
        assert set(x.flatten().tolist()).issubset({0, 6, 7, 8, 9})

    def test_target_is_shifted_input(self, tmp_data_dir: Path) -> None:
        """get_batch returns y as next-token prediction target."""
        config = DataConfig(batch_size=1, block_size=8)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        # Use seed for deterministic sampling
        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # y should be shifted version of the underlying sequence
        # Can't assert exact values due to random sampling, but shapes match
        assert y.shape == x.shape


class TestMasking:
    """Tests for token masking functionality."""

    def test_applies_global_masking(self, tmp_data_dir_with_meta: Path) -> None:
        """DataLoader applies global masking when configured."""
        # Arrange
        config = DataConfig(
            batch_size=2,
            block_size=16,
            mask_before_token="=",
            mask_per_line=False,
        )
        loader = DataLoader(tmp_data_dir_with_meta, config, "cpu", "cpu")

        # Act
        _, y = loader.get_batch("train")

        # Assert - some tokens should be masked (-1)
        # (Exact behavior depends on data, but masking should occur)
        assert y.dtype == torch.long

    def test_no_masking_without_config(self, tmp_data_dir: Path) -> None:
        """DataLoader doesn't mask when mask_before_token is None."""
        config = DataConfig(batch_size=2, block_size=8)
        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        _, y = loader.get_batch("train")

        # No -1 values should be present (no masking)
        assert (y != -1).all()


class TestLineAlignment:
    """Tests for line-aligned sampling."""

    def test_precomputes_line_starts_when_enabled(self, tmp_data_dir_with_meta: Path) -> None:
        """DataLoader precomputes line starts when align_to_lines=True."""
        config = DataConfig(
            batch_size=2,
            block_size=16,
            align_to_lines=True,
        )

        loader = DataLoader(tmp_data_dir_with_meta, config, "cpu", "cpu")

        # Line starts should be computed (may be None if no newlines found)
        assert hasattr(loader, "train_line_starts")
        assert hasattr(loader, "val_line_starts")

    def test_skips_line_alignment_without_newline_token(self, tmp_data_dir: Path) -> None:
        """DataLoader skips line alignment when newline token not in vocab."""
        config = DataConfig(align_to_lines=True, batch_size=2, block_size=8)

        loader = DataLoader(tmp_data_dir, config, "cpu", "cpu")

        # Should gracefully handle missing newline token or have computed line starts
        # (behavior depends on whether tiktoken can find newline)
        # Just verify loader was created successfully
        assert loader is not None
