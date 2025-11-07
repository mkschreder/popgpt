"""Tests for device utilities."""

import os
from contextlib import nullcontext

import pytest
import torch

from popgpt.utils.device import setup_device, setup_ddp, cleanup_ddp


class TestSetupDevice:
    """Tests for device setup."""

    def test_returns_cpu_device(self) -> None:
        """setup_device configures CPU device correctly."""
        # Act
        device, device_type, ctx, ptdtype = setup_device("cpu", "float32")

        # Assert
        assert device == "cpu"
        assert device_type == "cpu"
        assert ptdtype == torch.float32
        assert isinstance(ctx, type(nullcontext()))

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("float32", torch.float32),
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
        ],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_maps_dtype_strings_correctly(self, dtype: str, expected: torch.dtype) -> None:
        """setup_device maps dtype strings to torch dtypes."""
        _, _, _, ptdtype = setup_device("cpu", dtype)
        assert ptdtype == expected

    def test_cuda_device_when_available(self) -> None:
        """setup_device handles CUDA device."""
        device, device_type, ctx, ptdtype = setup_device("cuda", "float16")

        assert "cuda" in device
        assert device_type == "cuda"
        # ctx will be autocast context (not nullcontext)

    def test_extracts_device_type_from_device_string(self) -> None:
        """setup_device extracts device type from full device string."""
        _, device_type, _, _ = setup_device("cuda:0", "float32")
        assert device_type == "cuda"

        _, device_type, _, _ = setup_device("cpu", "float32")
        assert device_type == "cpu"


class TestSetupDDP:
    """Tests for distributed data parallel setup."""

    def test_returns_non_ddp_when_rank_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_ddp returns non-DDP configuration when RANK not set."""
        # Arrange - ensure RANK is not set
        monkeypatch.delenv("RANK", raising=False)

        # Act
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset = setup_ddp()

        # Assert
        assert ddp is False
        assert ddp_rank == 0
        assert ddp_local_rank == 0
        assert ddp_world_size == 1
        assert master_process is True
        assert seed_offset == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for DDP tests")
    def test_detects_ddp_when_rank_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_ddp detects DDP mode when RANK is set."""
        # Arrange
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "4")

        # Mock init_process_group to avoid actual initialization
        def mock_init(*args, **kwargs):
            pass

        monkeypatch.setattr("torch.distributed.init_process_group", mock_init)

        # Act
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset = setup_ddp()

        # Assert
        assert ddp is True
        assert ddp_rank == 1
        assert ddp_local_rank == 1
        assert ddp_world_size == 4
        assert master_process is False  # Rank 1 is not master
        assert seed_offset == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for DDP tests")
    def test_identifies_master_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_ddp correctly identifies master process."""
        # Arrange - rank 0 is master
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        def mock_init(*args, **kwargs):
            pass

        monkeypatch.setattr("torch.distributed.init_process_group", mock_init)

        # Act
        _, _, _, _, master_process, _ = setup_ddp()

        # Assert
        assert master_process is True


class TestCleanupDDP:
    """Tests for DDP cleanup."""

    def test_no_op_when_ddp_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup_ddp does nothing when DDP is not active."""
        # Arrange
        cleanup_called = False

        def mock_destroy():
            nonlocal cleanup_called
            cleanup_called = True

        monkeypatch.setattr("torch.distributed.destroy_process_group", mock_destroy)

        # Act
        cleanup_ddp(ddp=False)

        # Assert
        assert cleanup_called is False

    def test_calls_destroy_when_ddp_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup_ddp calls destroy_process_group when DDP is active."""
        # Arrange
        cleanup_called = False

        def mock_destroy():
            nonlocal cleanup_called
            cleanup_called = True

        monkeypatch.setattr("torch.distributed.destroy_process_group", mock_destroy)

        # Act
        cleanup_ddp(ddp=True)

        # Assert
        assert cleanup_called is True
