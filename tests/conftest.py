"""Shared pytest fixtures for PopGPT tests."""

import pickle
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate each test from the real environment."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # Prevent DDP from accidentally initializing
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory with minimal test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create minimal binary files (vocab_size=10, so indices 0-9)
    train_data = np.array([1, 2, 3, 4, 5] * 100, dtype=np.uint16)
    val_data = np.array([6, 7, 8, 9, 0] * 100, dtype=np.uint16)  # Changed 10 to 0

    train_data.tofile(data_dir / "train.bin")
    val_data.tofile(data_dir / "val.bin")

    return data_dir


@pytest.fixture
def tmp_data_dir_with_meta(tmp_data_dir: Path) -> Path:
    """Create temporary data directory with metadata."""
    meta = {
        "vocab_size": 10,
        "stoi": {str(i): i for i in range(10)},
        "itos": {i: str(i) for i in range(10)},
    }
    meta["stoi"]["\n"] = 5
    meta["stoi"]["="] = 8

    with open(tmp_data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return tmp_data_dir


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "io": {
            "out_dir": "out-test",
            "eval_interval": 100,
            "log_interval": 10,
            "eval_iters": 10,
            "eval_only": False,
            "always_save_checkpoint": True,
            "init_from": "scratch",
        },
        "data": {
            "dataset": "test_data",
            "batch_size": 4,
            "block_size": 16,
        },
        "model": {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "dropout": 0.0,
            "bias": False,
        },
        "optimizer": {
            "learning_rate": 1e-3,
            "max_iters": 100,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "gradient_accumulation_steps": 1,
        },
        "lr_schedule": {
            "decay_lr": True,
            "warmup_iters": 10,
            "lr_decay_iters": 100,
            "min_lr": 1e-4,
        },
        "system": {
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "seed": 1337,
        },
    }


@pytest.fixture
def sample_checkpoint(tmp_path: Path) -> Path:
    """Create a sample checkpoint file."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    # Create minimal GPT config
    model_args = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "block_size": 16,
        "bias": False,
        "vocab_size": 10,
        "dropout": 0.0,
    }

    # Create dummy model state
    from popgpt.core import GPT, GPTConfig

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    checkpoint = {
        "model": model.state_dict(),
        "model_args": model_args,
        "iter_num": 100,
        "best_val_loss": 1.5,
        "config": {"data": {"dataset": "test"}},
        "optimizer": {},  # Minimal optimizer state
    }

    checkpoint_path = checkpoint_dir / "ckpt.pt"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_dir


@pytest.fixture
def minimal_gpt_model() -> torch.nn.Module:
    """Create a minimal GPT model for testing."""
    from popgpt.core import GPT, GPTConfig

    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
        vocab_size=10,
        dropout=0.0,
        bias=False,
    )
    return GPT(config)


@pytest.fixture
def seed_random() -> Generator[None, None, None]:
    """Seed random generators for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset after test if needed
