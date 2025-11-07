"""Integration tests for training flow."""

from pathlib import Path

import pytest
import torch

from popgpt.config import load_training_config
from popgpt.training import Trainer


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingFlow:
    """Integration tests for complete training flow."""

    def test_trains_minimal_model_from_scratch(
        self, tmp_data_dir: Path, tmp_path: Path, sample_config_dict: dict
    ) -> None:
        """Trains a minimal model for few iterations."""
        # Arrange
        sample_config_dict["io"]["out_dir"] = str(tmp_path / "out")
        sample_config_dict["io"]["eval_interval"] = 5
        sample_config_dict["io"]["log_interval"] = 1
        sample_config_dict["optimizer"]["max_iters"] = 10
        sample_config_dict["data"]["data_dir"] = str(tmp_data_dir)
        sample_config_dict["model"]["vocab_size"] = 10

        config = load_training_config(overrides=[])
        for section, values in sample_config_dict.items():
            for key, value in values.items():
                setattr(getattr(config, section), key, value)

        # Act
        trainer = Trainer(config)
        trainer.train()

        # Assert
        out_dir = Path(sample_config_dict["io"]["out_dir"])
        assert out_dir.exists()
        assert (out_dir / "ckpt.pt").exists()
        assert trainer.iter_num == 10

    def test_resumes_training_from_checkpoint(
        self, sample_checkpoint: Path, tmp_data_dir: Path, sample_config_dict: dict
    ) -> None:
        """Resumes training from existing checkpoint."""
        # Arrange
        sample_config_dict["io"]["out_dir"] = str(sample_checkpoint)
        sample_config_dict["io"]["init_from"] = "resume"
        sample_config_dict["optimizer"]["max_iters"] = 5
        sample_config_dict["data"]["data_dir"] = str(tmp_data_dir)

        config = load_training_config(overrides=[])
        for section, values in sample_config_dict.items():
            for key, value in values.items():
                setattr(getattr(config, section), key, value)

        # Act
        trainer = Trainer(config)
        initial_iter = trainer.iter_num
        trainer.train()

        # Assert
        assert initial_iter == 100  # From fixture
        assert trainer.iter_num > initial_iter

    def test_loads_weights_only_for_finetuning(
        self, sample_checkpoint: Path, tmp_data_dir: Path, sample_config_dict: dict, tmp_path: Path
    ) -> None:
        """Loads weights only without training state."""
        # Arrange
        sample_config_dict["io"]["out_dir"] = str(tmp_path / "finetune")
        sample_config_dict["io"]["resume_from"] = str(sample_checkpoint)
        sample_config_dict["io"]["weights_only"] = True
        sample_config_dict["optimizer"]["max_iters"] = 5
        sample_config_dict["data"]["data_dir"] = str(tmp_data_dir)
        sample_config_dict["model"]["vocab_size"] = 10  # Match checkpoint vocab_size

        config = load_training_config(overrides=[])
        for section, values in sample_config_dict.items():
            for key, value in values.items():
                setattr(getattr(config, section), key, value)

        # Act
        trainer = Trainer(config)

        # Assert
        assert trainer.iter_num == 0  # Reset, not 100 from checkpoint
        assert trainer.best_val_loss == 1e9  # Reset


@pytest.mark.integration
@pytest.mark.slow
class TestCheckpointLoading:
    """Integration tests for checkpoint loading."""

    def test_checkpoint_contains_all_required_fields(
        self, tmp_data_dir: Path, tmp_path: Path, sample_config_dict: dict
    ) -> None:
        """Saved checkpoint contains all required fields."""
        # Arrange & Act
        sample_config_dict["io"]["out_dir"] = str(tmp_path / "out")
        sample_config_dict["io"]["eval_interval"] = 5
        sample_config_dict["optimizer"]["max_iters"] = 10
        sample_config_dict["data"]["data_dir"] = str(tmp_data_dir)
        sample_config_dict["model"]["vocab_size"] = 10

        config = load_training_config(overrides=[])
        for section, values in sample_config_dict.items():
            for key, value in values.items():
                setattr(getattr(config, section), key, value)

        trainer = Trainer(config)
        trainer.train()

        # Assert
        checkpoint = torch.load(tmp_path / "out" / "ckpt.pt", map_location="cpu")
        assert "model" in checkpoint
        assert "optimizer" in checkpoint
        assert "model_args" in checkpoint
        assert "iter_num" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "config" in checkpoint
