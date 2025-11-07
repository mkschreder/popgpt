"""Tests for configuration loading and merging."""

import pytest
from pathlib import Path
import yaml

from popgpt.config.loader import (
    load_yaml_config,
    deep_merge,
    parse_cli_override,
    apply_cli_overrides,
    load_training_config,
)


class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_loads_valid_yaml_file(self, tmp_path: Path) -> None:
        """load_yaml_config reads and parses YAML correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  n_layer: 6\n  n_head: 4\n")
        
        result = load_yaml_config(config_file)
        assert result["model"]["n_layer"] == 6
        assert result["model"]["n_head"] == 4

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """load_yaml_config returns empty dict for empty file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        
        result = load_yaml_config(config_file)
        assert result == {}

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """load_yaml_config raises FileNotFoundError for missing file."""
        nonexistent = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_yaml_config(nonexistent)


class TestDeepMerge:
    """Tests for deep dictionary merging."""

    def test_merges_flat_dicts(self) -> None:
        """deep_merge combines flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merges_nested_dicts(self) -> None:
        """deep_merge recursively merges nested dictionaries."""
        base = {"model": {"n_layer": 12, "n_head": 12}, "data": {"batch_size": 32}}
        override = {"model": {"n_layer": 6}, "optimizer": {"learning_rate": 1e-3}}
        
        result = deep_merge(base, override)
        assert result["model"]["n_layer"] == 6  # Overridden
        assert result["model"]["n_head"] == 12  # Preserved
        assert result["data"]["batch_size"] == 32  # Preserved
        assert result["optimizer"]["learning_rate"] == 1e-3  # Added

    def test_preserves_original_dicts(self) -> None:
        """deep_merge doesn't mutate input dictionaries."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        
        result = deep_merge(base, override)
        assert "c" not in base["a"]
        assert "b" in result["a"]


class TestParseCliOverride:
    """Tests for parsing CLI override strings."""

    @pytest.mark.parametrize(
        "override,expected_key,expected_value,expected_type",
        [
            ("model.n_layer=6", "model.n_layer", 6, int),
            ("optimizer.learning_rate=1e-3", "optimizer.learning_rate", 1e-3, float),
            ("system.compile=False", "system.compile", False, bool),
            ("data.dataset=shakespeare", "data.dataset", "shakespeare", str),
            ("io.out_dir=my-output", "io.out_dir", "my-output", str),
        ],
        ids=["int", "float", "bool", "string", "path"],
    )
    def test_parses_various_types(
        self, override: str, expected_key: str, expected_value: any, expected_type: type
    ) -> None:
        """parse_cli_override handles different value types."""
        key, value = parse_cli_override(override)
        assert key == expected_key
        assert value == expected_value
        assert isinstance(value, expected_type)

    def test_raises_on_missing_equals(self) -> None:
        """parse_cli_override raises ValueError without '='."""
        with pytest.raises(ValueError, match="Invalid override format"):
            parse_cli_override("model.n_layer")

    def test_handles_string_with_equals(self) -> None:
        """parse_cli_override preserves '=' in value."""
        key, value = parse_cli_override("data.mask_before_token==")
        assert key == "data.mask_before_token"
        assert value == "="


class TestApplyCliOverrides:
    """Tests for applying CLI overrides to config dict."""

    def test_applies_flat_override(self) -> None:
        """apply_cli_overrides modifies top-level keys."""
        config = {"learning_rate": 1e-3}
        overrides = ["learning_rate=5e-4"]
        
        result = apply_cli_overrides(config, overrides)
        assert result["learning_rate"] == 5e-4

    def test_applies_nested_override(self) -> None:
        """apply_cli_overrides handles nested keys."""
        config = {"model": {"n_layer": 12}}
        overrides = ["model.n_layer=6"]
        
        result = apply_cli_overrides(config, overrides)
        assert result["model"]["n_layer"] == 6

    def test_creates_missing_nested_keys(self) -> None:
        """apply_cli_overrides creates intermediate dicts."""
        config = {}
        overrides = ["model.n_layer=6", "model.n_head=4"]
        
        result = apply_cli_overrides(config, overrides)
        assert result["model"]["n_layer"] == 6
        assert result["model"]["n_head"] == 4

    def test_applies_multiple_overrides(self) -> None:
        """apply_cli_overrides handles multiple overrides."""
        config = {"model": {"n_layer": 12}}
        overrides = [
            "model.n_layer=6",
            "model.n_head=4",
            "optimizer.learning_rate=1e-3",
        ]
        
        result = apply_cli_overrides(config, overrides)
        assert result["model"]["n_layer"] == 6
        assert result["model"]["n_head"] == 4
        assert result["optimizer"]["learning_rate"] == 1e-3


class TestLoadTrainingConfig:
    """Tests for complete training config loading."""

    def test_loads_with_defaults_only(self) -> None:
        """load_training_config works without config file."""
        config = load_training_config()
        assert config.io.out_dir == Path("out")
        assert config.model.n_layer == 12

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """load_training_config loads from YAML file."""
        config_file = tmp_path / "train.yaml"
        config_file.write_text(
            "model:\n  n_layer: 6\ndata:\n  batch_size: 64\n"
        )
        
        config = load_training_config(config_path=config_file)
        assert config.model.n_layer == 6
        assert config.data.batch_size == 64

    def test_applies_overrides_to_yaml(self, tmp_path: Path) -> None:
        """load_training_config applies CLI overrides over YAML."""
        config_file = tmp_path / "train.yaml"
        config_file.write_text("model:\n  n_layer: 6\n")
        
        config = load_training_config(
            config_path=config_file,
            overrides=["model.n_layer=4", "optimizer.learning_rate=5e-4"],
        )
        assert config.model.n_layer == 4  # Overridden
        assert config.optimizer.learning_rate == 5e-4  # Added

    def test_validates_final_config(self, tmp_path: Path) -> None:
        """load_training_config validates Pydantic constraints."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("model:\n  dropout: 1.5\n")  # Invalid
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_training_config(config_path=config_file)

