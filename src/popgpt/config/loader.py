"""Configuration loading and merging utilities."""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml

from .models import TrainingConfig, SamplingConfig


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        return {}

    return config_dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def parse_cli_override(override: str) -> tuple[str, Any]:
    """Parse CLI override in format 'key=value' or 'section.key=value'.

    Returns:
        Tuple of (dotted_key, value)
    """
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}. Expected 'key=value'")

    key, value_str = override.split("=", 1)

    # Try to parse as Python literal
    try:
        import ast

        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If that fails, treat as string
        value = value_str

    return key, value


def apply_cli_overrides(config_dict: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides to configuration dictionary.

    Supports nested keys like 'model.n_layer=24' or flat keys like 'learning_rate=1e-3'.
    """
    for override in overrides:
        key, value = parse_cli_override(override)

        # Handle nested keys
        parts = key.split(".")
        current = config_dict

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return config_dict


def load_training_config(
    config_path: Optional[Path] = None,
    overrides: Optional[list[str]] = None,
) -> TrainingConfig:
    """Load training configuration with optional overrides.

    Args:
        config_path: Path to YAML config file (optional)
        overrides: List of CLI overrides in format 'key=value' (optional)

    Returns:
        Validated TrainingConfig instance
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {}

    # Load from YAML if provided
    if config_path:
        yaml_config = load_yaml_config(config_path)
        config_dict = deep_merge(config_dict, yaml_config)

    # Apply CLI overrides
    if overrides:
        config_dict = apply_cli_overrides(config_dict, overrides)

    # Validate and return
    return TrainingConfig(**config_dict)


def load_sampling_config(
    config_path: Optional[Path] = None,
    overrides: Optional[list[str]] = None,
) -> SamplingConfig:
    """Load sampling configuration with optional overrides.

    Args:
        config_path: Path to YAML config file (optional)
        overrides: List of CLI overrides in format 'key=value' (optional)

    Returns:
        Validated SamplingConfig instance
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {}

    # Load from YAML if provided
    if config_path:
        yaml_config = load_yaml_config(config_path)
        config_dict = deep_merge(config_dict, yaml_config)

    # Apply CLI overrides
    if overrides:
        config_dict = apply_cli_overrides(config_dict, overrides)

    # Validate and return
    return SamplingConfig(**config_dict)
