"""Command-line interface for PopGPT."""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Optional

from .config import load_training_config, load_sampling_config
from .training import Trainer
from .sampling import Sampler


def train_command(args: argparse.Namespace) -> None:
    """Execute training command.

    Args:
        args: Parsed command-line arguments
    """
    # Load configuration
    config_path = Path(args.config) if args.config else None
    overrides = args.override or []

    config = load_training_config(config_path=config_path, overrides=overrides)

    # Override out_dir if specified
    if args.out_dir:
        config.io.out_dir = Path(args.out_dir)

    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


def sample_command(args: argparse.Namespace) -> None:
    """Execute sampling command.

    Args:
        args: Parsed command-line arguments
    """
    # Load configuration
    config_path = Path(args.config) if args.config else None
    overrides = args.override or []

    config = load_sampling_config(config_path=config_path, overrides=overrides)

    # Override out_dir if specified
    if args.out_dir:
        config.out_dir = Path(args.out_dir)

    # Override start text if specified
    if args.start:
        config.start = args.start

    # Create and run sampler
    sampler = Sampler(config)
    sampler.generate_and_print()


def eval_command(args: argparse.Namespace) -> None:
    """Execute evaluation command.

    Args:
        args: Parsed command-line arguments
    """
    # Load configuration with eval_only=True
    config_path = Path(args.config) if args.config else None
    overrides = args.override or []
    overrides.append("io.eval_only=True")

    config = load_training_config(config_path=config_path, overrides=overrides)

    # Override out_dir if specified
    if args.out_dir:
        config.io.out_dir = Path(args.out_dir)

    # Create and run trainer in eval mode
    trainer = Trainer(config)
    trainer.train()  # Will only evaluate due to eval_only=True


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="popgpt",
        description="A production-ready implementation of PopGPT for training and sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with YAML config
  popgpt train --config configs/train_shakespeare.yaml

  # Train with config and overrides
  popgpt train --config configs/train_shakespeare.yaml --override model.n_layer=6 --override optimizer.learning_rate=1e-3

  # Train with overrides only (uses defaults)
  popgpt train --override data.dataset=shakespeare_char --override optimizer.max_iters=5000

  # Sample from trained model
  popgpt sample --out-dir out-shakespeare-char --start "To be or not to be"

  # Sample with config
  popgpt sample --config configs/sample_shakespeare.yaml

  # Evaluate model
  popgpt eval --config configs/train_shakespeare.yaml --out-dir out-shakespeare-char
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    train_parser.add_argument(
        "-o",
        "--override",
        action="append",
        help="Override config value (e.g., model.n_layer=6)",
    )
    train_parser.add_argument("--out-dir", type=str, help="Output directory for checkpoints")
    train_parser.set_defaults(func=train_command)

    # Sample subcommand
    sample_parser = subparsers.add_parser("sample", help="Generate text samples")
    sample_parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    sample_parser.add_argument(
        "-o",
        "--override",
        action="append",
        help="Override config value (e.g., temperature=0.9)",
    )
    sample_parser.add_argument("--out-dir", type=str, help="Directory with model checkpoint")
    sample_parser.add_argument("--start", type=str, help="Start text for generation")
    sample_parser.set_defaults(func=sample_command)

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    eval_parser.add_argument(
        "-o",
        "--override",
        action="append",
        help="Override config value (e.g., data.batch_size=32)",
    )
    eval_parser.add_argument("--out-dir", type=str, help="Directory with model checkpoint")
    eval_parser.set_defaults(func=eval_command)

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
