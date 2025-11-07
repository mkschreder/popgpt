"""Common utilities for data generation scripts."""

import pickle
from pathlib import Path
from typing import Protocol, Callable, Optional
from abc import ABC, abstractmethod

import numpy as np
import tiktoken


class TokenCounter:
    """Utility for counting tokens in text."""

    def __init__(self, encoding: str = "gpt2") -> None:
        """Initialize token counter.

        Args:
            encoding: Tiktoken encoding name
        """
        self.enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        return len(self.enc.encode(text))


class DatasetGenerator(ABC):
    """Abstract base class for dataset generators."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize generator.

        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_examples(self, num_examples: int) -> str:
        """Generate examples and return as text.

        Args:
            num_examples: Number of examples to generate

        Returns:
            Generated examples as text
        """
        pass

    def save(self, text: str, train_split: float = 0.9) -> None:
        """Save generated text to binary files.

        Args:
            text: Generated text
            train_split: Fraction of data for training
        """
        # Save text
        input_path = self.output_dir / "input.txt"
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved {len(text):,} characters to {input_path}")

        # Create train/val split
        n = len(text)
        train_data = text[: int(n * train_split)]
        val_data = text[int(n * train_split) :]

        # Encode with tiktoken
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)

        print(f"Train: {len(train_ids):,} tokens")
        print(f"Val: {len(val_ids):,} tokens")

        # Save binary files
        train_ids_array = np.array(train_ids, dtype=np.uint16)
        val_ids_array = np.array(val_ids, dtype=np.uint16)
        train_ids_array.tofile(self.output_dir / "train.bin")
        val_ids_array.tofile(self.output_dir / "val.bin")

        print(f"Saved binary files to {self.output_dir}")


def save_binary_dataset(
    data: str,
    output_dir: Path,
    train_split: float = 0.9,
    encoding: str = "gpt2",
) -> None:
    """Save text data as binary dataset.

    Args:
        data: Text data
        output_dir: Directory to save files
        train_split: Fraction for training
        encoding: Tiktoken encoding name
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text
    input_path = output_dir / "input.txt"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(data)

    # Split
    n = len(data)
    train_data = data[: int(n * train_split)]
    val_data = data[int(n * train_split) :]

    # Encode
    enc = tiktoken.get_encoding(encoding)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    print(f"Train: {len(train_ids):,} tokens")
    print(f"Val: {len(val_ids):,} tokens")

    # Save
    np.array(train_ids, dtype=np.uint16).tofile(output_dir / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(output_dir / "val.bin")
    
    print(f"Saved binary files to {output_dir}")


def save_metadata(
    output_dir: Path,
    vocab_size: int,
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    """Save character-level encoding metadata.

    Args:
        output_dir: Directory to save metadata
        vocab_size: Vocabulary size
        stoi: String to integer mapping
        itos: Integer to string mapping
    """
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {output_dir / 'meta.pkl'}")


def create_char_level_encoding(text: str) -> tuple[dict[str, int], dict[int, str], int]:
    """Create character-level encoding from text.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (stoi, itos, vocab_size)
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size


def encode_char_level(text: str, stoi: dict[str, int]) -> list[int]:
    """Encode text using character-level mapping.

    Args:
        text: Text to encode
        stoi: String to integer mapping

    Returns:
        List of token IDs
    """
    return [stoi[c] for c in text]


def save_char_level_dataset(
    data: str,
    output_dir: Path,
    train_split: float = 0.9,
) -> None:
    """Save text data as character-level binary dataset.

    Args:
        data: Text data
        output_dir: Directory to save files
        train_split: Fraction for training
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text
    input_path = output_dir / "input.txt"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(data)

    print(f"Dataset length: {len(data):,} characters")

    # Create character encoding
    stoi, itos, vocab_size = create_char_level_encoding(data)
    print(f"Vocab size: {vocab_size}")

    # Save metadata
    save_metadata(output_dir, vocab_size, stoi, itos)

    # Split
    n = len(data)
    train_data = data[: int(n * train_split)]
    val_data = data[int(n * train_split) :]

    # Encode
    train_ids = encode_char_level(train_data, stoi)
    val_ids = encode_char_level(val_data, stoi)

    print(f"Train: {len(train_ids):,} tokens")
    print(f"Val: {len(val_ids):,} tokens")

    # Save
    np.array(train_ids, dtype=np.uint16).tofile(output_dir / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(output_dir / "val.bin")

    print(f"Saved to {output_dir}")
