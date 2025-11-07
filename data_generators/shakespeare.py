#!/usr/bin/env python3
"""
Prepare the Shakespeare dataset for character-level language modeling.
Uses character-to-integer encoding with metadata.
"""

import os
from pathlib import Path
import requests

from common import save_char_level_dataset


def download_shakespeare(output_path: Path) -> str:
    """Download tiny shakespeare dataset.

    Args:
        output_path: Path to save downloaded file

    Returns:
        Downloaded text
    """
    if output_path.exists():
        print(f"Using existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    print("Downloading tiny shakespeare dataset...")
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    response = requests.get(url)
    text = response.text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Downloaded to {output_path}")
    return text


def main() -> None:
    """Generate Shakespeare dataset."""
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "shakespeare_char"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    input_path = data_dir / "input.txt"
    data = download_shakespeare(input_path)

    # Save as character-level dataset
    save_char_level_dataset(data, data_dir)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
