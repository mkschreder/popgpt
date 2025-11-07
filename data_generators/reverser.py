#!/usr/bin/env python3
"""
Prepare the String Reverser dataset using character-level encoding.
Generates random strings with their reversed versions.
"""

import random
import string
from pathlib import Path

from common import save_char_level_dataset


MAX_CHARS_PER_LINE = 64


def generate_simple_string() -> str:
    """Generate lowercase string."""
    length = random.randint(3, 20)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def generate_mixed_case_string() -> str:
    """Generate mixed case string."""
    length = random.randint(3, 20)
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def generate_alphanumeric_string() -> str:
    """Generate alphanumeric string."""
    length = random.randint(3, 25)
    char_set = string.ascii_letters + string.digits
    return "".join(random.choice(char_set) for _ in range(length))


def generate_string_with_spaces() -> str:
    """Generate string that includes spaces."""
    length = random.randint(3, 20)
    char_set = string.ascii_letters + string.digits + " "
    parts = []
    for _ in range(length):
        char = random.choice(char_set)
        parts.append(char)
    # Ensure at least one space is included
    result = "".join(parts)
    if " " not in result:
        # Insert a space at a random position
        pos = random.randint(1, len(result) - 1)
        result = result[:pos] + " " + result[pos:]
    return result


def generate_string_with_char_limit(max_attempts: int = 50) -> str:
    """Generate valid string reversal example within character limit."""
    for _ in range(max_attempts):
        rand = random.random()
        if rand < 0.25:
            input_str = generate_simple_string()
        elif rand < 0.5:
            input_str = generate_mixed_case_string()
        elif rand < 0.75:
            input_str = generate_alphanumeric_string()
        else:
            input_str = generate_string_with_spaces()

        # Normalize whitespace but preserve spaces
        input_str = " ".join(input_str.split())
        if not input_str:
            continue

        reversed_str = input_str[::-1]
        line = f"{input_str}={reversed_str}\n"

        if len(line) <= MAX_CHARS_PER_LINE:
            return line

    # Fallback
    input_str = generate_simple_string()[:10]
    reversed_str = input_str[::-1]
    return f"{input_str}={reversed_str}\n"


def main() -> None:
    """Generate reverser dataset."""
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "reverser"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples
    num_examples = 2000000
    print(f"Generating {num_examples:,} string reversal examples...")
    print(f"Max characters per line: {MAX_CHARS_PER_LINE}")

    examples = []

    for i in range(num_examples):
        line = generate_string_with_char_limit()
        examples.append(line)

        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1:,}/{num_examples:,} examples")

    # Combine into text
    data = "".join(examples)

    # Save as character-level dataset
    save_char_level_dataset(data, data_dir)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
