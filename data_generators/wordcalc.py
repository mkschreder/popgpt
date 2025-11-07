#!/usr/bin/env python3
"""
Prepare the Word Calculator dataset using character-level encoding.
Generates natural language representations of mathematical expressions.
Format: natural language = math expression
"""

import random
from pathlib import Path

import numpy as np

from common import save_char_level_dataset


MAX_CHARS_PER_LINE = 64


def number_to_words(num: int) -> str:
    """Convert a number to its word representation without spaces or hyphens."""
    if num == 0:
        return "zero"

    if num < 0:
        return "negative" + number_to_words(-num)

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if num < 10:
        return ones[num]
    elif num < 20:
        return teens[num - 10]
    elif num < 100:
        tens_digit = num // 10
        ones_digit = num % 10
        if ones_digit == 0:
            return tens[tens_digit]
        return tens[tens_digit] + ones[ones_digit]
    elif num < 1000:
        hundreds_digit = num // 100
        remainder = num % 100
        result = ones[hundreds_digit] + "hundred"
        if remainder > 0:
            result += number_to_words(remainder)
        return result
    elif num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        result = number_to_words(thousands) + "thousand"
        if remainder > 0:
            result += number_to_words(remainder)
        return result
    else:
        return str(num)  # Fallback for very large numbers


def operator_to_word(op: str) -> str:
    """Convert operator to word representation."""
    op_map = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
    return op_map[op]


def generate_simple_expression() -> tuple[str, str]:
    """Generate simple expression and return (math_expr, word_expr)."""
    # Keep numbers small and operations few to fit in 64 chars
    num_ops = random.randint(1, 3)
    numbers = []

    # Generate numbers - keep them small since word form takes more space
    for _ in range(num_ops + 1):
        if random.random() < 0.5:
            # Small numbers (1-20)
            num = random.randint(1, 20)
        elif random.random() < 0.8:
            # Medium numbers (21-100)
            num = random.randint(21, 100)
        else:
            # Larger numbers (100-999)
            num = random.randint(100, 999)

        # Occasionally make negative (but not too often to keep expressions simpler)
        if random.random() < 0.15:
            num = -num

        numbers.append(num)

    operations = [random.choice(["+", "-", "*", "/"]) for _ in range(num_ops)]

    # Build math expression
    math_parts = [str(numbers[0])]
    for i, op in enumerate(operations):
        math_parts.append(op)
        math_parts.append(str(numbers[i + 1]))
    math_expr = "".join(math_parts)

    # Build word expression
    word_parts = [number_to_words(numbers[0])]
    for i, op in enumerate(operations):
        word_parts.append(operator_to_word(op))
        word_parts.append(number_to_words(numbers[i + 1]))
    word_expr = " ".join(word_parts)

    return math_expr, word_expr


def generate_expression_with_char_limit(max_attempts: int = 100) -> str:
    """Generate valid expression within character limit."""
    for _ in range(max_attempts):
        try:
            math_expr, word_expr = generate_simple_expression()

            # Validate the math expression
            result = eval(math_expr)
            if not np.isfinite(result):
                continue

            line = f"{word_expr}={math_expr}\n"

            if len(line) <= MAX_CHARS_PER_LINE:
                return line

        except (ZeroDivisionError, OverflowError, ValueError, SyntaxError):
            continue

    # Fallback: simple addition
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    math_expr = f"{a}+{b}"
    word_expr = f"{number_to_words(a)} plus {number_to_words(b)}"
    return f"{word_expr}={math_expr}\n"


def main() -> None:
    """Generate word calculator dataset."""
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "wordcalc"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples
    num_examples = 2000000
    print(f"Generating {num_examples:,} word calculator examples...")
    print(f"Max characters per line: {MAX_CHARS_PER_LINE}")

    examples = []

    for i in range(num_examples):
        line = generate_expression_with_char_limit()
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
