#!/usr/bin/env python3
"""
Prepare the Calculator dataset using character-level encoding.
Generates diverse mathematical expressions with results.
"""

import random
from pathlib import Path

import numpy as np

from common import save_char_level_dataset


MAX_CHARS_PER_LINE = 64


def generate_number(min_val: float, max_val: float) -> float:
    """Generate a number with varied precision."""
    rand = random.random()
    if rand < 0.4:
        return float(random.randint(int(max(1, min_val)), int(max_val)))
    elif rand < 0.6:
        return round(random.uniform(min_val, max_val), 1)
    elif rand < 0.8:
        return round(random.uniform(min_val, max_val), 2)
    elif rand < 0.9:
        return round(random.uniform(min_val, max_val), 3)
    else:
        return round(random.uniform(min_val, max_val), 4)


def format_number(num: float) -> str:
    """Format number consistently."""
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.10f}".rstrip("0").rstrip(".")


def generate_simple_expression() -> str:
    """Generate simple 1-2 operation expression."""
    num_ops = random.randint(1, 2)
    numbers = []

    for _ in range(num_ops + 1):
        if random.random() < 0.3:
            num = generate_number(0.1, 10.0)
        elif random.random() < 0.6:
            num = generate_number(10.0, 100.0)
        else:
            num = generate_number(100.0, 1000.0)

        if random.random() < 0.3:
            num = -num
        numbers.append(num)

    operations = [random.choice(["+", "-", "*", "/"]) for _ in range(num_ops)]

    expr_parts = [format_number(numbers[0])]
    for i, op in enumerate(operations):
        expr_parts.append(op)
        expr_parts.append(format_number(numbers[i + 1]))

    return "".join(expr_parts)


def generate_expression_with_char_limit(max_attempts: int = 50) -> str:
    """Generate valid expression within character limit."""
    for _ in range(max_attempts):
        expression = generate_simple_expression()

        try:
            result = eval(expression)
            if not np.isfinite(result) or abs(result) > 1e10:
                continue

            result = round(result, 4)
            result_str = format_number(result)
            line = f"{expression}={result_str}\n"

            if len(line) <= MAX_CHARS_PER_LINE:
                return line

        except (ZeroDivisionError, OverflowError, ValueError, SyntaxError):
            continue

    # Fallback
    a = generate_number(1.0, 10.0)
    b = generate_number(1.0, 10.0)
    op = random.choice(["+", "-", "*"])
    expression = f"{format_number(a)}{op}{format_number(b)}"
    result = eval(expression)
    result_str = format_number(round(result, 4))
    return f"{expression}={result_str}\n"


def main() -> None:
    """Generate calculator dataset."""
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "calculator"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples
    num_examples = 2000000
    print(f"Generating {num_examples:,} calculator examples...")
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
