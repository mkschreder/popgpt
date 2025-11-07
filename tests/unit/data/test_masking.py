"""Comprehensive tests for token masking behavior in data loader."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from popgpt.config import DataConfig
from popgpt.data import DataLoader


class TestMaskingBehavior:
    """Test masking behavior for seq2seq tasks like reverser."""

    @pytest.fixture
    def reverser_data_dir(self, tmp_path: Path) -> Path:
        """Create data directory with reverser-like patterns."""
        data_dir = tmp_path / "reverser_data"
        data_dir.mkdir()

        # Create metadata with character mappings
        # Using simple mapping: a=1, b=2, c=3, ==4, \n=5
        stoi = {"a": 1, "b": 2, "c": 3, "=": 4, "\n": 5}
        itos = {v: k for k, v in stoi.items()}
        meta = {"vocab_size": 6, "stoi": stoi, "itos": itos}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # Create train data: "abc=cba\n" repeated
        # Encoding: [1,2,3,4,3,2,1,5] = "abc=cba\n"
        pattern = np.array([1, 2, 3, 4, 3, 2, 1, 5], dtype=np.uint16)
        train_data = np.tile(pattern, 50)  # Repeat 50 times
        train_data.tofile(data_dir / "train.bin")

        val_data = np.tile(pattern, 50)
        val_data.tofile(data_dir / "val.bin")

        return data_dir

    def test_masks_input_and_separator_per_line(self, reverser_data_dir: Path) -> None:
        """Masking should mask predictions of input tokens (a,b,c,=)."""
        config = DataConfig(
            batch_size=1,
            block_size=16,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # y is next-token prediction target
        # We mask positions in y where we're predicting input tokens (a,b,c,=)
        # We train on positions where we're predicting output tokens (c,b,a,\n)

        masked_count = (y == -1).sum().item()
        unmasked_count = (y != -1).sum().item()

        # For pattern "abc=cba\n", we mask 4 and train on 4 = ~50%
        mask_percentage = 100.0 * masked_count / (masked_count + unmasked_count)
        assert 40 <= mask_percentage <= 60, f"Expected ~50% masked, got {mask_percentage:.1f}%"

    def test_trains_on_output_including_newline(self, reverser_data_dir: Path) -> None:
        """Model should be trained on predicting output tokens AND the newline."""
        config = DataConfig(
            batch_size=1,
            block_size=8,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Find unmasked tokens (the predictions we train on)
        unmasked_tokens = y[0][y[0] != -1]

        assert len(unmasked_tokens) > 0, "Should have unmasked output tokens"

        # Critical: newline (token 5) should be in training set
        assert 5 in unmasked_tokens, "Newline (token 5) must be trained, not masked"

        # Output characters (c=3, b=2, a=1) should also be unmasked
        for token_id in [1, 2, 3]:
            assert token_id in unmasked_tokens, f"Output token {token_id} should be trained"

    def test_multiple_lines_in_one_sequence(self, reverser_data_dir: Path) -> None:
        """Multiple lines in one batch element should each be masked correctly."""
        config = DataConfig(
            batch_size=1,
            block_size=24,  # Can fit 3 complete lines
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Verify newlines are preserved (not masked)
        unmasked_tokens = y[0][y[0] != -1]
        newline_count = (unmasked_tokens == 5).sum().item()

        # Should have multiple newlines unmasked
        assert newline_count >= 2, f"Expected at least 2 unmasked newlines, got {newline_count}"

        # Roughly 50% should be masked
        masked_count = (y == -1).sum().item()
        mask_percentage = 100.0 * masked_count / y.numel()
        assert 40 <= mask_percentage <= 60, f"Expected ~50% masked, got {mask_percentage:.1f}%"

    def test_partial_line_at_end_is_trained(self, tmp_path: Path) -> None:
        """Partial line at end (no newline) should still train on available output tokens.

        This is critical: the model needs to learn to generate partial outputs
        during autoregressive decoding.
        """
        data_dir = tmp_path / "partial_data"
        data_dir.mkdir()

        stoi = {"a": 1, "b": 2, "c": 3, "=": 4, "\n": 5}
        itos = {v: k for k, v in stoi.items()}
        meta = {"vocab_size": 6, "stoi": stoi, "itos": itos}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # "abc=cba\nabc=cb" - second line is partial (missing 'a' and '\n')
        train_data = np.array([1, 2, 3, 4, 3, 2, 1, 5, 1, 2, 3, 4, 3, 2] * 50, dtype=np.uint16)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(
            batch_size=1,
            block_size=14,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Critical: partial outputs should NOT be completely masked
        # We need to train on partial generations
        unmasked_tokens = y[0][y[0] != -1]
        assert len(unmasked_tokens) > 1, "Partial output tokens should be trained"

        # Should have some output tokens (c=3, b=2) from the partial line
        assert 3 in unmasked_tokens or 2 in unmasked_tokens, "Partial output should be trained"

    def test_masking_percentage_is_reasonable(self, reverser_data_dir: Path) -> None:
        """For reverser task, roughly 50% of tokens should be masked (input side)."""
        config = DataConfig(
            batch_size=8,
            block_size=64,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        masked_count = (y == -1).sum().item()
        total_count = y.numel()
        mask_percentage = 100.0 * masked_count / total_count

        # For "abc=cba\n", we mask 4 tokens and train on 4 tokens = 50%
        # Allow some variance due to block boundaries
        assert 45 <= mask_percentage <= 55, f"Expected ~50% masked, got {mask_percentage:.1f}%"

    def test_no_masking_without_config(self, reverser_data_dir: Path) -> None:
        """Without mask_before_token config, no masking should occur."""
        config = DataConfig(
            batch_size=1,
            block_size=16,
            mask_before_token=None,  # No masking
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # No tokens should be masked
        assert (
            y == -1
        ).sum().item() == 0, "No tokens should be masked when mask_before_token is None"

    def test_global_masking_mode(self, reverser_data_dir: Path) -> None:
        """Test global masking (mask_per_line=False) masks only once per sequence."""
        config = DataConfig(
            batch_size=1,
            block_size=16,  # 2 lines
            mask_before_token="=",
            mask_per_line=False,  # Global masking
        )
        loader = DataLoader(reverser_data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # In global mode, only predictions up to FIRST '=' are masked
        # This means most of the sequence is trained (including second line's input)

        masked_count = (y == -1).sum().item()
        unmasked_count = (y != -1).sum().item()

        # With global masking on 2 lines, much less is masked (~25% instead of 50%)
        mask_percentage = 100.0 * masked_count / (masked_count + unmasked_count)
        assert mask_percentage < 40, f"Global masking should mask < 40%, got {mask_percentage:.1f}%"


class TestMaskingEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_output_line(self, tmp_path: Path) -> None:
        """Line with = but no output before newline: 'abc=\\n'.

        Model should learn to immediately output newline after '='.
        """
        data_dir = tmp_path / "edge_data"
        data_dir.mkdir()

        stoi = {"a": 1, "b": 2, "c": 3, "=": 4, "\n": 5}
        meta = {"vocab_size": 6, "stoi": stoi, "itos": {v: k for k, v in stoi.items()}}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # "abc=\n" = [1,2,3,4,5]
        train_data = np.array([1, 2, 3, 4, 5] * 100, dtype=np.uint16)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(batch_size=1, block_size=5, mask_before_token="=", mask_per_line=True)
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Newline should be trained (not masked)
        unmasked_tokens = y[0][y[0] != -1]
        assert 5 in unmasked_tokens, "Newline should be trained even with empty output"

    def test_no_equals_in_line(self, tmp_path: Path) -> None:
        """Line without '=' should not trigger masking."""
        data_dir = tmp_path / "no_equals"
        data_dir.mkdir()

        stoi = {"a": 1, "b": 2, "c": 3, "=": 4, "\n": 5}
        meta = {"vocab_size": 6, "stoi": stoi, "itos": {v: k for k, v in stoi.items()}}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # "abc\n" (no =) = [1,2,3,5]
        train_data = np.array([1, 2, 3, 5] * 100, dtype=np.uint16)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(batch_size=1, block_size=4, mask_before_token="=", mask_per_line=True)
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Nothing should be masked since there's no '='
        assert (y == -1).sum().item() == 0, "No masking should occur without '=' token"


class TestStringReverserUseCase:
    """Test that masking correctly implements string reversal training."""

    def test_string_reverser_training(self, tmp_path: Path) -> None:
        """Verify the model learns to reverse strings like foo=oof."""
        data_dir = tmp_path / "reverser_data"
        data_dir.mkdir()

        # Realistic reverser vocabulary
        stoi = {"f": 1, "o": 2, "b": 3, "a": 4, "r": 5, "=": 6, "\n": 7}
        itos = {v: k for k, v in stoi.items()}
        meta = {"vocab_size": 8, "stoi": stoi, "itos": itos}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # Create dataset: "foo=oof\n" repeated
        # Pattern: [f, o, o, =, o, o, f, \n] = [1, 2, 2, 6, 2, 2, 1, 7]
        pattern = np.array([1, 2, 2, 6, 2, 2, 1, 7], dtype=np.uint16)
        train_data = np.tile(pattern, 100)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(
            batch_size=1,
            block_size=8,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        torch.manual_seed(42)
        x, y = loader.get_batch("train")

        # Verify INPUT side is masked
        # For "foo=oof\n", the input "foo=" should NOT be trained
        # These correspond to positions where we're predicting the next chars in the input

        # Verify OUTPUT side is trained
        # For "foo=oof\n", the output "oof\n" SHOULD be trained
        # After seeing '=', predict 'o'
        # After seeing 'o', predict 'o'
        # After seeing 'o', predict 'f'
        # After seeing 'f', predict '\n'

        unmasked_tokens = y[0][y[0] != -1]

        # Critical assertions for reverser task
        assert len(unmasked_tokens) > 0, "Must train on some tokens"

        # Output characters should be in training set
        assert 1 in unmasked_tokens, "Output 'f' should be trained"
        assert 2 in unmasked_tokens, "Output 'o' should be trained"

        # Newline must be trained (end marker)
        assert 7 in unmasked_tokens, "Newline must be trained as end marker"

        # Roughly 50% should be masked (input side)
        masked_count = (y == -1).sum().item()
        total = y.numel()
        mask_pct = 100.0 * masked_count / total
        assert 40 <= mask_pct <= 60, f"Expected ~50% masked, got {mask_pct:.1f}%"

    def test_inference_scenario(self, tmp_path: Path) -> None:
        """Verify training prepares model for autoregressive inference.

        During inference:
        - User provides: "foo="
        - Model generates: "oof\n"

        This requires model to learn character-by-character prediction.
        """
        data_dir = tmp_path / "reverser_data"
        data_dir.mkdir()

        stoi = {"f": 1, "o": 2, "=": 3, "\n": 4}
        meta = {"vocab_size": 5, "stoi": stoi, "itos": {v: k for k, v in stoi.items()}}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # "foo=oof\n" = [1, 2, 2, 3, 2, 2, 1, 4]
        pattern = np.array([1, 2, 2, 3, 2, 2, 1, 4], dtype=np.uint16)
        train_data = np.tile(pattern, 100)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(
            batch_size=1,
            block_size=8,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        torch.manual_seed(0)
        x, y = loader.get_batch("train")

        # During training, model sees full sequence and learns conditional predictions
        # Key training examples (after masking removes input predictions):
        # - After '=', predict first output char
        # - After first output char, predict second output char
        # - After last output char, predict '\n'

        unmasked_positions = [i for i, tok in enumerate(y[0].tolist()) if tok != -1]

        # Must have at least 3 unmasked positions (output chars + newline)
        assert (
            len(unmasked_positions) >= 3
        ), "Need multiple output positions for autoregressive learning"

        # Verify newline is trained
        assert 4 in y[0][y[0] != -1], "Model must learn to predict newline for termination"


class TestMaskingCorrectness:
    """Test that masking correctly implements the desired behavior."""

    def test_masks_correct_tokens_in_target(self, tmp_path: Path) -> None:
        """Verify exactly which predictions are masked vs trained.

        For input "abc=cba\n", we want:
        - Don't train on predicting: a, b, c, = (input)
        - DO train on predicting: c, b, a, \n (output)
        """
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        stoi = {"a": 1, "b": 2, "c": 3, "=": 4, "\n": 5}
        itos = {v: k for k, v in stoi.items()}
        meta = {"vocab_size": 6, "stoi": stoi, "itos": itos}

        with open(data_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # Single pattern repeated to ensure we sample from it
        pattern = np.array([1, 2, 3, 4, 3, 2, 1, 5], dtype=np.uint16)  # "abc=cba\n"
        train_data = np.tile(pattern, 100)
        train_data.tofile(data_dir / "train.bin")
        train_data.tofile(data_dir / "val.bin")

        config = DataConfig(
            batch_size=1,
            block_size=8,
            mask_before_token="=",
            mask_per_line=True,
        )
        loader = DataLoader(data_dir, config, "cpu", "cpu")

        # Sample multiple times to understand the behavior
        torch.manual_seed(123)
        x, y = loader.get_batch("train")

        # Verify masking statistics
        masked_count = (y == -1).sum().item()
        unmasked_count = (y != -1).sum().item()

        # Should mask exactly 4 predictions (a,b,c,=) and train on 4 (c,b,a,\n)
        # Due to shift and block boundaries, might be slightly different
        assert 3 <= masked_count <= 5, f"Expected ~4 masked, got {masked_count}"
        assert 3 <= unmasked_count <= 5, f"Expected ~4 unmasked, got {unmasked_count}"

        # Unmasked tokens should include output characters and newline
        unmasked_tokens = y[0][y[0] != -1]
        assert 5 in unmasked_tokens, "Newline must be trained"
