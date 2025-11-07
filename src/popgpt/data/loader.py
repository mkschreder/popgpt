"""Data loading implementation for PopGPT."""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import DataConfig


class DataLoader:
    """Data loader for training and validation data.

    Implements the DataLoaderProtocol.
    """

    def __init__(
        self,
        data_dir: Path,
        config: DataConfig,
        device: str,
        device_type: str,
    ) -> None:
        """Initialize data loader.

        Args:
            data_dir: Directory containing train.bin and val.bin
            config: Data configuration
            device: Device string ('cpu', 'cuda', etc.)
            device_type: Device type ('cpu' or 'cuda')
        """
        self.data_dir = data_dir
        self.config = config
        self.device = device
        self.device_type = device_type

        # Load metadata for masking if needed
        self.mask_token_id: Optional[int] = None
        self.newline_token_id: Optional[int] = None
        self._load_metadata()

        # Pre-compute line starts if alignment is enabled
        self.train_line_starts: Optional[np.ndarray] = None
        self.val_line_starts: Optional[np.ndarray] = None
        if config.align_to_lines:
            self._precompute_line_starts()

    def _load_metadata(self) -> None:
        """Load metadata for token masking and alignment."""
        if self.config.mask_before_token is None and not self.config.align_to_lines:
            return

        meta_path = self.data_dir / "meta.pkl"
        if meta_path.exists():
            # Use character-level encoding from meta.pkl
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            # Print vocabulary information
            if "stoi" in meta and "itos" in meta:
                vocab_size = len(meta["stoi"])
                chars = sorted(meta["stoi"].keys())
                # Format special characters for display
                display_chars = []
                for ch in chars[:50]:  # Show first 50 chars
                    if ch == "\n":
                        display_chars.append("\\n")
                    elif ch == "\t":
                        display_chars.append("\\t")
                    elif ch == "\r":
                        display_chars.append("\\r")
                    elif ch == " ":
                        display_chars.append("SPACE")
                    else:
                        display_chars.append(ch)

                suffix = f" ... (+{vocab_size - 50} more)" if vocab_size > 50 else ""
                print(f"Loaded character-level vocabulary: {vocab_size} chars")
                print(f"  Characters: {' '.join(display_chars)}{suffix}")

            if (
                self.config.mask_before_token is not None
                and "stoi" in meta
                and self.config.mask_before_token in meta["stoi"]
            ):
                self.mask_token_id = meta["stoi"][self.config.mask_before_token]
                print(
                    f"Masking tokens before and including '{self.config.mask_before_token}' "
                    f"(token_id={self.mask_token_id})"
                )

            if "stoi" in meta and "\n" in meta["stoi"]:
                self.newline_token_id = meta["stoi"]["\n"]
        else:
            # Use tiktoken GPT-2 tokenizer
            try:
                import tiktoken

                enc = tiktoken.get_encoding("gpt2")
                if self.config.mask_before_token is not None:
                    mask_token_ids = enc.encode(self.config.mask_before_token)
                    if len(mask_token_ids) == 1:
                        self.mask_token_id = mask_token_ids[0]
                        print(
                            f"Masking tokens before and including '{self.config.mask_before_token}' "
                            f"(token_id={self.mask_token_id})"
                        )
                    else:
                        print(
                            f"Warning: mask_before_token '{self.config.mask_before_token}' "
                            f"encodes to {len(mask_token_ids)} tokens, masking disabled"
                        )

                newline_token_ids = enc.encode("\n")
                if len(newline_token_ids) == 1:
                    self.newline_token_id = newline_token_ids[0]
            except ImportError:
                print(
                    "Warning: tiktoken not available and meta.pkl not found, "
                    "masking/alignment disabled"
                )

    def _precompute_line_starts(self) -> None:
        """Pre-compute line start positions for aligned sampling."""
        if self.newline_token_id is None:
            print("Warning: align_to_lines=True but newline token not found in vocabulary")
            return

        print("Computing line start positions for aligned sampling...")
        print(
            f"Only including lines where the next newline fits within block_size={self.config.block_size}"
        )

        for split_name in ["train", "val"]:
            data = np.memmap(str(self.data_dir / f"{split_name}.bin"), dtype=np.uint16, mode="r")

            line_starts = []
            newline_positions = np.where(data == self.newline_token_id)[0]

            # Check start of file
            if self.config.block_size <= len(data):
                if len(newline_positions) > 0 and newline_positions[0] < self.config.block_size:
                    line_starts.append(0)
                elif len(newline_positions) == 0:
                    line_starts.append(0)

            # For each newline, check if the next line fits
            for i in range(len(newline_positions) - 1):
                line_start = newline_positions[i] + 1
                next_newline = newline_positions[i + 1]
                if (
                    next_newline - line_start < self.config.block_size
                    and line_start + self.config.block_size <= len(data)
                ):
                    line_starts.append(line_start)

            # Handle last line
            if len(newline_positions) > 0:
                last_line_start = newline_positions[-1] + 1
                if last_line_start + self.config.block_size <= len(data):
                    line_starts.append(last_line_start)

            if split_name == "train":
                self.train_line_starts = np.array(line_starts, dtype=np.int64)
            else:
                self.val_line_starts = np.array(line_starts, dtype=np.int64)

            print(f"  {split_name}: found {len(line_starts)} valid line starts")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data.

        Args:
            split: 'train' or 'val'

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Recreate memmap to avoid memory leak
        data = np.memmap(str(self.data_dir / f"{split}.bin"), dtype=np.uint16, mode="r")

        # Get line starts for this split
        line_starts = self.train_line_starts if split == "train" else self.val_line_starts

        # Sample starting positions
        if self.config.align_to_lines and line_starts is not None:
            line_indices = torch.randint(len(line_starts), (self.config.batch_size,))
            ix = torch.tensor([line_starts[i] for i in line_indices], dtype=torch.long)
        else:
            ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))

        # Create input and target tensors
        x = torch.stack(
            [torch.from_numpy((data[i : i + self.config.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + self.config.block_size]).astype(np.int64))
                for i in ix
            ]
        )

        # Apply masking if configured
        if self.config.mask_before_token is not None and self.mask_token_id is not None:
            y = self._apply_masking(y)

        # Move to device
        if self.device_type == "cuda":
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _apply_masking(self, y: torch.Tensor) -> torch.Tensor:
        """Apply token masking to targets.

        Masks tokens up to and including the mask_before_token on each line,
        allowing the model to predict the output portion.
        """
        if self.config.mask_per_line:
            # Mask per-line: mask everything before and including the mask token on each line
            for batch_idx in range(y.shape[0]):
                line_start = 0
                for pos in range(y.shape[1]):
                    if y[batch_idx, pos] == self.mask_token_id:
                        # Mask from line start up to and including the mask token
                        y[batch_idx, line_start : pos + 1] = -1
                    if (
                        self.newline_token_id is not None
                        and y[batch_idx, pos] == self.newline_token_id
                    ):
                        line_start = pos + 1

                # NOTE: Removed the partial line masking at the end
                # The model should learn to predict partial outputs too
        else:
            # Mask globally: mask everything before and including the first occurrence
            mask_positions = (y == self.mask_token_id).nonzero(as_tuple=True)
            for batch_idx in range(y.shape[0]):
                batch_mask_pos = mask_positions[1][mask_positions[0] == batch_idx]
                if len(batch_mask_pos) > 0:
                    first_mask_pos = batch_mask_pos[0].item()
                    y[batch_idx, : first_mask_pos + 1] = -1

        # Log masking statistics (only on first call)
        if not hasattr(self, "_masking_stats_logged"):
            masked_tokens = (y == -1).sum().item()
            total_tokens = y.numel()
            pct = 100.0 * masked_tokens / total_tokens
            print(
                f"  Masking statistics: {masked_tokens}/{total_tokens} tokens masked ({pct:.1f}%)"
            )
            print(
                f"  Loss will be computed on {total_tokens - masked_tokens} tokens ({100-pct:.1f}%)"
            )
            self._masking_stats_logged = True

        return y
