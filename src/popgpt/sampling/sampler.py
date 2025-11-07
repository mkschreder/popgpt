"""Text generation and sampling."""

import os
import pickle
from pathlib import Path
from typing import Optional, Callable

import torch
import tiktoken

from ..config import SamplingConfig
from ..core import GPT, GPTConfig
from ..utils import setup_device


class Sampler:
    """Text generation from trained models."""

    def __init__(self, config: SamplingConfig) -> None:
        """Initialize sampler.

        Args:
            config: Sampling configuration
        """
        self.config = config

        # Setup device and precision
        self.device, self.device_type, self.ctx, self.ptdtype = setup_device(
            config.device, config.dtype
        )

        # Set random seed
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        # Load model
        self.model = self._load_model()

        # Setup encoder/decoder
        self.encode, self.decode = self._setup_codec()

        # Get stop token ID if configured
        self.stop_token_id = self._get_stop_token_id()

    def _load_model(self) -> GPT:
        """Load model from checkpoint or pretrained."""
        if self.config.init_from == "resume":
            ckpt_path = self.config.out_dir / "ckpt.pt"
            # Always load to CPU first to avoid device issues
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            gptconf = GPTConfig(**checkpoint["model_args"])
            model = GPT(gptconf)

            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)

        elif self.config.init_from.startswith("gpt2"):
            model = GPT.from_pretrained(self.config.init_from, {"dropout": 0.0})

        else:
            raise ValueError(f"Unknown init_from: {self.config.init_from}")

        model.eval()
        # Move to target device after loading
        model.to(self.device)
        print(f"Model loaded on device: {self.device}")

        # Compile if enabled (requires PyTorch 2.0+)
        if self.config.compile:
            # Check PyTorch version for compile support
            torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
            if hasattr(torch, "compile") and torch_version >= (2, 0):
                try:
                    print("Compiling model...")
                    model = torch.compile(model)
                except Exception as e:
                    print(f"Warning: torch.compile failed: {e}")
                    print("Continuing without compilation")
            else:
                print(
                    f"Warning: torch.compile not available (requires PyTorch 2.0+, found {torch.__version__}), skipping compilation"
                )

        return model

    def _setup_codec(self) -> tuple[Callable, Callable]:
        """Setup encoder and decoder for text.

        Returns:
            Tuple of (encode_fn, decode_fn)
        """
        load_meta = False

        if self.config.init_from == "resume":
            # Try to load meta from data directory
            ckpt_path = self.config.out_dir / "ckpt.pt"
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                if "config" in checkpoint and "data" in checkpoint["config"]:
                    dataset = checkpoint["config"]["data"].get("dataset")
                    if dataset:
                        meta_path = Path("data") / dataset / "meta.pkl"
                        load_meta = meta_path.exists()
                        if load_meta:
                            with open(meta_path, "rb") as f:
                                meta = pickle.load(f)
                            stoi, itos = meta["stoi"], meta["itos"]
                            encode = lambda s: [stoi[c] for c in s]
                            decode = lambda l: "".join([itos[i] for i in l])

                            # Print vocabulary information
                            vocab_size = len(stoi)
                            chars = sorted(stoi.keys())
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
                            print(f"Loading character-level vocabulary from {meta_path}")
                            print(f"  Vocabulary: {vocab_size} chars")
                            print(f"  Characters: {' '.join(display_chars)}{suffix}")

                            return encode, decode

        # Default to GPT-2 encodings
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        return encode, decode

    def _get_stop_token_id(self) -> Optional[int]:
        """Get stop token ID if configured."""
        if self.config.stop_token_char is None:
            return None

        # Try to encode with current encoder
        try:
            encoded = self.encode(self.config.stop_token_char)
            if len(encoded) == 1:
                stop_id = encoded[0]
                print(
                    f"Stop generation at token '{self.config.stop_token_char}' "
                    f"(token_id={stop_id})"
                )
                return stop_id
            else:
                print(
                    f"Warning: stop_token_char '{self.config.stop_token_char}' "
                    f"encodes to {len(encoded)} tokens, cannot use as stop token"
                )
        except Exception as e:
            print(f"Warning: Could not encode stop token: {e}")

        return None

    def generate(
        self, start_text: Optional[str | list[str]] = None
    ) -> dict[str, list[str]] | list[str]:
        """Generate text samples.

        Args:
            start_text: Optional start text(s) - single string or list of strings (overrides config)

        Returns:
            If start_text is a list: dict mapping start text to list of generated samples
            If start_text is a string: list of generated text samples (backward compatible)
        """
        # Get start text from config if not provided
        if start_text is None:
            start_text = self.config.start

        # Handle list of start sequences
        if isinstance(start_text, list):
            results = {}
            for seq in start_text:
                samples = self._generate_from_single_start(seq)
                results[seq] = samples
            return results

        # Single start text (backward compatible)
        return self._generate_from_single_start(start_text)

    def _generate_from_single_start(self, start_text: str) -> list[str]:
        """Generate samples from a single start text.

        Args:
            start_text: Start text for generation

        Returns:
            List of generated text samples
        """
        # Handle FILE: prefix
        if start_text.startswith("FILE:"):
            file_path = Path(start_text[5:])
            with open(file_path, "r", encoding="utf-8") as f:
                start_text = f.read()

        # Encode start text
        start_ids = self.encode(start_text)

        # Handle empty start (need at least one token for generation)
        if len(start_ids) == 0:
            # Use a neutral start token (e.g., newline or space)
            start_ids = self.encode("\n")  # Use newline as default start
            print("Warning: Empty start text, using newline as start token")

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # Ensure input tensor is on correct device
        x = x.to(self.device)

        # Generate samples
        samples = []
        with torch.no_grad():
            with self.ctx:
                for k in range(self.config.num_samples):
                    y = self.model.generate(
                        x,
                        self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        stop_token=self.stop_token_id,
                    )
                    generated_text = self.decode(y[0].cpu().tolist())  # Ensure on CPU for decoding
                    samples.append(generated_text)

        return samples

    def generate_and_print(self) -> None:
        """Generate and print samples to console."""
        results = self.generate()

        # Handle dict results (multiple start sequences)
        if isinstance(results, dict):
            for start_seq, samples in results.items():
                # Calculate correct answer if it's a math expression
                correct_answer = self._calculate_correct_answer(start_seq)
                answer_str = f"{correct_answer}" if correct_answer is not None else ""

                print(f"\n{'='*80}")
                print(
                    f"Results for start sequence: {start_seq[:60]}{'...' if len(start_seq) > 60 else ''}{answer_str}"
                )
                print(f"{'='*80}")

                # Track evaluation results
                total = len(samples)
                correct = 0
                invalid = 0

                for k, text in enumerate(samples, 1):
                    # Try to evaluate if it looks like a math expression
                    eval_result = self._try_evaluate_expression(text, print_result=False)

                    # Format output on a single line
                    status_str = ""
                    if eval_result == "correct":
                        correct += 1
                        status_str = " ✓"
                    elif eval_result == "wrong":
                        status_str = " ✗"
                    elif eval_result == "invalid":
                        invalid += 1
                        status_str = " ✗ (invalid)"

                    print(f"[{k}/{total}] {text.strip()}{status_str}")

                # Print summary for this start sequence
                if correct > 0 or invalid > 0:
                    print(f"\nSummary for '{start_seq}':")
                    print(
                        f"  Total: {total}, Correct: {correct} ({100*correct/total:.1f}%), ", end=""
                    )
                    wrong = total - correct - invalid
                    print(f"Wrong: {wrong} ({100*wrong/total:.1f}%)", end="")
                    if invalid > 0:
                        print(f", Invalid: {invalid} ({100*invalid/total:.1f}%)", end="")
                    print()
                print()

        # Handle list results (single start sequence - backward compatible)
        else:
            samples = results
            total = len(samples)
            correct = 0
            invalid = 0

            for k, text in enumerate(samples, 1):
                # Try to evaluate if it looks like a math expression
                eval_result = self._try_evaluate_expression(text, print_result=False)

                # Format output on a single line
                status_str = ""
                if eval_result == "correct":
                    correct += 1
                    status_str = " ✓"
                elif eval_result == "wrong":
                    status_str = " ✗"
                elif eval_result == "invalid":
                    invalid += 1
                    status_str = " ✗ (invalid)"

                print(f"[{k}/{total}] {text.strip()}{status_str}")

            # Print summary
            if correct > 0 or invalid > 0:
                print(
                    f"\nSummary: Total: {total}, Correct: {correct} ({100*correct/total:.1f}%), ",
                    end="",
                )
                wrong = total - correct - invalid
                print(f"Wrong: {wrong} ({100*wrong/total:.1f}%)", end="")
                if invalid > 0:
                    print(f", Invalid: {invalid} ({100*invalid/total:.1f}%)", end="")
                print()

    def _calculate_correct_answer(self, text: str) -> Optional[float]:
        """Calculate the correct answer for a math expression.

        Args:
            text: Text that may contain a math expression

        Returns:
            The correct numerical answer, or None if not evaluable
        """
        try:
            if "=" in text:
                parts = text.split("=")
                if len(parts) >= 1:
                    expression = parts[0].strip()
                    try:
                        result = eval(expression)
                        return round(result, 4)
                    except (ZeroDivisionError, SyntaxError, NameError, TypeError):
                        pass
        except Exception:
            pass
        return None

    def _try_evaluate_expression(self, text: str, print_result: bool = False) -> Optional[str]:
        """Try to evaluate if text contains a math expression.

        Args:
            text: Generated text to evaluate
            print_result: Whether to print the evaluation result

        Returns:
            "correct", "wrong", "invalid", or None if not evaluable
        """
        try:
            if "=" in text:
                parts = text.split("=")
                if len(parts) >= 2:
                    expression = parts[0].strip()
                    generated_result = parts[1].strip().split("\n")[0].strip()

                    try:
                        correct_result = eval(expression)
                        correct_result = round(correct_result, 4)

                        try:
                            gen_num = float(generated_result)
                            is_correct = abs(gen_num - correct_result) < 0.0001
                            status = "✓ CORRECT" if is_correct else "✗ WRONG"

                            if print_result:
                                print(f"  Generated: {generated_result}")
                                print(f"  Correct:   {correct_result} {status}")

                            return "correct" if is_correct else "wrong"
                        except ValueError:
                            if print_result:
                                print(f"  Generated: {generated_result}")
                                print(f"  Correct:   {correct_result} ✗ INVALID")
                            return "invalid"
                    except (ZeroDivisionError, SyntaxError, NameError):
                        if print_result:
                            print("  (Could not evaluate expression)")
        except Exception:
            pass  # Skip evaluation if parsing fails

        return None
