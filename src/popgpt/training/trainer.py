"""Main training orchestrator following Single Responsibility Principle."""

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import TrainingConfig
from ..core import GPT, GPTConfig
from ..data import DataLoader
from ..checkpoint import CheckpointManager
from ..utils import setup_device, setup_ddp, cleanup_ddp, plot_loss_history
from .evaluator import Evaluator
from .scheduler import CosineDecayWithWarmup


class Trainer:
    """Main training orchestrator.

    Follows Single Responsibility Principle by delegating specific tasks
    to specialized components (DataLoader, Evaluator, CheckpointManager, etc.)
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Setup distributed training
        (
            self.ddp,
            self.ddp_rank,
            self.ddp_local_rank,
            self.ddp_world_size,
            self.master_process,
            seed_offset,
        ) = setup_ddp()

        # Adjust gradient accumulation for DDP
        if self.ddp:
            if config.optimizer.gradient_accumulation_steps % self.ddp_world_size != 0:
                raise ValueError(
                    f"gradient_accumulation_steps ({config.optimizer.gradient_accumulation_steps}) "
                    f"must be divisible by world_size ({self.ddp_world_size}) for DDP training. "
                    f"Please set gradient_accumulation_steps to a multiple of {self.ddp_world_size}."
                )
            self.gradient_accumulation_steps = (
                config.optimizer.gradient_accumulation_steps // self.ddp_world_size
            )
        else:
            self.gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps

        # Setup device and precision
        device = config.system.device
        if self.ddp:
            device = f"cuda:{self.ddp_local_rank}"

        self.device, self.device_type, self.ctx, self.ptdtype = setup_device(
            device, config.system.dtype
        )

        # Set random seed
        torch.manual_seed(config.system.seed + seed_offset)

        # Calculate tokens per iteration
        self.tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.ddp_world_size
            * config.data.batch_size
            * config.data.block_size
        )
        print(f"Tokens per iteration: {self.tokens_per_iter:,}")

        # Initialize components
        self.data_loader = self._create_data_loader()
        # Ensure out_dir is a Path (handles both str and Path)
        self.out_dir = (
            Path(config.io.out_dir) if isinstance(config.io.out_dir, str) else config.io.out_dir
        )
        self.checkpoint_manager = CheckpointManager(self.out_dir)

        # Training state (will be overwritten if resuming)
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.loss_history = {"iters": [], "train_loss": [], "val_loss": []}

        # Initialize model_args that will be used for checkpointing
        self.model_args = {}

        # Load from checkpoint first if resuming (creates model and optimizer)
        # Otherwise create model and optimizer normally
        if config.io.init_from == "resume" or config.io.resume_from is not None:
            self._load_checkpoint()
        else:
            self.model = self._create_model()
            self.optimizer = self._create_optimizer()

        self.lr_scheduler = self._create_lr_scheduler()
        self.evaluator = Evaluator(self.model, self.data_loader, config.io.eval_iters, self.ctx)

        # Initialize GradScaler for fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.system.dtype == "float16"))

        # Compile model if enabled (requires PyTorch 2.0+)
        if config.system.compile:
            # Check PyTorch version for compile support
            torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
            if hasattr(torch, "compile") and torch_version >= (2, 0):
                try:
                    print("Compiling model... (takes ~1 minute)")
                    self.unoptimized_model = self.model
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print(f"Warning: torch.compile failed: {e}")
                    print("Continuing without compilation")
                    config.system.compile = False
            else:
                print(
                    f"Warning: torch.compile not available (requires PyTorch 2.0+, found {torch.__version__}), skipping compilation"
                )
                config.system.compile = False  # Disable so we don't try to revert later

        # Wrap in DDP if distributed
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        # Get raw model (unwrap DDP if needed)
        self.raw_model = self.model.module if self.ddp else self.model

        # Wandb logging
        self.wandb_log = config.wandb.enabled and self.master_process
        if self.wandb_log:
            import wandb

            wandb.init(
                project=config.wandb.project,
                name=config.wandb.run_name,
                config=self._config_to_dict(),
            )

    def _create_data_loader(self) -> DataLoader:
        """Create data loader."""
        data_dir = self.config.get_data_dir()
        return DataLoader(
            data_dir=data_dir,
            config=self.config.data,
            device=self.device,
            device_type=self.device_type,
        )

    def _create_model(self) -> GPT:
        """Create or load model."""
        # Detect vocab size from dataset
        meta_vocab_size = self._detect_vocab_size()

        # Build model arguments
        model_args = {
            "n_layer": self.config.model.n_layer,
            "n_head": self.config.model.n_head,
            "n_embd": self.config.model.n_embd,
            "block_size": self.config.data.block_size,
            "bias": self.config.model.bias,
            "vocab_size": None,
            "dropout": self.config.model.dropout,
        }

        if self.config.io.init_from == "scratch":
            print("Initializing a new model from scratch")
            if meta_vocab_size is None:
                print("Defaulting to vocab_size of 50304 (GPT-2 rounded up for efficiency)")
            model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif self.config.io.init_from == "resume":
            # This shouldn't be reached - resume is handled in __init__ before calling _create_model
            raise RuntimeError(
                "Internal error: _create_model() should not be called when init_from='resume'. "
                "The checkpoint should be loaded first."
            )
        elif self.config.io.init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {self.config.io.init_from}")
            override_args = {"dropout": self.config.model.dropout}
            model = GPT.from_pretrained(self.config.io.init_from, override_args)
            # Read off the created config params
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = getattr(model.config, k)
        else:
            raise ValueError(f"Unknown init_from: {self.config.io.init_from}")

        # Crop block size if needed
        if self.config.data.block_size < model.config.block_size:
            model.crop_block_size(self.config.data.block_size)
            model_args["block_size"] = self.config.data.block_size

        # Store model args for checkpointing
        self.model_args = model_args

        model.to(self.device)
        return model

    def _detect_vocab_size(self) -> Optional[int]:
        """Detect vocabulary size from dataset metadata."""
        meta_path = self.config.get_data_dir() / "meta.pkl"
        if meta_path.exists():
            import pickle

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            vocab_size = meta["vocab_size"]
            print(f"Found vocab_size = {vocab_size} (from {meta_path})")

            # Print vocabulary if available
            if "stoi" in meta and "itos" in meta:
                chars = sorted(meta["stoi"].keys())
                # Format special characters for display
                display_chars = []
                for ch in chars:
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
                print(f"Vocabulary ({len(chars)} chars): {' '.join(display_chars)}")

            return vocab_size
        return None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return self.model.configure_optimizers(
            weight_decay=self.config.optimizer.weight_decay,
            learning_rate=self.config.optimizer.learning_rate,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            device_type=self.device_type,
        )

    def _create_lr_scheduler(self) -> CosineDecayWithWarmup:
        """Create learning rate scheduler."""
        return CosineDecayWithWarmup(
            learning_rate=self.config.optimizer.learning_rate,
            warmup_iters=self.config.lr_schedule.warmup_iters,
            lr_decay_iters=self.config.lr_schedule.lr_decay_iters,
            min_lr=self.config.lr_schedule.min_lr,
            decay_enabled=self.config.lr_schedule.decay_lr,
        )

    def _load_checkpoint(self) -> None:
        """Load from checkpoint."""
        # Determine checkpoint path
        if self.config.io.resume_from:
            # Ensure resume_from is a Path (handles both str and Path)
            checkpoint_path = (
                Path(self.config.io.resume_from)
                if isinstance(self.config.io.resume_from, str)
                else self.config.io.resume_from
            )
            if checkpoint_path.is_dir():
                checkpoint_path = checkpoint_path / "ckpt.pt"

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Cannot resume training. Either create the checkpoint first or use init_from: 'scratch'"
                )

            print(f"Initializing from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            # Ensure out_dir is a Path (handles both str and Path)
            out_dir = (
                Path(self.config.io.out_dir)
                if isinstance(self.config.io.out_dir, str)
                else self.config.io.out_dir
            )
            checkpoint_path = out_dir / "ckpt.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Cannot resume training from {self.config.io.out_dir}. "
                    f"Either train from scratch first or specify a different checkpoint with resume_from."
                )

            print(f"Resuming training from {out_dir}")
            checkpoint = self.checkpoint_manager.load()

        # Restore model args from checkpoint
        checkpoint_model_args = checkpoint["model_args"]

        if self.config.io.weights_only:
            # For weights-only loading (fine-tuning), use config architecture
            print("Loading weights only - using model architecture from config")
            meta_vocab_size = self._detect_vocab_size()
            # Priority: config.model.vocab_size > meta_vocab_size > default 50304
            vocab_size = self.config.model.vocab_size or meta_vocab_size or 50304
            self.model_args = {
                "n_layer": self.config.model.n_layer,
                "n_head": self.config.model.n_head,
                "n_embd": self.config.model.n_embd,
                "block_size": self.config.data.block_size,
                "bias": self.config.model.bias,
                "vocab_size": vocab_size,
                "dropout": self.config.model.dropout,
            }

            # Warn if architectures differ
            arch_diffs = []
            for k in ["n_layer", "n_head", "n_embd", "vocab_size"]:
                if k in checkpoint_model_args and checkpoint_model_args[k] != self.model_args[k]:
                    arch_diffs.append(
                        f"{k}: checkpoint={checkpoint_model_args[k]} vs config={self.model_args[k]}"
                    )
            if arch_diffs:
                print("⚠ Warning: Model architecture differs from checkpoint:")
                for diff in arch_diffs:
                    print(f"  - {diff}")
                print("  Some weights may not load correctly!")
        else:
            # For full resume, use checkpoint architecture (required to load optimizer state)
            print("Resuming training - using model architecture from checkpoint")
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                self.model_args[k] = checkpoint_model_args[k]

            # Warn if config differs from checkpoint
            config_arch = {
                "n_layer": self.config.model.n_layer,
                "n_head": self.config.model.n_head,
                "n_embd": self.config.model.n_embd,
            }
            arch_diffs = []
            for k, v in config_arch.items():
                if k in self.model_args and self.model_args[k] != v:
                    arch_diffs.append(f"{k}: checkpoint={self.model_args[k]} vs config={v}")
            if arch_diffs:
                print("ℹ Note: Config model architecture ignored (using checkpoint architecture):")
                for diff in arch_diffs:
                    print(f"  - {diff}")

        # Recreate model
        gptconf = GPTConfig(**self.model_args)
        self.model = GPT(gptconf)

        # Load state dict
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)

        # Move to device
        self.model.to(self.device)

        # Recreate optimizer with the new model's parameters
        self.optimizer = self._create_optimizer()

        # Restore training state (unless weights_only)
        if not self.config.io.weights_only:
            self.iter_num = checkpoint.get("iter_num", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", 1e9)

            # Load optimizer state if available
            if "optimizer" in checkpoint:
                try:
                    # Load optimizer state dict
                    optimizer_state = checkpoint["optimizer"]

                    # Move optimizer state to the correct device
                    # This handles cases where checkpoint was saved on different device
                    if "state" in optimizer_state:
                        for state in optimizer_state["state"].values():
                            if isinstance(state, dict):
                                for k, v in list(state.items()):
                                    if torch.is_tensor(v):
                                        state[k] = v.to(self.device)

                    self.optimizer.load_state_dict(optimizer_state)
                    print(f"Restored optimizer state")
                except Exception as e:
                    import traceback

                    print(f"\nWarning: Could not restore optimizer state: {type(e).__name__}: {e}")
                    print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
                    print(f"  Checkpoint may be from a different model architecture or device.")
                    print(f"  Continuing with fresh optimizer state.")
                    if os.getenv("DEBUG"):
                        traceback.print_exc()
                    print()

            # Load loss history
            if self.config.io.resume_from is None:
                self.loss_history = self.checkpoint_manager.load_loss_history()
                print(f"Loaded loss history with {len(self.loss_history['iters'])} checkpoints")
        else:
            print("Loaded weights only (training state not restored)")

    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return self.config.model_dump()

    def train(self) -> None:
        """Main training loop."""
        # Eval-only mode
        if self.config.io.eval_only:
            if self.master_process:
                losses = self.evaluator.estimate_loss()
                print(
                    f"Eval-only mode - train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}"
                )
            return

        # Training loop
        X, Y = self.data_loader.get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        while True:
            # Set learning rate
            lr = self.lr_scheduler.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate and checkpoint
            if self.iter_num % self.config.io.eval_interval == 0 and self.master_process:
                losses = self.evaluator.estimate_loss()
                train_ppl = math.exp(losses["train"])
                val_ppl = math.exp(losses["val"])
                print(
                    f"Step {self.iter_num}: train loss {losses['train']:.4f}, "
                    f"train ppl {train_ppl:.2f}, val loss {losses['val']:.4f}, val ppl {val_ppl:.2f}"
                )

                # Update loss history
                self.loss_history["iters"].append(self.iter_num)
                self.loss_history["train_loss"].append(losses["train"])
                self.loss_history["val_loss"].append(losses["val"])

                # Wandb logging
                if self.wandb_log:
                    import wandb

                    wandb.log(
                        {
                            "iter": self.iter_num,
                            "train/loss": losses["train"],
                            "train/perplexity": train_ppl,
                            "val/loss": losses["val"],
                            "val/perplexity": val_ppl,
                            "lr": lr,
                            "mfu": running_mfu * 100,
                        }
                    )

                # Save checkpoint
                if (
                    losses["val"] < self.best_val_loss or self.config.io.always_save_checkpoint
                ) and self.iter_num > 0:
                    self.best_val_loss = losses["val"]
                    self.checkpoint_manager.save(
                        model_state=self.raw_model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        config={
                            "model_args": self.model_args,
                            **self._config_to_dict(),
                        },
                        iter_num=self.iter_num,
                        best_val_loss=self.best_val_loss,
                    )
                    self.checkpoint_manager.save_loss_history(self.loss_history)
                    plot_loss_history(self.loss_history, self.out_dir)

            # Training step
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )

                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.gradient_accumulation_steps

                # Prefetch next batch
                X, Y = self.data_loader.get_batch("train")

                # Backward pass
                self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.optimizer.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.grad_clip
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % self.config.io.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.gradient_accumulation_steps
                ppl = math.exp(lossf)
                if local_iter_num >= 5:
                    mfu = self.raw_model.estimate_mfu(
                        self.config.data.batch_size * self.gradient_accumulation_steps, dt
                    )
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(
                    f"Iter {self.iter_num}: loss {lossf:.4f}, ppl {ppl:.2f}, "
                    f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )

            self.iter_num += 1
            local_iter_num += 1

            # Check termination
            if self.iter_num >= self.config.optimizer.max_iters:
                break

        # Cleanup
        cleanup_ddp(self.ddp)
