"""Pydantic configuration models for PopGPT."""

from typing import Optional, Literal, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class IOConfig(BaseModel):
    """Configuration for input/output operations."""

    out_dir: Path = Field(default=Path("out"), description="Directory for checkpoints and logs")
    eval_interval: int = Field(default=2000, ge=1, description="Evaluation interval in iterations")
    log_interval: int = Field(default=1, ge=1, description="Logging interval in iterations")
    eval_iters: int = Field(default=200, ge=1, description="Number of iterations for evaluation")
    eval_only: bool = Field(default=False, description="Only evaluate, don't train")
    always_save_checkpoint: bool = Field(
        default=True, description="Save checkpoint after each eval"
    )
    init_from: str = Field(
        default="scratch",
        description="Init strategy: 'scratch', 'resume', 'gpt2*', or path to checkpoint",
    )
    resume_from: Optional[Path] = Field(
        default=None,
        description="Path to checkpoint to resume/initialize from (overrides out_dir for checkpoint loading)",
    )
    weights_only: bool = Field(
        default=False,
        description="Only load model weights, not training state (useful for fine-tuning)",
    )

    @field_validator("out_dir", "resume_from", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert strings to Path objects."""
        if v is None:
            return v
        return Path(v) if isinstance(v, str) else v


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(default=False, description="Enable wandb logging")
    project: str = Field(default="popgpt", description="Wandb project name")
    run_name: str = Field(default="run", description="Wandb run name")


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset: str = Field(default="shakespeare_char", description="Dataset name")
    data_dir: Optional[Path] = Field(default=None, description="Override data directory")
    batch_size: int = Field(default=12, ge=1, description="Micro-batch size")
    block_size: int = Field(default=1024, ge=1, description="Sequence length")
    mask_before_token: Optional[str] = Field(
        default=None, description="Mask tokens before this character"
    )
    mask_per_line: bool = Field(default=False, description="Apply masking per line")
    align_to_lines: bool = Field(
        default=False, description="Sample starting positions at line boundaries"
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert strings to Path objects."""
        if v is None:
            return v
        return Path(v) if isinstance(v, str) else v


class ModelConfig(BaseModel):
    """Configuration for GPT model architecture."""

    n_layer: int = Field(default=12, ge=1, description="Number of transformer layers")
    n_head: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of attention heads (auto-calculated if d_head is provided)",
    )
    n_embd: Optional[int] = Field(
        default=None, ge=1, description="Embedding dimension (legacy name for d_model)"
    )
    d_model: Optional[int] = Field(
        default=None, ge=1, description="Model dimension (embedding dimension)"
    )
    d_head: Optional[int] = Field(
        default=None,
        ge=1,
        description="Dimension per attention head (auto-calculated if not provided)",
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    bias: bool = Field(default=False, description="Use bias in Linear and LayerNorm layers")
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size (auto-detected)")

    def model_post_init(self, __context) -> None:
        """Validate and calculate n_head or d_head based on provided values."""
        # Handle d_model / n_embd (they are aliases)
        if self.d_model is not None and self.n_embd is not None:
            if self.d_model != self.n_embd:
                raise ValueError(
                    f"d_model ({self.d_model}) and n_embd ({self.n_embd}) must match if both are provided"
                )
        elif self.d_model is not None:
            self.n_embd = self.d_model
        elif self.n_embd is not None:
            self.d_model = self.n_embd
        else:
            # Set defaults if neither is provided
            self.n_embd = 768
            self.d_model = 768

        # Handle n_head / d_head calculations
        # If both n_head and d_head are None, set default n_head
        if self.n_head is None and self.d_head is None:
            self.n_head = 12  # Default value
            self.d_head = self.n_embd // self.n_head
        # If d_head is provided but n_head is not, calculate n_head
        elif self.d_head is not None and self.n_head is None:
            if self.n_embd % self.d_head != 0:
                raise ValueError(
                    f"n_embd/d_model ({self.n_embd}) must be divisible by d_head ({self.d_head})"
                )
            self.n_head = self.n_embd // self.d_head
        # If n_head is provided but d_head is not, calculate d_head
        elif self.n_head is not None and self.d_head is None:
            if self.n_embd % self.n_head != 0:
                raise ValueError(
                    f"n_embd/d_model ({self.n_embd}) must be divisible by n_head ({self.n_head})"
                )
            self.d_head = self.n_embd // self.n_head
        # If both are provided, validate consistency
        else:
            if self.n_head * self.d_head != self.n_embd:
                raise ValueError(
                    f"n_head ({self.n_head}) * d_head ({self.d_head}) must equal n_embd/d_model ({self.n_embd})"
                )


class OptimizerConfig(BaseModel):
    """Configuration for optimizer and training."""

    learning_rate: float = Field(default=6e-4, gt=0.0, description="Maximum learning rate")
    max_iters: int = Field(default=600000, ge=1, description="Total training iterations")
    weight_decay: float = Field(default=1e-1, ge=0.0, description="Weight decay")
    beta1: float = Field(default=0.9, ge=0.0, le=1.0, description="Adam beta1")
    beta2: float = Field(default=0.95, ge=0.0, le=1.0, description="Adam beta2")
    grad_clip: float = Field(default=1.0, ge=0.0, description="Gradient clipping (0 = disabled)")
    gradient_accumulation_steps: int = Field(
        default=40, ge=1, description="Gradient accumulation steps"
    )


class LRScheduleConfig(BaseModel):
    """Configuration for learning rate schedule."""

    decay_lr: bool = Field(default=True, description="Enable learning rate decay")
    warmup_iters: int = Field(default=2000, ge=0, description="Warmup iterations")
    lr_decay_iters: int = Field(default=600000, ge=1, description="LR decay iterations")
    min_lr: float = Field(default=6e-5, ge=0.0, description="Minimum learning rate")


class DDPConfig(BaseModel):
    """Configuration for distributed data parallel training."""

    backend: Literal["nccl", "gloo", "mpi"] = Field(default="nccl", description="DDP backend")


class SystemConfig(BaseModel):
    """Configuration for system resources."""

    device: str = Field(
        default="cpu",
        description="Device: 'cpu', 'cuda', 'cuda:0', 'mps', etc.",
    )
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="bfloat16", description="Data type for training"
    )
    compile: bool = Field(default=True, description="Use torch.compile")
    seed: int = Field(default=1337, description="Random seed")

    @field_validator("dtype", mode="before")
    @classmethod
    def auto_select_dtype(cls, v: str) -> str:
        """Auto-select dtype based on hardware if not explicitly set."""
        if v == "auto":
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "bfloat16"
            elif torch.cuda.is_available():
                return "float16"
            return "float32"
        return v


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    io: IOConfig = Field(default_factory=IOConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = Field(default_factory=LRScheduleConfig)
    ddp: DDPConfig = Field(default_factory=DDPConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    def get_data_dir(self) -> Path:
        """Get the data directory path."""
        if self.data.data_dir:
            # Ensure it's a Path object (handles both str and Path)
            return (
                Path(self.data.data_dir)
                if isinstance(self.data.data_dir, str)
                else self.data.data_dir
            )
        return Path("data") / self.data.dataset


class SamplingConfig(BaseModel):
    """Configuration for text generation/sampling."""

    init_from: str = Field(
        default="resume",
        description="Init from: 'resume' or 'gpt2'/'gpt2-medium'/'gpt2-large'/'gpt2-xl'",
    )
    out_dir: Path = Field(default=Path("out"), description="Directory with checkpoint")
    start: Union[str, list[str]] = Field(
        default="\n",
        description="Start prompt(s): single string, list of strings, or 'FILE:path/to/file.txt'",
    )
    num_samples: int = Field(
        default=10, ge=1, description="Number of samples to generate per start sequence"
    )
    max_new_tokens: int = Field(default=500, ge=1, description="Tokens to generate per sample")
    temperature: float = Field(default=0.8, ge=0.0, description="Sampling temperature")
    top_k: int = Field(default=200, ge=1, description="Top-k sampling")
    stop_token_char: Optional[str] = Field(
        default=None, description="Stop generation at this character"
    )
    seed: int = Field(default=1337, description="Random seed")
    device: str = Field(default="cuda", description="Device for inference")
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="bfloat16", description="Data type"
    )
    compile: bool = Field(default=False, description="Use torch.compile")

    @field_validator("out_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert strings to Path objects."""
        if v is None:
            return v
        return Path(v) if isinstance(v, str) else v
