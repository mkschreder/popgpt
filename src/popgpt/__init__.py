"""PopGPT - A production-ready implementation of GPT for training and inference."""

__version__ = "2.0.0"

from .core import GPT, GPTConfig
from .config import TrainingConfig, SamplingConfig, load_training_config, load_sampling_config
from .training import Trainer
from .sampling import Sampler

__all__ = [
    "GPT",
    "GPTConfig",
    "TrainingConfig",
    "SamplingConfig",
    "load_training_config",
    "load_sampling_config",
    "Trainer",
    "Sampler",
    "__version__",
]
