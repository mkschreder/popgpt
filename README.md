
# PopGPT

![PopGPT](assets/popgpt.jpg)

Clean and structured GPT implementation for training and sampling GPT‑style
models with YAML configs and a clean CLI.

![repro124m](assets/gpt2_124M_loss.png)

## install

From this directory:

```sh
pip3 install -e .
```

Recommended dependencies (install as needed, see `requirements.txt`):

- **PyTorch**: see install matrix on `https://pytorch.org`
- **numpy**
- **tiktoken**
- **PyYAML**
- **pydantic**
- **requests** (used by dataset downloader)
- **wandb** (optional)
- **matplotlib** (optional, for loss charts)
- **transformers** (optional, for GPT‑2 init)

Or use the Makefile helper:

```sh
make install
```

## quick start

Train a tiny Shakespeare character‑level model end‑to‑end using built‑in
targets:

```sh
# 1) Generate data
make generate-shakespeare

# 2) Train (auto-scales to all visible GPUs via torchrun)
make train-shakespeare

# 3) Sample
make sample-shakespeare
```

Equivalent direct CLI:

```sh
python3 -m popgpt.cli train --config configs/train_shakespeare_char.yaml
python3 -m popgpt.cli sample --config configs/sample_shakespeare.yaml
```

### CPU-only quick demo (smaller model)

```sh
python3 -m popgpt.cli train \
  --config configs/train_shakespeare_char.yaml \
  --override system.device="cpu" \
  --override system.compile=False \
  --override io.eval_iters=20 \
  --override io.log_interval=1 \
  --override data.block_size=64 \
  --override data.batch_size=12 \
  --override model.n_layer=4 \
  --override model.n_head=4 \
  --override model.n_embd=128 \
  --override optimizer.max_iters=2000 \
  --override lr_schedule.lr_decay_iters=2000 \
  --override model.dropout=0.0

python3 -m popgpt.cli sample \
  --out-dir out-shakespeare-char \
  --start "To be or not to be"
```

On Apple Silicon, prefer `--override system.device="mps"` if supported.

## datasets

Dataset generators live in `data_generators/` and write to
`data/<dataset>`:

- `shakespeare.py`
- `calculator.py`
- `reverser.py`

Make targets:

```sh
make generate-shakespeare
make generate-calculator
make generate-reverser
make generate-all
```

Each script creates `input.txt`, `train.bin`, `val.bin`, and a `meta.pkl`
for character‑level vocab. Training auto‑detects `vocab_size` from
`meta.pkl` when available.

## CLI usage

The CLI is module-based; run as:

```sh
python3 -m popgpt.cli <command> [options]
```

Commands:

- `train`: Train a model
- `sample`: Generate text from a checkpoint or GPT‑2 family
- `eval`: Run evaluation only (no training)

Examples:

```sh
# Train with YAML
python3 -m popgpt.cli train --config configs/train_reverser.yaml

# Train with overrides
python3 -m popgpt.cli train \
  --config configs/train_reverser.yaml \
  --override model.n_layer=6 \
  --override optimizer.learning_rate=1e-3

# Train without a config (use defaults + overrides)
python3 -m popgpt.cli train \
  --override data.dataset="shakespeare_char" \
  --override optimizer.max_iters=5000

# Sample from a trained run
python3 -m popgpt.cli sample \
  --out-dir out-shakespeare-char \
  --start "O Romeo, Romeo!"

# Sample with YAML
python3 -m popgpt.cli sample --config configs/sample_shakespeare.yaml

# Evaluate a trained model (no weight updates)
python3 -m popgpt.cli eval \
  --config configs/train_shakespeare_char.yaml \
  --out-dir out-shakespeare-char
```

Tip: you can pass multiple `--override section.key=value` arguments. Values
are parsed as Python literals when possible (e.g., numbers, booleans).

## configuration

Training config is validated via Pydantic and organized into sections:
`io`, `wandb`, `data`, `model`, `optimizer`, `lr_schedule`, `ddp`, and
`system`.

Example (excerpt from `configs/train_calculator.yaml`):

```yaml
io:
  out_dir: "out-calculator"
  eval_interval: 500
  log_interval: 10
  eval_iters: 200
  always_save_checkpoint: true
  init_from: "resume"   # "scratch", "resume", or "gpt2*"

data:
  dataset: "calculator"
  batch_size: 128
  block_size: 64
  mask_before_token: "="
  mask_per_line: true
  align_to_lines: true

model:
  n_layer: 4
  n_head: 4           # number of attention heads
  d_model: 256        # model dimension (standard transformer terminology)
  d_head: 64          # head dimension (n_head × d_head = d_model)
  dropout: 0.0
  bias: false         # can also use n_embd instead of d_model (backward compatible)

optimizer:
  learning_rate: 5e-4
  max_iters: 10000

lr_schedule:
  decay_lr: true
  warmup_iters: 200
  lr_decay_iters: 10000
  min_lr: 5e-5

system:
  device: "cuda"        # "cpu", "cuda", "cuda:0", "mps", etc.
  dtype: "float32"      # "float32", "bfloat16", "float16"
  compile: false
  seed: 1337
```

**Model architecture**: Modern transformer parameter naming is now supported:

- **Recommended**: Specify all four parameters explicitly (`n_layer`, `n_head`,
  `d_model`, `d_head`) for maximum clarity. The system validates that
  `n_head × d_head = d_model`.
- **Flexible**: You can omit any parameter and it will be auto-calculated:
  - Specify `d_model` and `d_head` → calculates `n_head`
  - Specify `d_model` and `n_head` → calculates `d_head`
  - Use `n_embd` instead of `d_model` (legacy, fully compatible)

All approaches produce identical model architectures when the ratios match.

Note: The parameter count depends heavily on `vocab_size` (auto-detected from
your dataset). A calculator dataset (~100 tokens) will have ~85M parameters
with these settings, while GPT-2's vocabulary (50,304 tokens) yields ~124M.

Sampling config (`configs/sample_*.yaml`) supports:

- `init_from`: `"resume"` or GPT‑2 family (`"gpt2"`, `"gpt2-medium"`, ...)
- `out_dir`: directory containing `ckpt.pt` when resuming
- `start`: prompt string or `FILE:path.txt`
- `num_samples`, `max_new_tokens`, `temperature`, `top_k`
- `stop_token_char`: stop decoding at a character if present

## distributed training

The Makefile auto-detects GPUs and uses `torchrun` when available:

```sh
make train-shakespeare         # all visible GPUs
make train-reverser NGPUS=2    # explicitly pick number of GPUs
make train CONFIG=configs/train_calculator.yaml
```

You can also call `torchrun` directly; the Makefile shows examples for
single‑GPU and multi‑GPU invocations.

## sampling / inference details

```sh
python3 -m popgpt.cli sample \
  --out-dir out-calculator \
  --start "2+4-8*10=" \
  --override temperature=0.1 --override top_k=5
```

Notes:

- If `init_from="resume"`, the sampler loads `ckpt.pt` from `out_dir`.
- If a dataset `meta.pkl` exists, character-level encodings are used;
  otherwise GPT‑2 BPE (`tiktoken`) is used.
- `start` supports `FILE:...` to read a prompt from a file.
- `stop_token_char` can stop generation at a character (if it maps to a
  single token in the current encoding).

## efficiency notes

- PyTorch 2.0+ `torch.compile` can be enabled via `system.compile=true`.
- Mixed precision: set `system.dtype` to `"bfloat16"` or `"float16"`
  where supported for speedups.

## troubleshooting

- Set `--override system.compile=False` if you hit compile issues on your
  platform.
- For DDP, ensure NCCL is available and configured; on networks without
  Infiniband, you may need to export `NCCL_IB_DISABLE=1`.

## acknowledgements

Inspired by the original NanoGPT by Andrej Karpathy. Thanks to the open
source community and GPU providers for making this work accessible.
