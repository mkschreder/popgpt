# Calculator fine-tune configuration parameters

This document explains all parameters defined in
`applications/popgpt/config/finetune_calculator.py`, how they relate to the
calculator dataset format (`{expression}={result}\n` per line), and how they
affect model capabilities.

## I/O, evaluation, and logging

- **out_dir**: Directory where checkpoints and artifacts are saved.
- **eval_interval**: Iterations between evaluations. Frequent eval helps track
  overfitting on the small dataset.
- **eval_iters**: Number of batches used to compute eval loss for stability.
- **log_interval**: Iterations between training log prints.
- **always_save_checkpoint**: If true, saves a checkpoint every eval regardless
  of validation improvement. Here it is false to avoid noisy saves while
  overfitting.
- **init_from**: How to initialize training: `scratch` (new), `resume` (from
  `out_dir/ckpt.pt`). Resuming preserves the learned calculator ability.
- **wandb_log / wandb_project / wandb_run_name**: Controls Weights & Biases
  logging. Disabled by default in this config.

## Dataset and sampling

- **dataset**: Dataset name under `data/`. Here, `calculator` expects lines of
  the form `12.5+7.3=19.8\n`.
- **block_size**: Maximum context length in tokens for both inputs and targets.
  Must be large enough for the longest expression, the `=` sign, the result,
  and the trailing newline. Too small truncates lines; larger enables longer
  expressions but increases memory and compute.
- **align_to_lines**: When true, training samples always start at the beginning
  of a line and include only complete lines that fit within `block_size`.
  This avoids cutting expressions mid-line, improving learning signal and
  stability for per-line datasets.
- **mask_before_token**: Character up to and including which targets are masked
  (set to -1) for loss. With `'='`, the model sees the full input expression
  but only incurs loss on result tokens. This focuses capacity on performing
  the calculation rather than echoing the prompt.
- **mask_per_line**: When true, masking is applied independently per line in a
  batch, resetting after each newline. This matches the one-example-per-line
  dataset structure and prevents masking from spilling across lines.

Effect on capabilities: the trio `align_to_lines`, `mask_before_token='='`, and
`mask_per_line=True` makes training target only the result portion for each
complete expression. This dramatically improves precision of result generation
and encourages deterministic computation skills over generic language modeling.

## Batch and accumulation

- **batch_size**: Number of sequences per optimization micro-step.
- **gradient_accumulation_steps**: Number of micro-steps whose gradients are
  accumulated before a single optimizer step. Effective batch size is
  `batch_size × gradient_accumulation_steps`. Larger effective batches generally
  yield smoother gradients and more stable learning at the cost of memory/time.

Relation to data: combined with `block_size`, these determine tokens processed
per iteration and the ratio of expressions to results seen by the model.

## Model capacity and regularization

- **n_layer**: Number of Transformer blocks. More layers increase depth and
  capacity to model complex operations, but cost more compute and risk
  overfitting on small datasets.
- **d_model**: Model dimension (embedding/hidden dimension). This is the
  standard transformer terminology. Higher width increases representational
  capacity and parameter count; on tiny datasets it may overfit. (Legacy name:
  **n_embd** - both work identically.)
- **d_head**: Dimension per attention head. This controls the representational
  capacity of each head. The number of heads is automatically calculated as
  `n_head = d_model / d_head`. Larger head dimensions allow each head to
  capture more complex patterns. (Alternatively, you can specify **n_head**
  directly, and `d_head` will be calculated automatically for backward
  compatibility.)
- **dropout**: Drop probability applied in attention and MLP layers. Helps
  regularize and improve generalization; keep moderate for small data.

Effect on capabilities: the trio `n_layer`, `d_model`, and `d_head` primarily
sets the ceiling on what complexity the model can compute and remember within
`block_size`. Dropout trades raw memorization for robustness.

## Optimization and schedule

- **learning_rate**: Base AdamW learning rate. Higher speeds learning but risks
  divergence; smaller is safer but slower. Here a relatively high value works
  with a "baby" model and masked task.
- **warmup_iters**: Linear warmup steps to reach the peak LR, preventing early
  instability.
- **lr_decay_iters**: Iteration at which cosine decay reaches `min_lr`. Often
  set to `max_iters` to decay over the full run.
- **min_lr**: Floor LR at the end of cosine schedule, typically LR/10.
- **beta2**: AdamW second-moment decay. A slightly larger value (0.99) smooths
  noisy estimates when tokens per iteration are small.
- **max_iters**: Maximum training iterations (termination condition). Longer
  runs allow further refinement but risk overfitting on tiny datasets.

Relation to data: schedule shape (warmup/decay) and LR smooth the signal from
per-line masked supervision, impacting how quickly the model learns accurate
results and how well it generalizes to unseen expressions.

## System/runtime

- **device**: Compute device, e.g., `cuda` or `cpu`. GPU is recommended.
- **compile**: If true, enable `torch.compile` for speed (PyTorch 2+).
- **dtype**: Compute precision, e.g., `float32`. Lower precision can improve
  throughput; use with care for numerical stability on arithmetic tasks.

## Practical guidance for calculator data

- Ensure each line fully fits within `block_size` so masking and alignment apply
  cleanly. Include the newline terminator in your data.
- Keep `mask_before_token='='`, `mask_per_line=True`, and `align_to_lines=True`
  for this per-line arithmetic format to maximize capability on producing the
  correct result tokens.
