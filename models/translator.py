"""
Full definition of an Encoder-Decoder Transformer for Machine Translation.
Based on the architecture in model.py but adapted for sequence-to-sequence tasks.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import reusable components from the base model
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import LayerNorm, MLP


@dataclass
class TranslatorConfig:
    src_vocab_size: int = 50304  # Source vocabulary size
    tgt_vocab_size: int = 50304  # Target vocabulary size
    src_block_size: int = 512  # Maximum source sequence length
    tgt_block_size: int = 512  # Maximum target sequence length
    n_layer: int = 6  # Number of encoder/decoder layers each
    n_head: int = 8  # Number of attention heads
    n_embd: int = 512  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # True: bias in Linears and LayerNorms


class SelfAttention(nn.Module):
    """Bidirectional self-attention for encoder (no causal masking)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def forward(self, x, mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # bidirectional self-attention (no causal masking)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # For bidirectional attention, is_causal=False
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalSelfAttention(nn.Module):
    """Causal self-attention for decoder (with causal masking)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.tgt_block_size, config.tgt_block_size)
                ).view(1, 1, config.tgt_block_size, config.tgt_block_size),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    """Cross-attention for decoder attending to encoder outputs."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query projection from decoder
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # key and value projections from encoder
        self.kv_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()  # decoder input
        B, T_enc, C = encoder_output.size()  # encoder output

        # queries from decoder
        q = self.q_attn(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # keys and values from encoder
        k, v = self.kv_attn(encoder_output).split(self.n_embd, dim=2)
        k = k.view(B, T_enc, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_enc, hs)
        v = v.view(B, T_enc, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T_enc, hs)

        # cross-attention (decoder attends to encoder)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T_enc) x (B, nh, T_enc, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class EncoderBlock(nn.Module):
    """Transformer encoder block with bidirectional self-attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attention and cross-attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, encoder_output, src_mask=None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), encoder_output, src_mask)
        x = x + self.mlp(self.ln_3(x))
        return x


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.src_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.src_block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.src_block_size
        ), f"Cannot forward sequence of length {t}, source block size is only {self.config.src_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the encoder
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x, mask)
        x = self.ln_f(x)

        return x


class Decoder(nn.Module):
    """Transformer decoder stack."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.tgt_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.tgt_block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, idx, encoder_output, src_mask=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.tgt_block_size
        ), f"Cannot forward sequence of length {t}, target block size is only {self.config.tgt_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the decoder
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x, encoder_output, src_mask)
        x = self.ln_f(x)

        return x


class Translator(nn.Module):
    """Encoder-Decoder Transformer for Machine Translation."""

    def __init__(self, config):
        super().__init__()
        assert config.src_vocab_size is not None
        assert config.tgt_vocab_size is not None
        assert config.src_block_size is not None
        assert config.tgt_block_size is not None
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(config.n_embd, config.tgt_vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wpe.weight.numel()
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src_idx, tgt_idx, src_mask=None, tgt_mask=None, targets=None):
        """
        Forward pass for training.

        Args:
            src_idx: source token indices (B, T_src)
            tgt_idx: target token indices (B, T_tgt)
            src_mask: optional source padding mask
            tgt_mask: optional target padding mask
            targets: target tokens for loss computation (B, T_tgt)

        Returns:
            logits: (B, T_tgt, tgt_vocab_size)
            loss: scalar if targets provided, else None
        """
        # encode source sequence
        encoder_output = self.encoder(src_idx, src_mask)

        # decode target sequence
        decoder_output = self.decoder(tgt_idx, encoder_output, src_mask)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(decoder_output)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                decoder_output[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        src_idx,
        max_new_tokens,
        bos_token,
        eos_token=None,
        temperature=1.0,
        top_k=None,
        src_mask=None,
    ):
        """
        Generate target sequence given source sequence.

        Args:
            src_idx: source token indices (B, T_src)
            max_new_tokens: maximum number of tokens to generate
            bos_token: beginning of sequence token id for target
            eos_token: optional end of sequence token id to stop generation
            temperature: sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: if set, only sample from top k most likely tokens
            src_mask: optional source padding mask

        Returns:
            Generated target token indices (B, T_tgt)
        """
        # encode source sequence once
        encoder_output = self.encoder(src_idx, src_mask)

        # start with BOS token
        b = src_idx.size(0)
        device = src_idx.device
        tgt_idx = torch.full((b, 1), bos_token, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at tgt_block_size
            tgt_idx_cond = (
                tgt_idx
                if tgt_idx.size(1) <= self.config.tgt_block_size
                else tgt_idx[:, -self.config.tgt_block_size :]
            )
            # forward the decoder
            decoder_output = self.decoder(tgt_idx_cond, encoder_output, src_mask)
            # get logits for the last position
            logits = self.lm_head(decoder_output[:, -1, :])

            if temperature == 0.0:
                # greedy decoding: just take the most likely token
                tgt_idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                # scale by temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                tgt_idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            tgt_idx = torch.cat((tgt_idx, tgt_idx_next), dim=1)

            # check if we should stop (if all sequences in batch generated the EOS token)
            if eos_token is not None and (tgt_idx_next == eos_token).all():
                break

        return tgt_idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay for 2D parameters (weights) but not 1D (biases, layernorms).
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        This is a rough estimate for encoder-decoder models.
        """
        # Rough estimate: count encoder and decoder params
        N = self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
        # Average block sizes for estimate
        T_avg = (cfg.src_block_size + cfg.tgt_block_size) // 2
        # Approximate FLOPs (this is a rough estimate)
        flops_per_token = 6 * N + 12 * L * H * Q * T_avg
        flops_per_fwdbwd = flops_per_token * T_avg
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
