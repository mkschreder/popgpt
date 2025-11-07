"""
Full definition of a Whisper-style Speech-to-Text Model.
Encoder-Decoder architecture for audio-to-text translation.
Based on the architecture patterns in model.py and translator.py.
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
class WhisperConfig:
    """Configuration for Whisper speech-to-text model."""

    # Audio parameters
    n_mels: int = 80  # Number of mel filterbank channels
    sample_rate: int = 16000  # Audio sample rate in Hz
    n_fft: int = 400  # FFT window size
    hop_length: int = 160  # Number of samples between successive frames
    n_audio_ctx: int = 1500  # Maximum audio context length (frames after conv)

    # Text parameters
    vocab_size: int = 50304  # Text vocabulary size
    tgt_block_size: int = 448  # Maximum target text sequence length

    # Model parameters
    n_layer: int = 6  # Number of encoder/decoder layers each
    n_head: int = 8  # Number of attention heads
    n_embd: int = 512  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # Use bias in Linear and LayerNorm layers


class MelSpectrogram(nn.Module):
    """Convert raw audio waveform to mel spectrogram."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels

        # Create mel filterbank
        mel_basis = self._create_mel_filterbank(
            config.sample_rate, config.n_fft, config.n_mels
        )
        self.register_buffer("mel_basis", mel_basis)

        # Window function for STFT
        window = torch.hann_window(config.n_fft)
        self.register_buffer("window", window)

    def _create_mel_filterbank(self, sr, n_fft, n_mels, fmin=0.0, fmax=None):
        """Create mel filterbank matrix."""
        if fmax is None:
            fmax = sr / 2.0

        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595.0 * torch.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Create mel points
        min_mel = hz_to_mel(torch.tensor(fmin))
        max_mel = hz_to_mel(torch.tensor(fmax))
        mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Create frequency bins
        freq_bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)

        # Build filterbank
        filterbank = torch.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Rising slope
            rise = (freq_bins - left) / (center - left)
            # Falling slope
            fall = (right - freq_bins) / (right - center)

            # Combine slopes
            filterbank[i] = torch.maximum(
                torch.zeros_like(freq_bins), torch.minimum(rise, fall)
            )

        return filterbank

    def forward(self, audio):
        """
        Convert audio to mel spectrogram.

        Args:
            audio: (B, T_audio) raw audio waveform

        Returns:
            mel_spec: (B, n_mels, T_frames) mel spectrogram
        """
        # Compute STFT
        # audio: (B, T_audio)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        # stft: (B, n_fft//2 + 1, T_frames)

        # Compute magnitude spectrogram
        mag_spec = torch.abs(stft)  # (B, n_fft//2 + 1, T_frames)

        # Apply mel filterbank
        mel_spec = torch.matmul(
            self.mel_basis, mag_spec
        )  # (B, n_mels, T_frames) if mel_basis is (n_mels, n_fft//2 + 1)

        # Log scale (with small epsilon for numerical stability)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))

        return mel_spec


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for downsampling mel spectrogram.
    Uses 1D convolutions with stride for temporal compression.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Two conv layers similar to Whisper
        # First conv: n_mels -> n_embd
        self.conv1 = nn.Conv1d(
            config.n_mels, config.n_embd, kernel_size=3, stride=1, padding=1
        )
        self.gelu1 = nn.GELU()

        # Second conv: n_embd -> n_embd with stride 2 for downsampling
        self.conv2 = nn.Conv1d(
            config.n_embd, config.n_embd, kernel_size=3, stride=2, padding=1
        )
        self.gelu2 = nn.GELU()

    def forward(self, mel_spec):
        """
        Encode mel spectrogram with convolutions.

        Args:
            mel_spec: (B, n_mels, T_frames)

        Returns:
            encoded: (B, n_embd, T_enc) where T_enc ~ T_frames / 2
        """
        x = self.conv1(mel_spec)  # (B, n_embd, T_frames)
        x = self.gelu1(x)

        x = self.conv2(x)  # (B, n_embd, T_enc) where T_enc ~ T_frames / 2
        x = self.gelu2(x)

        return x


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


class AudioEncoder(nn.Module):
    """Audio encoder: mel spectrogram -> convolutions -> transformer blocks."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Mel spectrogram computation
        self.mel_spec = MelSpectrogram(config)

        # Convolutional frontend
        self.conv_encoder = ConvEncoder(config)

        # Position embeddings for audio frames
        self.wpe = nn.Embedding(config.n_audio_ctx, config.n_embd)

        # Transformer blocks
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, audio, mask=None):
        """
        Encode audio to features.

        Args:
            audio: (B, T_audio) raw audio waveform
            mask: optional audio padding mask

        Returns:
            encoder_output: (B, T_enc, n_embd)
        """
        # Convert to mel spectrogram
        mel = self.mel_spec(audio)  # (B, n_mels, T_frames)

        # Apply convolutional encoder
        x = self.conv_encoder(mel)  # (B, n_embd, T_enc)

        # Transpose to (B, T_enc, n_embd)
        x = x.transpose(1, 2)

        b, t, c = x.size()
        assert (
            t <= self.config.n_audio_ctx
        ), f"Audio sequence length {t} exceeds max context {self.config.n_audio_ctx}"

        # Add position embeddings
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = self.drop(x + pos_emb)

        # Apply transformer blocks
        for block in self.h:
            x = block(x, mask)
        x = self.ln_f(x)

        return x


class Decoder(nn.Module):
    """Transformer decoder stack for text generation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
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


class Whisper(nn.Module):
    """Whisper-style Speech-to-Text Model."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.tgt_block_size is not None
        assert config.n_audio_ctx is not None
        self.config = config

        self.encoder = AudioEncoder(config)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, audio, tgt_idx, audio_mask=None, tgt_mask=None, targets=None):
        """
        Forward pass for training.

        Args:
            audio: raw audio waveform (B, T_audio)
            tgt_idx: target token indices (B, T_tgt)
            audio_mask: optional audio padding mask
            tgt_mask: optional target padding mask
            targets: target tokens for loss computation (B, T_tgt)

        Returns:
            logits: (B, T_tgt, vocab_size)
            loss: scalar if targets provided, else None
        """
        # encode audio
        encoder_output = self.encoder(audio, audio_mask)

        # decode text
        decoder_output = self.decoder(tgt_idx, encoder_output, audio_mask)

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
        audio,
        max_new_tokens,
        bos_token,
        eos_token=None,
        temperature=1.0,
        top_k=None,
        audio_mask=None,
    ):
        """
        Generate text from audio.

        Args:
            audio: raw audio waveform (B, T_audio)
            max_new_tokens: maximum number of tokens to generate
            bos_token: beginning of sequence token id for target
            eos_token: optional end of sequence token id to stop generation
            temperature: sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: if set, only sample from top k most likely tokens
            audio_mask: optional audio padding mask

        Returns:
            Generated text token indices (B, T_tgt)
        """
        # encode audio once
        encoder_output = self.encoder(audio, audio_mask)

        # start with BOS token
        b = audio.size(0)
        device = audio.device
        tgt_idx = torch.full((b, 1), bos_token, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at tgt_block_size
            tgt_idx_cond = (
                tgt_idx
                if tgt_idx.size(1) <= self.config.tgt_block_size
                else tgt_idx[:, -self.config.tgt_block_size :]
            )
            # forward the decoder
            decoder_output = self.decoder(tgt_idx_cond, encoder_output, audio_mask)
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
        # Average audio and text context for estimate
        T_avg = (cfg.n_audio_ctx + cfg.tgt_block_size) // 2
        # Approximate FLOPs (this is a rough estimate)
        flops_per_token = 6 * N + 12 * L * H * Q * T_avg
        flops_per_fwdbwd = flops_per_token * T_avg
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
