"""
UltraGPT Configuration
======================
Central configuration for the decoder-only Transformer.
All hyperparameters are defined here so the model can scale
from a toy debug setup to multi-billion parameter configurations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UltraGPTConfig:
    """Configuration for UltraGPT model architecture and training."""

    # ── Model Architecture ──────────────────────────────────────────────
    d_model: int = 128              # Hidden / embedding dimension
    n_heads: int = 4                # Number of query attention heads
    n_kv_heads: int = 2             # Number of key/value heads (GQA)
    n_layers: int = 4               # Number of decoder blocks
    vocab_size: int = 50257         # Tokenizer vocabulary size (GPT-2 BPE)
    block_size: int = 512           # Maximum context / sequence length
    ffn_mult: float = 4.0           # SwiGLU hidden dimension multiplier
    rope_theta: float = 10000.0     # RoPE base frequency
    tie_weights: bool = True        # Share embedding ↔ LM-head weights
    norm_eps: float = 1e-6          # RMSNorm epsilon
    dropout_rate: float = 0.0       # Residual / attention dropout (0 = off)

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    mixed_precision: bool = True    # Use mixed_float16 policy

    # ── Data Pipeline ───────────────────────────────────────────────────
    shuffle_buffer: int = 10000
    prefetch_buffer: int = -1       # -1 = tf.data.AUTOTUNE
    num_parallel_calls: int = -1    # -1 = tf.data.AUTOTUNE

    # ── Inference ───────────────────────────────────────────────────────
    max_gen_length: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # ── Derived ─────────────────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def kv_head_dim(self) -> int:
        """Dimension per KV head (same as head_dim in standard GQA)."""
        return self.head_dim

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per KV head group."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    @property
    def d_ffn(self) -> int:
        """SwiGLU intermediate dimension, rounded to nearest 256 for HW alignment."""
        raw = int(self.ffn_mult * self.d_model * 2 / 3)
        return ((raw + 255) // 256) * 256

    def summary(self) -> str:
        """Human-readable summary of this configuration."""
        params_embed = self.vocab_size * self.d_model
        params_attn = self.n_layers * (
            self.d_model * self.head_dim * self.n_heads        # Q
            + self.d_model * self.head_dim * self.n_kv_heads   # K
            + self.d_model * self.head_dim * self.n_kv_heads   # V
            + self.d_model * self.d_model                      # O
        )
        params_ffn = self.n_layers * (
            self.d_model * self.d_ffn   # W_gate
            + self.d_model * self.d_ffn # W_up
            + self.d_ffn * self.d_model # W_down
        )
        params_norm = self.n_layers * 2 * self.d_model + self.d_model  # per-block + final
        params_head = 0 if self.tie_weights else self.d_model * self.vocab_size
        total = params_embed + params_attn + params_ffn + params_norm + params_head
        return (
            f"UltraGPT Config Summary\n"
            f"{'─' * 45}\n"
            f"  d_model      : {self.d_model}\n"
            f"  n_heads      : {self.n_heads} (Q) / {self.n_kv_heads} (KV)\n"
            f"  n_layers     : {self.n_layers}\n"
            f"  block_size   : {self.block_size}\n"
            f"  d_ffn        : {self.d_ffn}\n"
            f"  vocab_size   : {self.vocab_size}\n"
            f"  tie_weights  : {self.tie_weights}\n"
            f"  Total params : {total:,} (~{total / 1e6:.1f}M)\n"
            f"{'─' * 45}"
        )


# ── Named Presets ───────────────────────────────────────────────────────

def toy_config(**overrides) -> UltraGPTConfig:
    """Tiny model for debugging and smoke-testing."""
    defaults = dict(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=4,
        block_size=512, batch_size=8, learning_rate=1e-3,
        max_steps=500, warmup_steps=50,
    )
    defaults.update(overrides)
    return UltraGPTConfig(**defaults)


def small_config(**overrides) -> UltraGPTConfig:
    """~125M parameter model (GPT-2-small scale)."""
    defaults = dict(
        d_model=768, n_heads=12, n_kv_heads=4, n_layers=12,
        block_size=2048, batch_size=32, learning_rate=3e-4,
        max_steps=10000, warmup_steps=200, label_smoothing=0.1,
    )
    defaults.update(overrides)
    return UltraGPTConfig(**defaults)


def medium_config(**overrides) -> UltraGPTConfig:
    """~1.3B parameter model."""
    defaults = dict(
        d_model=2048, n_heads=32, n_kv_heads=8, n_layers=24,
        block_size=4096, batch_size=16, learning_rate=2e-4,
        max_steps=50000, warmup_steps=1000, label_smoothing=0.1,
        rope_theta=500000.0,
    )
    defaults.update(overrides)
    return UltraGPTConfig(**defaults)
