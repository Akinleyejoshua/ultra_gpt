"""
Rotary Position Embeddings (RoPE)
=================================
Applies rotation to pairs of dimensions in Q and K tensors to inject
relative positional information without learnable parameters.

Used by LLaMA, Mistral, and most modern open-weight LLMs.

Reference: Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import tensorflow as tf
import numpy as np


class RotaryPositionEmbedding(tf.keras.layers.Layer):
    """Precomputes and caches RoPE sin/cos frequency tables.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency (10000 for standard, 500000 for long-context).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192,
                 theta: float = 10000.0, **kwargs):
        super().__init__(**kwargs)
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

    def build(self, input_shape):
        # Precompute the inverse frequency vector: shape (head_dim // 2,)
        dim_pairs = self.head_dim // 2
        inv_freq = 1.0 / (
            self.theta ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim)
        )  # (dim_pairs,)

        # Outer product with position indices: (max_seq_len, dim_pairs)
        positions = np.arange(self.max_seq_len, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # (max_seq_len, dim_pairs)

        # Store cos and sin tables as non-trainable weights
        self.cos_cached = self.add_weight(
            name="cos_cached",
            shape=freqs.shape,
            initializer=tf.keras.initializers.Constant(np.cos(freqs)),
            trainable=False,
            dtype=tf.float32,
        )
        self.sin_cached = self.add_weight(
            name="sin_cached",
            shape=freqs.shape,
            initializer=tf.keras.initializers.Constant(np.sin(freqs)),
            trainable=False,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, seq_len, offset=0):
        """Return (cos, sin) slices for positions [offset, offset + seq_len).

        Args:
            seq_len: Number of positions to retrieve.
            offset: Starting position index (used during KV-cache inference).

        Returns:
            Tuple of (cos, sin), each of shape (seq_len, head_dim // 2).
        """
        cos = self.cos_cached[offset : offset + seq_len]  # (seq_len, dim_pairs)
        sin = self.sin_cached[offset : offset + seq_len]
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "theta": self.theta,
        })
        return config


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to a tensor.

    Rotates pairs of dimensions: (x0, x1) → (x0·cos − x1·sin, x0·sin + x1·cos)

    Args:
        x: Tensor of shape (batch, n_heads, seq_len, head_dim).
        cos: Cosine frequencies, shape (seq_len, head_dim // 2).
        sin: Sine frequencies, shape (seq_len, head_dim // 2).

    Returns:
        Rotated tensor, same shape as x.
    """
    # Split head_dim into pairs
    x_float = tf.cast(x, tf.float32)
    cos_float = tf.cast(cos, tf.float32)
    sin_float = tf.cast(sin, tf.float32)
    d = tf.shape(x_float)[-1]
    half_d = d // 2

    x0 = x_float[..., :half_d]  # (batch, heads, seq, half_d)
    x1 = x_float[..., half_d:]

    # Broadcast cos/sin: (seq, half_d) → (1, 1, seq, half_d)
    cos_broadcast = cos_float[tf.newaxis, tf.newaxis, :, :]
    sin_broadcast = sin_float[tf.newaxis, tf.newaxis, :, :]

    # Apply rotation
    rotated = tf.concat([
        x0 * cos_broadcast - x1 * sin_broadcast,
        x0 * sin_broadcast + x1 * cos_broadcast,
    ], axis=-1)

    return tf.cast(rotated, x.dtype)
