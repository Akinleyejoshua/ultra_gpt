"""
Grouped-Query Attention (GQA)
=============================
Memory-efficient multi-head attention where multiple query heads share
the same key/value heads. Reduces KV-cache size and memory bandwidth
during inference while maintaining model quality.

Special cases:
  - n_kv_heads == n_heads → standard Multi-Head Attention (MHA)
  - n_kv_heads == 1       → Multi-Query Attention (MQA)

Used by LLaMA 2/3, Mistral, Claude, and Gemma.

Reference: Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints"
"""

import tensorflow as tf
from models.layers.rope import RotaryPositionEmbedding, apply_rope


class GroupedQueryAttention(tf.keras.layers.Layer):
    """Grouped-Query Attention with RoPE and optional KV-cache.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per head (derived from d_model / n_heads if None).
        max_seq_len: Maximum sequence length for RoPE precomputation.
        rope_theta: RoPE base frequency.
        dropout_rate: Attention dropout rate (applied during training only).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int = None,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.n_groups = n_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

    def build(self, input_shape):
        # Query projection: d_model → n_heads * head_dim
        self.wq = self.add_weight(
            name="wq",
            shape=(self.d_model, self.n_heads * self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Key projection: d_model → n_kv_heads * head_dim
        self.wk = self.add_weight(
            name="wk",
            shape=(self.d_model, self.n_kv_heads * self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Value projection: d_model → n_kv_heads * head_dim
        self.wv = self.add_weight(
            name="wv",
            shape=(self.d_model, self.n_kv_heads * self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Output projection: n_heads * head_dim → d_model
        self.wo = self.add_weight(
            name="wo",
            shape=(self.n_heads * self.head_dim, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # RoPE
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=self.rope_theta,
            name="rope",
        )
        self.rope.build(None)

        # Dropout for attention weights
        if self.dropout_rate > 0:
            self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, mask=None, cache=None, training=False):
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Causal attention mask, shape (1, 1, seq_len, total_len)
                  where masked positions are large negative values.
            cache: Optional tuple (cached_k, cached_v) from previous steps.
                   Each has shape (batch, n_kv_heads, cached_len, head_dim).
            training: Boolean flag for dropout.

        Returns:
            output: Tensor of shape (batch, seq_len, d_model).
            new_cache: Updated (k, v) cache tuple.
        """
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # ── Linear projections ────────────────────────────────────────
        q = tf.linalg.matmul(x, self.wq)  # (B, S, n_heads * head_dim)
        k = tf.linalg.matmul(x, self.wk)  # (B, S, n_kv_heads * head_dim)
        v = tf.linalg.matmul(x, self.wv)  # (B, S, n_kv_heads * head_dim)

        # Reshape to multi-head format
        q = tf.reshape(q, (batch, seq_len, self.n_heads, self.head_dim))
        k = tf.reshape(k, (batch, seq_len, self.n_kv_heads, self.head_dim))
        v = tf.reshape(v, (batch, seq_len, self.n_kv_heads, self.head_dim))

        # Transpose to (batch, heads, seq_len, head_dim)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # ── Apply RoPE ────────────────────────────────────────────────
        # Determine position offset from cache
        offset = 0
        if cache is not None:
            offset = tf.shape(cache[0])[2]  # cached sequence length

        cos, sin = self.rope(seq_len, offset=offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── Update KV-Cache ───────────────────────────────────────────
        if cache is not None:
            cached_k, cached_v = cache
            k = tf.concat([cached_k, k], axis=2)
            v = tf.concat([cached_v, v], axis=2)
        new_cache = (k, v)

        # ── Expand KV heads for GQA ──────────────────────────────────
        # Repeat each KV head `n_groups` times to match query heads
        if self.n_groups > 1:
            k = tf.repeat(k, repeats=self.n_groups, axis=1)
            v = tf.repeat(v, repeats=self.n_groups, axis=1)

        # ── Scaled Dot-Product Attention ──────────────────────────────
        # (B, n_heads, S_q, head_dim) @ (B, n_heads, head_dim, S_kv)
        attn_weights = tf.linalg.matmul(q, k, transpose_b=True) * self.scale

        # Apply causal mask (additive, masked positions are -inf)
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if self.dropout_rate > 0 and training:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # (B, n_heads, S_q, head_dim)
        attn_output = tf.linalg.matmul(attn_weights, v)

        # ── Concatenate heads and project output ──────────────────────
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (B, S_q, n_heads, head_dim)
        attn_output = tf.reshape(attn_output, (batch, seq_len, self.n_heads * self.head_dim))
        output = tf.linalg.matmul(attn_output, self.wo)  # (B, S_q, d_model)

        return output, new_cache

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "dropout_rate": self.dropout_rate,
        })
        return config
