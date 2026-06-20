"""
SwiGLU Feed-Forward Network
============================
Gated linear unit with SiLU (Swish) activation, used in LLaMA, PaLM,
and most modern LLMs as a replacement for standard ReLU FFN.

Computes:
    SwiGLU(x) = (swish(x @ W_gate)) ⊙ (x @ W_up) @ W_down

The hidden dimension is typically (8/3) * d_model, rounded for HW alignment.

Reference: Shazeer (2020) — "GLU Variants Improve Transformer"
"""

import tensorflow as tf


class SwiGLU(tf.keras.layers.Layer):
    """SwiGLU (Swish-Gated Linear Unit) feed-forward network.

    Args:
        d_model: Model hidden dimension.
        d_ffn: Intermediate FFN dimension. If None, computed as
               int(4 * d_model * 2/3) rounded to nearest 256.
        dropout_rate: Dropout rate on the output projection.
    """

    def __init__(self, d_model: int, d_ffn: int = None,
                 dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        if d_ffn is None:
            raw = int(4.0 * d_model * 2 / 3)
            d_ffn = ((raw + 255) // 256) * 256
        self.d_ffn = d_ffn
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Gate projection: d_model → d_ffn (goes through swish)
        self.w_gate = self.add_weight(
            name="w_gate",
            shape=(self.d_model, self.d_ffn),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Up projection: d_model → d_ffn (element-wise multiplied with gate)
        self.w_up = self.add_weight(
            name="w_up",
            shape=(self.d_model, self.d_ffn),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Down projection: d_ffn → d_model
        self.w_down = self.add_weight(
            name="w_down",
            shape=(self.d_ffn, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.dropout_rate > 0:
            self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=False):
        """Forward pass.

        Args:
            x: Input tensor, shape (..., d_model).
            training: Boolean flag for dropout.

        Returns:
            Output tensor, shape (..., d_model).
        """
        # Gate: swish activation (SiLU)
        gate = tf.nn.silu(tf.linalg.matmul(x, self.w_gate))  # (..., d_ffn)
        # Up projection (no activation)
        up = tf.linalg.matmul(x, self.w_up)  # (..., d_ffn)
        # Element-wise gating
        hidden = gate * up  # (..., d_ffn)
        # Down projection
        output = tf.linalg.matmul(hidden, self.w_down)  # (..., d_model)

        if self.dropout_rate > 0 and training:
            output = self.ffn_dropout(output, training=training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ffn": self.d_ffn,
            "dropout_rate": self.dropout_rate,
        })
        return config
