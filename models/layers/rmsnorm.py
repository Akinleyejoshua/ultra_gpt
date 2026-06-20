"""
RMSNorm — Root Mean Square Layer Normalization
===============================================
Scale-only normalization without mean subtraction or bias.
Used in LLaMA, Mistral, and other modern LLMs as a faster
alternative to standard LayerNorm.

Reference: Zhang & Sennrich (2019) — "Root Mean Square Layer Normalization"
"""

import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    """Root Mean Square Normalization.

    Computes:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    Unlike LayerNorm, RMSNorm does not subtract the mean or learn a bias,
    making it faster and more parameter-efficient.

    Args:
        dim: Dimensionality of the input (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (..., dim)
        # Compute in float32 for stability, cast back to input dtype
        x_float = tf.cast(x, tf.float32)
        variance = tf.math.reduce_mean(tf.math.square(x_float), axis=-1, keepdims=True)
        x_normed = x_float * tf.math.rsqrt(variance + self.eps)
        # Cast back and scale
        return tf.cast(x_normed, x.dtype) * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "eps": self.eps})
        return config
