"""
UltraGPT Transformer Model
===========================
Decoder-only Transformer assembling RMSNorm, GQA, and SwiGLU
into a modern LLM architecture with Pre-LN residual connections,
optional weight tying, and a custom training step.
"""

import tensorflow as tf
from config import UltraGPTConfig
from models.layers.rmsnorm import RMSNorm
from models.layers.attention import GroupedQueryAttention
from models.layers.swiglu import SwiGLU
from models.loss import causal_lm_loss, PerplexityMetric


class DecoderBlock(tf.keras.layers.Layer):
    """Single Transformer decoder block (Pre-LN architecture).

    Architecture:
        x → RMSNorm → GQA(+causal_mask, +kv_cache) + residual
          → RMSNorm → SwiGLU + residual

    Args:
        config: UltraGPTConfig instance.
    """

    def __init__(self, config: UltraGPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        cfg = self.config
        self.attn_norm = RMSNorm(dim=cfg.d_model, eps=cfg.norm_eps, name="attn_norm")
        self.attention = GroupedQueryAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.block_size,
            rope_theta=cfg.rope_theta,
            dropout_rate=cfg.dropout_rate,
            name="gqa",
        )
        self.ffn_norm = RMSNorm(dim=cfg.d_model, eps=cfg.norm_eps, name="ffn_norm")
        self.ffn = SwiGLU(
            d_model=cfg.d_model,
            d_ffn=cfg.d_ffn,
            dropout_rate=cfg.dropout_rate,
            name="swiglu",
        )
        super().build(input_shape)

    def call(self, x, mask=None, cache=None, training=False):
        """Forward pass through one decoder block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Causal attention mask.
            cache: Optional KV-cache tuple from previous generation step.
            training: Boolean flag.

        Returns:
            output: Shape (batch, seq_len, d_model).
            new_cache: Updated KV-cache tuple.
        """
        # ── Self-Attention with Pre-LN ────────────────────────────────
        residual = x
        x_normed = self.attn_norm(x)
        attn_out, new_cache = self.attention(
            x_normed, mask=mask, cache=cache, training=training
        )
        x = residual + attn_out

        # ── Feed-Forward with Pre-LN ─────────────────────────────────
        residual = x
        x_normed = self.ffn_norm(x)
        ffn_out = self.ffn(x_normed, training=training)
        x = residual + ffn_out

        return x, new_cache

    def get_config(self):
        config = super().get_config()
        # Serialize the UltraGPTConfig as a dict
        config["config"] = self.config.__dict__
        return config


class UltraGPT(tf.keras.Model):
    """UltraGPT: Production-grade decoder-only Transformer.

    Architecture mirrors modern LLMs (LLaMA 3, Mistral, Claude 3):
      - RoPE for positional encoding
      - Grouped-Query Attention (GQA)
      - Pre-LN with RMSNorm
      - SwiGLU feed-forward
      - Optional weight tying (embedding ↔ LM head)

    Args:
        config: UltraGPTConfig instance.
    """

    def __init__(self, config: UltraGPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # ── Token Embedding ───────────────────────────────────────────
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            name="token_embedding",
        )

        # ── Decoder Stack ─────────────────────────────────────────────
        self.blocks = [
            DecoderBlock(config, name=f"block_{i}")
            for i in range(config.n_layers)
        ]

        # ── Final Normalization ───────────────────────────────────────
        self.final_norm = RMSNorm(
            dim=config.d_model, eps=config.norm_eps, name="final_norm"
        )

        # ── LM Head ──────────────────────────────────────────────────
        if not config.tie_weights:
            self.lm_head_weight = self.add_weight(
                name="lm_head_weight",
                shape=(config.d_model, config.vocab_size),
                initializer="glorot_uniform",
                trainable=True,
            )

        # ── Metrics ──────────────────────────────────────────────────
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.perplexity_tracker = PerplexityMetric(name="perplexity")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    def _get_causal_mask(self, seq_len):
        """Build a lower-triangular causal mask.

        Returns:
            Mask of shape (1, 1, seq_len, seq_len) where masked
            positions are -1e9 (large negative) and valid positions are 0.
        """
        # Lower triangular: 1 where attention is allowed
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
        )
        # Convert: 0 → -1e9, 1 → 0
        causal_mask = (1.0 - mask) * -1e9
        return causal_mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, S, S)

    def _get_causal_mask_with_cache(self, seq_len, total_len):
        """Build a causal mask for generation with KV-cache.

        During cached generation, seq_len is typically 1 (current token)
        and total_len is the full sequence length including cache.

        Returns:
            Mask of shape (1, 1, seq_len, total_len).
        """
        # For autoregressive generation with cache, the new token can
        # attend to all previous tokens + itself
        mask = tf.zeros((1, 1, seq_len, total_len), dtype=tf.float32)
        return mask

    def call(self, input_ids, cache_list=None, training=False):
        """Forward pass.

        Args:
            input_ids: Integer token IDs, shape (batch, seq_len).
            cache_list: Optional list of KV-cache tuples, one per layer.
                        None during training, populated during generation.
            training: Boolean flag.

        Returns:
            logits: Shape (batch, seq_len, vocab_size).
            new_cache_list: List of updated KV-cache tuples per layer.
        """
        seq_len = tf.shape(input_ids)[1]

        # ── Embed tokens ─────────────────────────────────────────────
        x = self.token_embedding(input_ids)  # (B, S, d_model)

        # Scale embeddings (common in some architectures)
        # x = x * tf.math.sqrt(tf.cast(self.config.d_model, x.dtype))

        # ── Build causal mask ─────────────────────────────────────────
        if cache_list is not None and cache_list[0] is not None:
            cached_len = tf.shape(cache_list[0][0])[2]
            total_len = cached_len + seq_len
            mask = self._get_causal_mask_with_cache(seq_len, total_len)
        else:
            mask = self._get_causal_mask(seq_len)

        # ── Pass through decoder blocks ──────────────────────────────
        new_cache_list = []
        for i, block in enumerate(self.blocks):
            layer_cache = None
            if cache_list is not None and cache_list[i] is not None:
                layer_cache = cache_list[i]
            x, new_cache = block(x, mask=mask, cache=layer_cache, training=training)
            new_cache_list.append(new_cache)

        # ── Final norm ───────────────────────────────────────────────
        x = self.final_norm(x)  # (B, S, d_model)

        # ── LM Head (project to vocabulary) ──────────────────────────
        if self.config.tie_weights:
            # Shared weight: use embedding matrix transposed
            embed_weights = self.token_embedding.embeddings  # (vocab, d_model)
            logits = tf.linalg.matmul(x, embed_weights, transpose_b=True)
        else:
            logits = tf.linalg.matmul(x, self.lm_head_weight)

        return logits, new_cache_list

    def train_step(self, data):
        """Custom training step with label-smoothed causal LM loss.

        Args:
            data: Tuple of (inputs, targets), each shape (batch, seq_len).
        """
        inputs, targets = data

        with tf.GradientTape() as tape:
            logits, _ = self(inputs, training=True)
            loss = causal_lm_loss(
                y_true=targets,
                y_pred=logits,
                label_smoothing=self.config.label_smoothing,
            )
            # Scale loss for mixed precision
            if self.config.mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Compute and apply gradients
        if self.config.mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_variables)

        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.config.grad_clip_norm
            )

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(loss)
        self.accuracy_tracker.update_state(targets, logits)

        return {
            "loss": self.loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data):
        """Custom validation step."""
        inputs, targets = data
        logits, _ = self(inputs, training=False)
        loss = causal_lm_loss(
            y_true=targets,
            y_pred=logits,
            label_smoothing=0.0,  # No smoothing during eval
        )
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(loss)
        self.accuracy_tracker.update_state(targets, logits)
        return {
            "loss": self.loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.perplexity_tracker, self.accuracy_tracker]
