"""
Causal Language Model Loss & Metrics
=====================================
Label-smoothed sparse categorical cross-entropy loss with
padding token masking and explicit perplexity tracking.
"""

import tensorflow as tf


def causal_lm_loss(y_true, y_pred, label_smoothing=0.0, ignore_index=-1):
    """Compute label-smoothed cross-entropy loss for causal language modeling.

    Args:
        y_true: Ground truth token IDs, shape (batch, seq_len). Integer tensor.
        y_pred: Predicted logits, shape (batch, seq_len, vocab_size). Float tensor.
        label_smoothing: Label smoothing factor in [0, 1). 0 = no smoothing.
        ignore_index: Token ID to ignore in loss computation (e.g., padding).
                      Set to -1 to disable.

    Returns:
        Scalar loss value (mean over non-masked tokens).
    """
    vocab_size = tf.shape(y_pred)[-1]

    # Build mask for valid tokens
    if ignore_index >= 0:
        mask = tf.cast(tf.not_equal(y_true, ignore_index), tf.float32)
    else:
        mask = tf.ones_like(y_true, dtype=tf.float32)

    if label_smoothing > 0.0:
        # Smooth labels: (1 - smoothing) on correct class, smoothing / (V-1) elsewhere
        y_true_one_hot = tf.one_hot(y_true, depth=vocab_size)
        y_true_smooth = y_true_one_hot * (1.0 - label_smoothing) + \
                        (label_smoothing / tf.cast(vocab_size - 1, tf.float32)) * (1.0 - y_true_one_hot)
        # Compute cross-entropy manually
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        loss_per_token = -tf.reduce_sum(y_true_smooth * log_probs, axis=-1)
    else:
        # Standard sparse cross-entropy
        loss_per_token = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )

    # Apply mask and compute mean
    masked_loss = loss_per_token * mask
    total_loss = tf.reduce_sum(masked_loss)
    num_tokens = tf.maximum(tf.reduce_sum(mask), 1.0)

    return total_loss / num_tokens


class PerplexityMetric(tf.keras.metrics.Metric):
    """Tracks perplexity as exp(mean_loss) over batches."""

    def __init__(self, name="perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, loss_value, sample_weight=None):
        self.total_loss.assign_add(tf.cast(loss_value, tf.float32))
        self.count.assign_add(1.0)

    def result(self):
        mean_loss = self.total_loss / tf.maximum(self.count, 1.0)
        return tf.exp(mean_loss)

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)
