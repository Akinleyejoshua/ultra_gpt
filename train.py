"""
UltraGPT Training Script
=========================
Production training loop with:
  - AdamW optimizer with cosine decay + linear warmup
  - Gradient clipping (global norm)
  - Mixed precision training (float16 on GPU)
  - Multi-GPU support via MirroredStrategy
  - Checkpointing, TensorBoard logging, CSV logging
  - Support for text file, TFRecord, and HuggingFace data sources

Usage:
    python train.py                          # Train on datasets/dataset.txt
    python train.py --source hf              # Train on HuggingFace dataset
    python train.py --source tfrecord        # Train on pre-tokenized TFRecords
    python train.py --preset small           # Use 125M parameter config
"""

import os
import sys
import argparse
import tensorflow as tf

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UltraGPTConfig, toy_config, small_config, medium_config
from models.transformer import UltraGPT
from data_pipeline.pipeline import (
    create_dataset_from_text,
    create_dataset_from_tfrecords,
    create_dataset_from_hf,
    TiktokenWrapper,
)


# ═══════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to 10% of peak LR.

    Args:
        peak_lr: Maximum learning rate after warmup.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak (default 0.1).
    """

    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = peak_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.peak_lr * (step / tf.maximum(warmup, 1.0))

        # Cosine decay
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(tf.constant(3.14159265) * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


# ═══════════════════════════════════════════════════════════════════════
# Per-Step Metrics Callback
# ═══════════════════════════════════════════════════════════════════════

class StepMetricsCallback(tf.keras.callbacks.Callback):
    """Records per-step metric values for plotting training curves.

    Keras 3 traces train_step as a graph, so float() cannot be called
    on symbolic tensors inside train_step. This callback receives eager
    values in on_train_batch_end, making it the correct place to record
    per-step history.

    Usage:
        step_metrics = StepMetricsCallback()
        model.fit(..., callbacks=[step_metrics])
        # Then plot: step_metrics.history["loss"], etc.
    """

    def __init__(self):
        super().__init__()
        self.history = {"loss": [], "perplexity": [], "accuracy": []}

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
        self.history["loss"].append(float(logs.get("loss", 0)))
        self.history["perplexity"].append(float(logs.get("perplexity", 0)))
        self.history["accuracy"].append(float(logs.get("accuracy", 0)))


# ═══════════════════════════════════════════════════════════════════════
# Training Setup
# ═══════════════════════════════════════════════════════════════════════

def build_model_and_optimizer(config: UltraGPTConfig, strategy=None):
    """Build model and optimizer, optionally within a distribution strategy.

    Args:
        config: UltraGPTConfig instance.
        strategy: Optional tf.distribute.Strategy.

    Returns:
        Tuple of (model, optimizer).
    """
    lr_schedule = WarmupCosineDecay(
        peak_lr=config.learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    def _build():
        model = UltraGPT(config, name="ultra_gpt")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            beta_1=0.9,
            beta_2=0.95,
            epsilon=1e-8,
            clipnorm=config.grad_clip_norm if config.grad_clip_norm > 0 else None,
        )
        if config.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        return model, optimizer

    if strategy is not None:
        with strategy.scope():
            model, optimizer = _build()
    else:
        model, optimizer = _build()

    return model, optimizer


def get_callbacks(config: UltraGPTConfig, output_dir: str):
    """Build training callbacks.

    Args:
        config: UltraGPTConfig instance.
        output_dir: Directory for checkpoints and logs.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    callbacks = []

    # Model checkpointing
    ckpt_path = os.path.join(output_dir, "checkpoints", "ultra_gpt_latest.weights.h5")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
    ))

    # TensorBoard
    tb_dir = os.path.join(output_dir, "tensorboard")
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        update_freq=100,
        profile_batch=(10, 20),  # Profile steps 10-20 for performance analysis
    ))

    # CSV Logger
    csv_path = os.path.join(output_dir, "training_log.csv")
    callbacks.append(tf.keras.callbacks.CSVLogger(csv_path))

    # Early stopping on loss plateau (optional, generous patience)
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ))

    return callbacks


def load_data(config: UltraGPTConfig, source: str, **kwargs):
    """Load training and validation data from the specified source.

    Args:
        config: UltraGPTConfig instance.
        source: One of "text", "tfrecord", "hf".
        **kwargs: Additional arguments for the data loader.

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer).
    """
    if source == "text":
        text_path = kwargs.get("text_path", "data_pipeline/dataset.txt")
        return create_dataset_from_text(
            text_path=text_path,
            block_size=config.block_size,
            batch_size=config.batch_size,
            shuffle_buffer=config.shuffle_buffer,
        )

    elif source == "tfrecord":
        tfrecord_dir = kwargs.get("tfrecord_dir", "data_pipeline/tfrecords")
        dataset = create_dataset_from_tfrecords(
            tfrecord_dir=tfrecord_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            shuffle_buffer=config.shuffle_buffer,
        )
        return dataset, None, TiktokenWrapper()

    elif source == "hf":
        dataset_name = kwargs.get("dataset_name", "openwebtext")
        dataset_config_name = kwargs.get("dataset_config", None)
        text_column = kwargs.get("text_column", "text")
        hf_streaming = kwargs.get("streaming", True)
        max_samples = kwargs.get("max_samples", None)
        train_ds, tokenizer = create_dataset_from_hf(
            dataset_name=dataset_name,
            block_size=config.block_size,
            batch_size=config.batch_size,
            shuffle_buffer=config.shuffle_buffer,
            text_column=text_column,
            dataset_config=dataset_config_name,
            streaming=hf_streaming,
            max_samples=max_samples,
        )
        return train_ds, None, tokenizer

    else:
        raise ValueError(f"Unknown data source: '{source}'. Use 'text', 'tfrecord', or 'hf'.")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train UltraGPT")
    parser.add_argument("--preset", choices=["toy", "small", "medium"],
                        default="toy", help="Model size preset")
    parser.add_argument("--source", choices=["text", "tfrecord", "hf"],
                        default="text", help="Data source type")
    parser.add_argument("--text-path", default="data_pipeline/dataset.txt",
                        help="Path to raw text file (for --source text)")
    parser.add_argument("--tfrecord-dir", default="data_pipeline/tfrecords",
                        help="TFRecord directory (for --source tfrecord)")
    parser.add_argument("--hf-dataset", default="openwebtext",
                        help="HuggingFace dataset name (for --source hf)")
    parser.add_argument("--hf-config", default=None,
                        help="HuggingFace dataset config/subset")
    parser.add_argument("--hf-text-col", default="text",
                        help="Text column name in HF dataset")
    parser.add_argument("--hf-streaming", action="store_true", default=True,
                        help="Use HF streaming mode")
    parser.add_argument("--hf-max-samples", type=int, default=None,
                        help="Max samples from HF dataset")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Enable multi-GPU with MirroredStrategy")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch, ignoring previous checkpoints")
    args = parser.parse_args()

    # ── Select config preset ──────────────────────────────────────────
    config_map = {"toy": toy_config, "small": small_config, "medium": medium_config}
    config = config_map[args.preset]()
    if args.no_mixed_precision:
        config.mixed_precision = False

    print(config.summary())
    print()

    # ── Mixed precision ──────────────────────────────────────────────
    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[Training] Mixed precision: float16 enabled")

    # ── Distribution strategy ─────────────────────────────────────────
    strategy = None
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[Training] Multi-GPU: {strategy.num_replicas_in_sync} replicas")

    # ── Load data ────────────────────────────────────────────────────
    train_dataset, val_dataset, tokenizer = load_data(
        config, args.source,
        text_path=args.text_path,
        tfrecord_dir=args.tfrecord_dir,
        dataset_name=args.hf_dataset,
        dataset_config=args.hf_config,
        text_column=args.hf_text_col,
        streaming=args.hf_streaming,
        max_samples=args.hf_max_samples,
    )

    # Distribute dataset if multi-GPU
    if strategy is not None:
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        if val_dataset is not None:
            val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # ── Build model ──────────────────────────────────────────────────
    model, optimizer = build_model_and_optimizer(config, strategy)
    model.compile(
        optimizer=optimizer,
        metrics=[model.perplexity_tracker, model.accuracy_tracker]
    )

    # Build model by running a dummy forward pass
    dummy_input = tf.zeros((1, config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    model.summary()

    # Search for and load the latest checkpoint to resume training
    if not args.no_resume:
        checkpoint_file = os.path.join(args.output_dir, "checkpoints", "ultra_gpt_latest.weights.h5")
        if os.path.exists(checkpoint_file):
            print(f"\n[Training] Found checkpoint! Resuming training by loading weights from: {checkpoint_file}")
            model.load_weights(checkpoint_file)
        else:
            print("\n[Training] No previous checkpoints found. Starting training from scratch.")
    else:
        print("\n[Training] Resume disabled via CLI flag (--no-resume). Starting training from scratch.")

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = get_callbacks(config, args.output_dir)

    # ── Train ────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Starting training: {args.preset} preset, {args.source} data")
    print(f"  Max steps: {config.max_steps}, Batch size: {config.batch_size}")
    print(f"{'═' * 60}\n")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=config.max_steps,
        callbacks=callbacks,
    )

    # ── Save final weights ───────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_weights.weights.h5")
    model.save_weights(final_path)
    print(f"\n[Training] Final weights saved to {final_path}")


if __name__ == "__main__":
    main()
