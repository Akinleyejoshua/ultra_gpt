"""
High-Performance Data Engineering Pipeline
===========================================
tf.data.Dataset pipeline for causal language model training.
Supports raw text ingestion with tiktoken tokenization and
pre-tokenized TFRecord I/O for maximum throughput.

Design goals:
  - Zero CPU bottleneck on GPU/TPU training
  - Streaming-friendly for large corpora
  - Deterministic shuffling with configurable buffer
"""

import os
import numpy as np
import tensorflow as tf


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer Wrapper
# ═══════════════════════════════════════════════════════════════════════

class TiktokenWrapper:
    """Wrapper around tiktoken for consistent tokenization interface.

    Uses GPT-2's r50k_base encoding (50257 tokens) by default.
    """

    def __init__(self, encoding_name: str = "r50k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.enc.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.enc.decode(token_ids)

    @property
    def eos_token_id(self) -> int:
        return self.enc.eot_token


# ═══════════════════════════════════════════════════════════════════════
# Raw Text → tf.data Pipeline
# ═══════════════════════════════════════════════════════════════════════

def create_dataset_from_text(
    text_path: str,
    block_size: int = 512,
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    encoding_name: str = "r50k_base",
    seed: int = 42,
    val_split: float = 0.1,
):
    """Build train and validation tf.data.Datasets from a raw text file with dialogue masking.

    Args:
        text_path: Path to the raw .txt file.
        block_size: Context window size.
        batch_size: Training batch size.
        shuffle_buffer: Size of the shuffle buffer.
        encoding_name: Tiktoken encoding name.
        seed: Random seed for shuffling.
        val_split: Fraction of dataset chunks reserved for validation.

    Returns:
        train_dataset: tf.data.Dataset for training.
        val_dataset: tf.data.Dataset for validation, or None if val_split = 0.0.
        tokenizer: TiktokenWrapper.
    """
    import re
    tokenizer = TiktokenWrapper(encoding_name)

    # ── Read and tokenize ─────────────────────────────────────────────
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    has_chatml = "<|im_start|>" in raw_text

    if has_chatml:
        # Dialogue masking logic
        pattern = r"(<\|im_start\|>user\n|<\|im_start\|>assistant\n|<\|im_end\|>\n?)"
        parts = re.split(pattern, raw_text)
        input_ids = []
        target_ids = []
        current_role = None
        for part in parts:
            if not part:
                continue
            if part == "<|im_start|>user\n":
                current_role = "user"
                tokens = tokenizer.encode(part)
                input_ids.extend(tokens)
                target_ids.extend([-100] * len(tokens))
            elif part == "<|im_start|>assistant\n":
                current_role = "assistant"
                tokens = tokenizer.encode(part)
                input_ids.extend(tokens)
                target_ids.extend([-100] * len(tokens))
            elif part.startswith("<|im_end|>"):
                tokens = tokenizer.encode(part)
                input_ids.extend(tokens)
                if current_role == "assistant":
                    target_ids.extend(tokens)
                else:
                    target_ids.extend([-100] * len(tokens))
                current_role = None
            else:
                tokens = tokenizer.encode(part)
                input_ids.extend(tokens)
                if current_role == "assistant":
                    target_ids.extend(tokens)
                else:
                    target_ids.extend([-100] * len(tokens))
    else:
        # Standard causal language modeling targets (copy input)
        all_tokens = tokenizer.encode(raw_text)
        input_ids = list(all_tokens)
        target_ids = list(all_tokens)

    input_ids = np.array(input_ids, dtype=np.int32)
    target_ids = np.array(target_ids, dtype=np.int32)

    # Shift targets by 1 relative to inputs
    inputs_all = input_ids[:-1]
    targets_all = target_ids[1:]

    # Chunk into block_size windows
    n_chunks = len(inputs_all) // block_size
    # Trim to exact multiple
    inputs_all = inputs_all[: n_chunks * block_size]
    targets_all = targets_all[: n_chunks * block_size]

    inputs_chunks = inputs_all.reshape(n_chunks, block_size)
    targets_chunks = targets_all.reshape(n_chunks, block_size)

    print(f"[Data Pipeline] Dialogue Masking: {has_chatml} | Tokenized {len(raw_text):,} chars → "
          f"{len(input_ids):,} tokens → {n_chunks:,} chunks of {block_size}")

    # Train / Val Split
    n_val = int(n_chunks * val_split)
    n_train = n_chunks - n_val

    train_inputs = inputs_chunks[:n_train]
    train_targets = targets_chunks[:n_train]

    val_inputs = inputs_chunks[n_train:]
    val_targets = targets_chunks[n_train:]

    # ── Build tf.data.Datasets ─────────────────────────────────────────
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    if n_val > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
        val_dataset = (
            val_dataset
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset, tokenizer


def create_dataset_from_generator(
    text_generator,
    block_size: int = 512,
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    encoding_name: str = "r50k_base",
    seed: int = 42,
):
    """Build a streaming tf.data.Dataset from a text generator.

    Useful for web scrapers, API streams, or very large corpora
    that don't fit in memory.

    Args:
        text_generator: Callable that yields text strings.
        block_size: Context window size.
        batch_size: Training batch size.
        shuffle_buffer: Shuffle buffer size.
        encoding_name: Tiktoken encoding name.
        seed: Random seed.

    Returns:
        tf.data.Dataset yielding (inputs, targets) batches.
    """
    tokenizer = TiktokenWrapper(encoding_name)
    chunk_size = block_size + 1

    def token_chunk_generator():
        """Yields fixed-size token chunks from streaming text."""
        buffer = []
        for text in text_generator():
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)
            # Yield complete chunks from the buffer
            while len(buffer) >= chunk_size:
                yield np.array(buffer[:chunk_size], dtype=np.int32)
                buffer = buffer[chunk_size:]

    dataset = tf.data.Dataset.from_generator(
        token_chunk_generator,
        output_signature=tf.TensorSpec(shape=(chunk_size,), dtype=tf.int32),
    )

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = (
        dataset
        .map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# TFRecord I/O (Maximum Throughput Path)
# ═══════════════════════════════════════════════════════════════════════

def write_tfrecords(
    text_path: str,
    output_dir: str,
    block_size: int = 512,
    shards: int = 16,
    encoding_name: str = "r50k_base",
):
    """Pre-tokenize a text file and write to sharded TFRecords.

    This is the highest-throughput path for training: tokenization
    happens once offline, and the training loop reads pre-tokenized
    binary records.

    Args:
        text_path: Path to raw .txt file.
        output_dir: Directory to write TFRecord shards.
        block_size: Context window size.
        shards: Number of output shard files.
        encoding_name: Tiktoken encoding name.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = TiktokenWrapper(encoding_name)

    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    all_tokens = tokenizer.encode(raw_text)
    all_tokens = np.array(all_tokens, dtype=np.int32)

    chunk_size = block_size + 1
    n_chunks = len(all_tokens) // chunk_size
    all_tokens = all_tokens[: n_chunks * chunk_size]
    chunks = all_tokens.reshape(n_chunks, chunk_size)

    # Shuffle chunks before writing
    rng = np.random.default_rng(42)
    rng.shuffle(chunks)

    # Write to sharded TFRecords
    chunks_per_shard = (n_chunks + shards - 1) // shards
    for shard_idx in range(shards):
        shard_path = os.path.join(output_dir, f"train_{shard_idx:05d}.tfrecord")
        start = shard_idx * chunks_per_shard
        end = min(start + chunks_per_shard, n_chunks)
        with tf.io.TFRecordWriter(shard_path) as writer:
            for i in range(start, end):
                feature = {
                    "tokens": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=chunks[i].tolist())
                    )
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

    print(f"[TFRecord] Wrote {n_chunks:,} chunks to {shards} shards in {output_dir}")


def create_dataset_from_tfrecords(
    tfrecord_dir: str,
    block_size: int = 512,
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    seed: int = 42,
):
    """Build a tf.data.Dataset from pre-tokenized TFRecords.

    This is the fastest training data path — no tokenization overhead.

    Args:
        tfrecord_dir: Directory containing .tfrecord shard files.
        block_size: Context window size (must match what was used to write).
        batch_size: Training batch size.
        shuffle_buffer: Shuffle buffer size.
        seed: Random seed.

    Returns:
        tf.data.Dataset yielding (inputs, targets) batches.
    """
    chunk_size = block_size + 1

    # Discover all shard files
    shard_files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    assert len(shard_files) > 0, f"No .tfrecord files found in {tfrecord_dir}"

    # Parse function
    feature_description = {
        "tokens": tf.io.FixedLenFeature([chunk_size], tf.int64),
    }

    def parse_example(serialized):
        parsed = tf.io.parse_single_example(serialized, feature_description)
        tokens = tf.cast(parsed["tokens"], tf.int32)
        inputs = tokens[:-1]
        targets = tokens[1:]
        return inputs, targets

    # Interleave shard files for parallel I/O
    files_dataset = tf.data.Dataset.from_tensor_slices(shard_files)
    files_dataset = files_dataset.shuffle(len(shard_files), seed=seed)

    dataset = files_dataset.interleave(
        lambda path: tf.data.TFRecordDataset(path),
        cycle_length=min(8, len(shard_files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    dataset = (
        dataset
        .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# HuggingFace Datasets Integration
# ═══════════════════════════════════════════════════════════════════════

def create_dataset_from_hf(
    dataset_name: str,
    block_size: int = 512,
    batch_size: int = 32,
    shuffle_buffer: int = 10000,
    encoding_name: str = "r50k_base",
    text_column: str = "text",
    split: str = "train",
    dataset_config: str = None,
    streaming: bool = True,
    max_samples: int = None,
    seed: int = 42,
):
    """Build a tf.data.Dataset from a HuggingFace Hub dataset.

    Supports both streaming mode (for massive datasets like The Pile,
    RedPajama, FineWeb) and in-memory mode for smaller datasets.

    Examples:
        # Stream OpenWebText (large corpus)
        ds, tok = create_dataset_from_hf("openwebtext", streaming=True)

        # Load WikiText-103 fully
        ds, tok = create_dataset_from_hf(
            "wikitext", dataset_config="wikitext-103-raw-v1",
            streaming=False, text_column="text"
        )

        # Stream a specific split with sample limit
        ds, tok = create_dataset_from_hf(
            "allenai/c4", dataset_config="en", split="train",
            streaming=True, max_samples=100000
        )

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "openwebtext",
                      "wikitext", "allenai/c4", "EleutherAI/the_pile").
        block_size: Context window size (number of input tokens per sample).
        batch_size: Training batch size.
        shuffle_buffer: Size of the shuffle buffer.
        encoding_name: Tiktoken encoding name.
        text_column: Name of the text field in the dataset. Common values:
                     "text", "content", "document". Auto-detected if possible.
        split: Dataset split to use (e.g., "train", "validation", "test").
        dataset_config: Dataset configuration/subset name (e.g., "wikitext-103-raw-v1",
                        "en" for C4). None for datasets without configs.
        streaming: If True, uses HF streaming to avoid downloading the full
                   dataset to disk. Essential for large corpora (100GB+).
        max_samples: Maximum number of raw text samples to process.
                     None = use entire dataset.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (tf.data.Dataset, TiktokenWrapper).
        Dataset yields (inputs, targets) batches of shape (batch_size, block_size).
    """
    from datasets import load_dataset

    tokenizer = TiktokenWrapper(encoding_name)
    chunk_size = block_size + 1

    # ── Load HuggingFace dataset ──────────────────────────────────────
    load_kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": streaming,
    }
    if dataset_config is not None:
        load_kwargs["name"] = dataset_config

    hf_dataset = load_dataset(**load_kwargs)

    # ── Auto-detect text column if needed ─────────────────────────────
    if not streaming:
        available_columns = hf_dataset.column_names
        if text_column not in available_columns:
            # Try common text column names
            candidates = ["text", "content", "document", "sentence", "passage"]
            detected = None
            for col in candidates:
                if col in available_columns:
                    detected = col
                    break
            if detected is None:
                raise ValueError(
                    f"Text column '{text_column}' not found. "
                    f"Available columns: {available_columns}. "
                    f"Specify the correct column via `text_column=`."
                )
            text_column = detected
            print(f"[HF Pipeline] Auto-detected text column: '{text_column}'")

    # ── Streaming path ────────────────────────────────────────────────
    if streaming:
        def hf_chunk_generator():
            """Stream HF dataset, tokenize, and yield fixed-size chunks."""
            buffer = []
            sample_count = 0
            for sample in hf_dataset:
                text = sample.get(text_column, "")
                if not text or not text.strip():
                    continue
                tokens = tokenizer.encode(text)
                buffer.extend(tokens)
                # Yield complete chunks
                while len(buffer) >= chunk_size:
                    yield np.array(buffer[:chunk_size], dtype=np.int32)
                    buffer = buffer[chunk_size:]
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break

        dataset = tf.data.Dataset.from_generator(
            hf_chunk_generator,
            output_signature=tf.TensorSpec(shape=(chunk_size,), dtype=tf.int32),
        )

    # ── In-memory path (smaller datasets) ─────────────────────────────
    else:
        print(f"[HF Pipeline] Loading '{dataset_name}' ({split}) into memory...")
        all_tokens = []
        sample_count = 0
        for sample in hf_dataset:
            text = sample[text_column]
            if not text or not text.strip():
                continue
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

        all_tokens = np.array(all_tokens, dtype=np.int32)
        n_chunks = len(all_tokens) // chunk_size
        all_tokens = all_tokens[: n_chunks * chunk_size]
        chunks = all_tokens.reshape(n_chunks, chunk_size)

        print(f"[HF Pipeline] {sample_count:,} samples → "
              f"{len(all_tokens):,} tokens → {n_chunks:,} chunks")

        dataset = tf.data.Dataset.from_tensor_slices(chunks)

    # ── Common pipeline tail ──────────────────────────────────────────
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = (
        dataset
        .map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer, seed=seed)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, tokenizer
