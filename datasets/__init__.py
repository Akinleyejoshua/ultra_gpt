"""
UltraGPT Data Pipeline
======================
High-performance tf.data.Dataset pipelines for causal LM training.
"""

from .pipeline import (
    TiktokenWrapper,
    create_dataset_from_text,
    create_dataset_from_generator,
    create_dataset_from_hf,
    write_tfrecords,
    create_dataset_from_tfrecords,
)
