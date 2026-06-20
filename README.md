# ⚡ UltraGPT

A production-grade, decoder-only Transformer (GPT-style) built from scratch with TensorFlow 2.x / Keras — incorporating state-of-the-art architectural innovations from LLaMA 3, Mistral, and Claude 3.

---

## 🏗️ Architecture

| Component | Implementation | Reference |
|---|---|---|
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | Su et al., 2021 |
| **Attention** | Grouped-Query Attention (GQA) | Ainslie et al., 2023 |
| **Normalization** | RMSNorm (Pre-LN) | Zhang & Sennrich, 2019 |
| **Feed-Forward** | SwiGLU (Swish-Gated Linear Unit) | Shazeer, 2020 |
| **LM Head** | Optional weight tying with embedding | Press & Wolf, 2017 |
| **Inference** | KV-Cache + XLA compilation | — |

### Model Presets

| Preset | Parameters | d_model | Heads (Q/KV) | Layers | Context |
|---|---|---|---|---|---|
| `toy` | ~3M | 128 | 4 / 2 | 4 | 512 |
| `small` | ~125M | 768 | 12 / 4 | 12 | 2048 |
| `medium` | ~1.3B | 2048 | 32 / 8 | 24 | 4096 |

---

## 📁 Project Structure

```
ultra_gpt/
├── config.py                  # Hyperparameters & model presets
├── train.py                   # Training entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data_pipeline/
│   ├── pipeline.py            # tf.data pipelines (text, TFRecord, HuggingFace)
│   └── dataset.txt            # Your training data (place here)
├── models/
│   ├── __init__.py
│   ├── transformer.py         # DecoderBlock + UltraGPT model
│   ├── loss.py                # Causal LM loss + perplexity metric
│   └── layers/
│       ├── __init__.py
│       ├── rmsnorm.py         # RMSNorm layer
│       ├── rope.py            # Rotary Position Embeddings
│       ├── attention.py       # Grouped-Query Attention (GQA)
│       └── swiglu.py          # SwiGLU feed-forward network
├── inference/
│   ├── __init__.py
│   └── sampler.py             # KV-cache inference + sampling strategies
├── notebook.ipynb             # Interactive walkthrough notebook
└── output/                    # Checkpoints, logs (created during training)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

> **GPU Support**: For GPU acceleration, install the GPU version of TensorFlow:
> ```bash
> pip install tensorflow[and-cuda]   # Linux with NVIDIA GPU
> ```

### 2. Prepare Training Data

**Option A — Raw text file** (simplest):
```bash
# Place any .txt file in the datasets directory
cp /path/to/your/corpus.txt data_pipeline/dataset.txt
```

**Option B — Pre-tokenize to TFRecords** (fastest throughput):
```python
from data_pipeline.pipeline import write_tfrecords

write_tfrecords(
    text_path="data_pipeline/dataset.txt",
    output_dir="data_pipeline/tfrecords",
    block_size=512,
    shards=16,
)
```

**Option C — HuggingFace datasets** (no download required with streaming):
```python
# Streams directly from the Hub — no local storage needed
from data_pipeline.pipeline import create_dataset_from_hf

dataset, tokenizer = create_dataset_from_hf(
    "openwebtext",          # or "wikitext", "allenai/c4", etc.
    block_size=512,
    batch_size=32,
    streaming=True,         # Stream without downloading
)
```

### 3. Train

```bash
# Toy model on a text file (quick smoke test)
python train.py --preset toy --source text --text-path data_pipeline/dataset.txt

# Toy model on HuggingFace dataset (streaming)
python train.py --preset toy --source hf --hf-dataset openwebtext

# Small (125M) model on HuggingFace WikiText
python train.py --preset small --source hf \
    --hf-dataset wikitext --hf-config wikitext-103-raw-v1 \
    --hf-streaming

# Train on pre-tokenized TFRecords
python train.py --preset small --source tfrecord --tfrecord-dir data_pipeline/tfrecords

# Medium (1.3B) model with multi-GPU
python train.py --preset medium --source hf \
    --hf-dataset openwebtext --multi-gpu

# Disable mixed precision (if GPU doesn't support float16)
python train.py --preset toy --source text --no-mixed-precision
```

### 4. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir output/tensorboard

# Training metrics are also logged to output/training_log.csv
```

### 5. Generate Text (Inference)

```python
import sys
sys.path.insert(0, ".")

from config import toy_config
from models.transformer import UltraGPT
from data_pipeline.pipeline import TiktokenWrapper
from inference.sampler import UltraGPTSampler

# Load model
config = toy_config()
model = UltraGPT(config)
model.load_weights("output/final_weights")

# Create sampler
tokenizer = TiktokenWrapper()
sampler = UltraGPTSampler(model, tokenizer, config)

# Generate with different strategies
# ── Greedy ────────────────────────────────
text = sampler.generate("Once upon a time", mode="greedy", max_new_tokens=100)
print(text)

# ── Top-K Sampling ───────────────────────
text = sampler.generate("The future of AI", top_k=50, temperature=0.8)
print(text)

# ── Nucleus (Top-p) Sampling ─────────────
text = sampler.generate("In the beginning", top_p=0.9, temperature=0.7)
print(text)

# ── Streaming output ─────────────────────
for token in sampler.generate("Hello world", stream=True):
    print(token, end="", flush=True)
```

---

## 📓 Interactive Notebook

The included `notebook.ipynb` provides a complete interactive walkthrough:

1. **Architecture deep-dive** — inspect each custom layer
2. **Data pipeline demo** — build and visualize training batches
3. **Training loop** — train a toy model end-to-end
4. **Inference** — generate text with all sampling strategies
5. **Profiling** — identify bottlenecks

```bash
jupyter notebook notebook.ipynb
```

---

## ⚙️ Configuration

All hyperparameters live in [`config.py`](config.py). You can create custom configurations:

```python
from config import UltraGPTConfig

# Custom 350M model
config = UltraGPTConfig(
    d_model=1024,
    n_heads=16,
    n_kv_heads=4,       # GQA: 4 query heads per KV head
    n_layers=16,
    block_size=2048,
    vocab_size=50257,
    tie_weights=True,
    rope_theta=10000.0,
    label_smoothing=0.1,
    batch_size=24,
    learning_rate=3e-4,
    max_steps=20000,
)
print(config.summary())
```

---

## 🧩 Key Design Decisions

### Grouped-Query Attention (GQA)
Multiple query heads share fewer KV heads, reducing KV-cache memory by `n_heads / n_kv_heads` × during inference without quality loss.

```
n_heads=32, n_kv_heads=8  → 4 query heads share each KV head
n_heads=32, n_kv_heads=32 → Standard Multi-Head Attention
n_heads=32, n_kv_heads=1  → Multi-Query Attention
```

### RoPE vs Learned Positional Embeddings
RoPE encodes relative positions through rotation matrices, enabling better length generalization and extrapolation. No learnable parameters.

### SwiGLU vs ReLU FFN
SwiGLU uses 3 weight matrices (gate, up, down) instead of 2, but with a smaller hidden dimension `(8/3 × d_model)` vs `(4 × d_model)`, resulting in similar parameter count but better performance.

### Pre-LN vs Post-LN
Pre-LN (normalize before attention/FFN) provides more stable gradients at depth, enabling training of deeper models without gradient issues.

---

## 📊 Training Tips

| Scenario | Recommendation |
|---|---|
| **Debugging** | Use `toy` preset, train on small `.txt` file |
| **Quick experiment** | Use `small` preset with HF streaming + `max_samples=10000` |
| **Serious training** | Pre-tokenize to TFRecords, use `mixed_precision`, multi-GPU |
| **Very large models** | Increase `rope_theta` to 500000 for long context support |
| **Overfitting** | Increase `dropout_rate`, add `label_smoothing=0.1` |
| **Unstable training** | Reduce `learning_rate`, increase `warmup_steps` |

---

## 🔧 CLI Reference

```
python train.py [OPTIONS]

Options:
  --preset {toy,small,medium}   Model size preset (default: toy)
  --source {text,tfrecord,hf}   Data source type (default: text)
  --text-path PATH              Raw text file path (default: data_pipeline/dataset.txt)
  --tfrecord-dir DIR            TFRecord directory (default: data_pipeline/tfrecords)
  --hf-dataset NAME             HuggingFace dataset name (default: openwebtext)
  --hf-config NAME              HuggingFace dataset config/subset
  --hf-text-col NAME            Text column name (default: text)
  --hf-streaming                Use HF streaming mode
  --hf-max-samples N            Limit samples from HF dataset
  --output-dir DIR              Output directory (default: output)
  --epochs N                    Training epochs (default: 1)
  --no-mixed-precision          Disable float16 mixed precision
  --multi-gpu                   Enable multi-GPU MirroredStrategy
```

---

## 📜 License

This project is for educational and research purposes.

---

## 📚 References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)
