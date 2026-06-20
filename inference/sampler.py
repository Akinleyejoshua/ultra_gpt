"""
UltraGPT Production Inference Sampler
======================================
Low-latency autoregressive text generation with:
  - KV-cache for O(1) per-token compute
  - XLA compilation via @tf.function(jit_compile=True)
  - Greedy, Top-K, Top-p (Nucleus), and Temperature sampling
  - Streaming token output

Inspired by Groq's LPU-style low-latency generation and
Claude's streaming inference engines.
"""

import time
import tensorflow as tf
import numpy as np
from config import UltraGPTConfig


class UltraGPTSampler:
    """Production inference sampler for UltraGPT.

    Manages KV-cache lifecycle and provides multiple decoding
    strategies with XLA-compiled generation steps.

    Args:
        model: Trained UltraGPT model instance.
        tokenizer: TiktokenWrapper (or compatible) tokenizer.
        config: UltraGPTConfig instance.
    """

    def __init__(self, model, tokenizer, config: UltraGPTConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Pre-compile the XLA generation step
        self._compiled_step = tf.function(
            self._generation_step,
            jit_compile=True,
        )

    def _generation_step(self, token_ids, cache_list):
        """Single autoregressive generation step (XLA-compiled).

        Takes one token (or a prompt), runs through the model with
        KV-cache, and returns logits + updated cache.

        Args:
            token_ids: Shape (1, seq_len) — single token during generation,
                       or full prompt during prefill.
            cache_list: List of (cached_k, cached_v) tuples per layer.

        Returns:
            logits: Shape (1, seq_len, vocab_size).
            new_cache_list: Updated cache list.
        """
        logits, new_cache_list = self.model(
            token_ids, cache_list=cache_list, training=False
        )
        return logits, new_cache_list

    def _init_empty_cache(self):
        """Initialize empty KV-cache for all layers.

        Returns:
            List of None values (one per decoder block).
            The model will populate them on the first forward pass.
        """
        return [None] * self.config.n_layers

    def _sample_from_logits(
        self,
        logits,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        mode="sample",
    ):
        """Apply sampling strategy to logits and return next token.

        Args:
            logits: Shape (vocab_size,) — logits for next token.
            temperature: Scales logits. Lower = more deterministic.
            top_k: If > 0, keep only top-K logits.
            top_p: If < 1.0, apply nucleus sampling.
            mode: "greedy", "sample", or "top_k"/"top_p".

        Returns:
            next_token: Scalar int32 tensor.
        """
        # ── Temperature scaling ───────────────────────────────────────
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        elif temperature <= 0:
            # Temperature 0 → greedy
            return tf.argmax(logits, axis=-1, output_type=tf.int32)

        # ── Greedy decoding ───────────────────────────────────────────
        if mode == "greedy":
            return tf.argmax(logits, axis=-1, output_type=tf.int32)

        # ── Top-K filtering ──────────────────────────────────────────
        if top_k > 0 and top_k < tf.shape(logits)[-1]:
            top_k_values, _ = tf.math.top_k(logits, k=top_k)
            threshold = top_k_values[-1]  # k-th largest value
            logits = tf.where(
                logits < threshold,
                tf.fill(tf.shape(logits), -1e9),
                logits,
            )

        # ── Top-p (Nucleus) filtering ────────────────────────────────
        if top_p < 1.0:
            sorted_logits = tf.sort(logits, direction="DESCENDING")
            sorted_probs = tf.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = tf.cumsum(sorted_probs, axis=-1)

            # Find cutoff index: first position where cumulative > top_p
            # Shift right by 1 so we keep the token that crosses threshold
            sorted_mask = cumulative_probs - sorted_probs > top_p
            # Set masked logits to -inf
            sorted_logits = tf.where(
                sorted_mask,
                tf.fill(tf.shape(sorted_logits), -1e9),
                sorted_logits,
            )
            # Map back to original ordering
            min_allowed = tf.reduce_min(
                tf.where(sorted_mask, tf.float32.max, sorted_logits)
            )
            logits = tf.where(
                logits < min_allowed,
                tf.fill(tf.shape(logits), -1e9),
                logits,
            )

        # ── Categorical sampling ─────────────────────────────────────
        probs = tf.nn.softmax(logits, axis=-1)
        next_token = tf.random.categorical(
            tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32
        )
        return tf.squeeze(next_token)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        mode: str = "sample",
        stop_token_id: int = None,
        stream: bool = False,
        verbose: bool = True,
    ):
        """Generate text from a prompt.

        Args:
            prompt: Input text string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (overrides config).
            top_k: Top-K value (overrides config).
            top_p: Top-p value (overrides config).
            mode: "greedy", "sample". top_k/top_p are applied if set.
            stop_token_id: Stop generation at this token. Defaults to EOS.
            stream: If True, yields tokens as they're generated.
            verbose: If True, prints timing stats.

        Returns:
            If stream=False: Generated text string.
            If stream=True: Generator yielding token strings.
        """
        # ── Defaults from config ──────────────────────────────────────
        max_new_tokens = max_new_tokens or self.config.max_gen_length
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        stop_token_id = stop_token_id if stop_token_id is not None else self.tokenizer.eos_token_id

        # ── Encode prompt ─────────────────────────────────────────────
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        if verbose:
            print(f"[Sampler] Prompt: {prompt_len} tokens, "
                  f"generating up to {max_new_tokens} tokens")
            print(f"[Sampler] Mode: {mode}, temp={temperature}, "
                  f"top_k={top_k}, top_p={top_p}")

        # ── Prefill: process entire prompt with KV-cache ──────────────
        t_start = time.perf_counter()

        input_ids = tf.constant([prompt_tokens], dtype=tf.int32)  # (1, prompt_len)
        cache_list = self._init_empty_cache()

        # Prefill pass (processes full prompt, populates cache)
        logits, cache_list = self._compiled_step(input_ids, cache_list)

        t_prefill = time.perf_counter() - t_start
        if verbose:
            print(f"[Sampler] Prefill: {t_prefill:.3f}s "
                  f"({prompt_len / t_prefill:.0f} tok/s)")

        # ── Get first generated token from last position ──────────────
        next_logits = logits[0, -1, :]  # (vocab_size,)
        next_token = self._sample_from_logits(
            next_logits, temperature=temperature,
            top_k=top_k, top_p=top_p, mode=mode,
        )

        generated_tokens = [next_token.numpy()]

        if stream:
            return self._stream_generate(
                next_token, cache_list, generated_tokens,
                max_new_tokens, temperature, top_k, top_p,
                mode, stop_token_id, verbose, t_start,
            )

        # ── Autoregressive decode loop ────────────────────────────────
        t_decode_start = time.perf_counter()

        for step in range(1, max_new_tokens):
            # Feed single token with cache
            token_input = tf.constant([[next_token.numpy()]], dtype=tf.int32)
            logits, cache_list = self._compiled_step(token_input, cache_list)

            next_logits = logits[0, -1, :]
            next_token = self._sample_from_logits(
                next_logits, temperature=temperature,
                top_k=top_k, top_p=top_p, mode=mode,
            )

            token_id = next_token.numpy()
            generated_tokens.append(token_id)

            # Stop on EOS
            if token_id == stop_token_id:
                break

        t_total = time.perf_counter() - t_start
        t_decode = time.perf_counter() - t_decode_start
        n_gen = len(generated_tokens)

        if verbose:
            print(f"[Sampler] Decode: {n_gen} tokens in {t_decode:.3f}s "
                  f"({n_gen / t_decode:.0f} tok/s)")
            print(f"[Sampler] Total: {t_total:.3f}s "
                  f"({(prompt_len + n_gen) / t_total:.0f} tok/s overall)")

        # Decode full output
        output_text = self.tokenizer.decode(generated_tokens)
        return prompt + output_text

    def _stream_generate(
        self, next_token, cache_list, generated_tokens,
        max_new_tokens, temperature, top_k, top_p,
        mode, stop_token_id, verbose, t_start,
    ):
        """Generator that yields tokens one at a time for streaming UX.

        Yields:
            Token strings as they are generated.
        """
        # Yield the first token
        yield self.tokenizer.decode([generated_tokens[-1]])

        for step in range(1, max_new_tokens):
            token_input = tf.constant([[next_token.numpy()]], dtype=tf.int32)
            logits, cache_list = self._compiled_step(token_input, cache_list)

            next_logits = logits[0, -1, :]
            next_token = self._sample_from_logits(
                next_logits, temperature=temperature,
                top_k=top_k, top_p=top_p, mode=mode,
            )

            token_id = next_token.numpy()
            generated_tokens.append(token_id)

            if token_id == stop_token_id:
                break

            yield self.tokenizer.decode([token_id])

        if verbose:
            t_total = time.perf_counter() - t_start
            n_gen = len(generated_tokens)
            print(f"\n[Sampler] Streamed {n_gen} tokens in {t_total:.3f}s "
                  f"({n_gen / t_total:.0f} tok/s)")

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        mode: str = "sample",
    ) -> list[str]:
        """Generate text for multiple prompts (no KV-cache, simpler path).

        For batch inference where individual KV-cache management is complex,
        this uses a simpler full-forward approach.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature.
            top_k: Top-K value.
            top_p: Top-p value.
            mode: Sampling mode.

        Returns:
            List of generated text strings.
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mode=mode,
                stream=False,
                verbose=False,
            )
            results.append(result)
        return results
