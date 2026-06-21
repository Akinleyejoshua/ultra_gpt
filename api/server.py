import os
import sys
import time
import asyncio
import queue
import threading
from typing import List, Tuple, Generator

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from config import UltraGPTConfig, toy_config, small_config, medium_config
from models.transformer import UltraGPT
from data_pipeline.pipeline import TiktokenWrapper
from inference.sampler import UltraGPTSampler
from api.schemas import ChatMessage

class ModelServer:
    """Manages the lifecycle, configuration, and inference lock for UltraGPT model."""
    
    def __init__(self, preset: str, weights_path: str):
        self.preset = preset
        self.weights_path = weights_path
        self.lock = asyncio.Lock()
        
        # Load configuration preset
        config_map = {
            "toy": toy_config,
            "small": small_config,
            "medium": medium_config,
            "notebook": lambda: UltraGPTConfig(
                d_model=1000,
                n_heads=100,
                n_kv_heads=20,
                n_layers=6,
                block_size=128
            )
        }
        if preset not in config_map:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: toy, small, medium, notebook")
        
        self.config = config_map[preset]()
        
        print(f"[ModelServer] Initializing model with preset '{preset}'...")
        self.model = UltraGPT(self.config)
        
        # Dummy forward pass to construct variables
        dummy_input = tf.zeros((1, self.config.block_size), dtype=tf.int32)
        _ = self.model(dummy_input, training=False)
        
        print(f"[ModelServer] Loading weights from {weights_path}...")
        self.model.load_weights(weights_path)
        
        self.tokenizer = TiktokenWrapper()
        self.sampler = UltraGPTSampler(self.model, self.tokenizer, self.config)
        print("[ModelServer] Model and sampler successfully loaded.")

    def format_and_prune_prompt(self, messages: List[ChatMessage], max_new_tokens: int) -> Tuple[str, int]:
        """Formats message list in ChatML template and prunes context dynamically to fit block_size."""
        # Always prioritize system prompt and the latest user message
        system_message = None
        other_messages = []
        for m in messages:
            if m.role == "system":
                system_message = m
            else:
                other_messages.append(m)

        # Minimum tokens we want to reserve for model generation
        min_generation_tokens = 16
        max_prompt_len = self.config.block_size - min_generation_tokens

        # Prune older history turns from the front, but always keep the last message
        while len(other_messages) > 1:
            prompt = self._build_prompt_string(system_message, other_messages)
            token_count = len(self.tokenizer.encode(prompt))
            if token_count <= max_prompt_len:
                break
            other_messages.pop(0)

        # Truncate prompt from left if the latest message still exceeds limits
        prompt = self._build_prompt_string(system_message, other_messages)
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > max_prompt_len:
            header_tokens = self.tokenizer.encode("<|im_start|>assistant\n")
            content_tokens = tokens[:-len(header_tokens)]
            allowed_content_len = max_prompt_len - len(header_tokens)
            content_tokens = content_tokens[-allowed_content_len:]
            tokens = content_tokens + header_tokens
            prompt = self.tokenizer.decode(tokens)

        # Dynamically scale max tokens to remaining context space
        prompt_len = len(self.tokenizer.encode(prompt))
        actual_max_tokens = min(max_new_tokens, self.config.block_size - prompt_len)
        if actual_max_tokens < 1:
            actual_max_tokens = 1

        return prompt, actual_max_tokens

    def _build_prompt_string(self, system_message: ChatMessage, messages: List[ChatMessage]) -> str:
        prompt = ""
        if system_message:
            prompt += f"<|im_start|>system\n{system_message.content}<|im_end|>\n"
        else:
            prompt += "<|im_start|>system\nYou are a helpful, focused, and concise AI assistant.<|im_end|>\n"
            
        for m in messages:
            prompt += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """Synchronously generate response (intended to run under thread executor)."""
        generator = self.sampler.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            mode="sample",
            stream=True,
            verbose=False,
        )
        
        full_reply = ""
        buffer = ""
        stop_patterns = ["<|im_start|>", "<|im_end|>", "<|", "|>", "im_start", "im_end"]
        for token in generator:
            buffer += token
            
            # Check for ChatML boundaries or leaked tags
            first_idx = len(buffer)
            found = False
            for pattern in stop_patterns:
                if pattern in buffer:
                    idx = buffer.find(pattern)
                    if idx < first_idx:
                        first_idx = idx
                        found = True
            if found:
                full_reply += buffer[:first_idx]
                break
            
            if len(buffer) > 20:
                yield_len = len(buffer) - 15
                full_reply += buffer[:yield_len]
                buffer = buffer[yield_len:]
        else:
            full_reply += buffer
            
        return full_reply.strip()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        q: queue.Queue
    ):
        """Streams generated tokens to a thread-safe queue."""
        try:
            generator = self.sampler.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mode="sample",
                stream=True,
                verbose=False,
            )
            
            buffer = ""
            stop_patterns = ["<|im_start|>", "<|im_end|>", "<|", "|>", "im_start", "im_end"]
            for token in generator:
                buffer += token
                
                # Check for ChatML boundaries or leaked tags
                first_idx = len(buffer)
                found = False
                for pattern in stop_patterns:
                    if pattern in buffer:
                        idx = buffer.find(pattern)
                        if idx < first_idx:
                            first_idx = idx
                            found = True
                if found:
                    q.put(buffer[:first_idx])
                    break
                
                if len(buffer) > 20:
                    yield_len = len(buffer) - 15
                    q.put(buffer[:yield_len])
                    buffer = buffer[yield_len:]
            else:
                q.put(buffer)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)  # Sentinel value signaling end of stream
