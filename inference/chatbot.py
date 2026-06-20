"""
UltraGPT Chatbot Interface
===========================
Interactive, multi-turn chatbot interface with context window clipping,
history tracking, and real-time streaming response generation.
"""

import os
import argparse
import tensorflow as tf
from config import UltraGPTConfig, toy_config, small_config, medium_config
from models.transformer import UltraGPT
from data_pipeline.pipeline import TiktokenWrapper
from inference.sampler import UltraGPTSampler


class UltraGPTChatbot:
    """Manages the conversation history, formatting, and generation lifecycle.

    Args:
        sampler: UltraGPTSampler instance.
        system_prompt: Default instructions for the assistant.
    """

    def __init__(self, sampler: UltraGPTSampler, system_prompt: str = None):
        self.sampler = sampler
        self.config = sampler.config
        self.tokenizer = sampler.tokenizer
        self.system_prompt = system_prompt or "You are a helpful, focused, and concise AI assistant."
        self.history = []  # List of dicts: {"user": str, "assistant": str}

    def clear_history(self):
        """Reset the conversation context."""
        self.history = []

    def _get_formatted_prompt(self, new_user_message: str, history_slice: list) -> str:
        """Format the system prompt, history slice, and new message into a flat string."""
        prompt = f"System: {self.system_prompt}\n"
        for turn in history_slice:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        prompt += f"User: {new_user_message}\nAssistant:"
        return prompt

    def _get_token_count(self, text: str) -> int:
        """Get the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def _get_pruned_prompt(self, new_user_message: str, max_new_tokens: int) -> tuple:
        """Prunes conversation history until the final prompt fits in the block_size limit.

        Ensures that (prompt_length + max_new_tokens) <= block_size.
        Returns a tuple of (prompt, adjusted_max_new_tokens).
        """
        # Context budget for the prompt: guarantee at least a small prompt budget (e.g. 40 tokens or 1/3 of block_size)
        min_prompt_budget = min(40, self.config.block_size // 3)
        if min_prompt_budget < 10:
            min_prompt_budget = 10
            
        if max_new_tokens > self.config.block_size - min_prompt_budget:
            max_new_tokens = self.config.block_size - min_prompt_budget

        max_prompt_budget = self.config.block_size - max_new_tokens

        # Slice history gradually from the left (oldest first) until it fits
        history_slice = list(self.history)
        while len(history_slice) > 0:
            prompt = self._get_formatted_prompt(new_user_message, history_slice)
            token_count = self._get_token_count(prompt)
            if token_count <= max_prompt_budget:
                return prompt, max_new_tokens
            # Pop oldest turn
            history_slice.pop(0)

        # If it still doesn't fit even with empty history, truncate the new message itself
        prompt = self._get_formatted_prompt(new_user_message, [])
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > max_prompt_budget:
            # Keep only the last part of the prompt that fits
            truncated_tokens = tokens[-max_prompt_budget:]
            prompt = self.tokenizer.decode(truncated_tokens)

        return prompt, max_new_tokens

    def respond(
        self,
        user_message: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        mode: str = "sample",
        stream: bool = True,
    ):
        """Generate response for a user message, yielding tokens if stream=True."""
        prompt, max_new_tokens = self._get_pruned_prompt(user_message, max_new_tokens)

        if stream:
            # We must strip the prompt from the output to only yield new assistant text
            # Sampler.generate returns prompt + completion. For streaming it yields just tokens.
            generator = self.sampler.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mode=mode,
                stream=True,
                verbose=False,
            )
            
            full_reply = ""
            for token in generator:
                full_reply += token
                yield token
                
            # Save completed turn to history
            self.history.append({"user": user_message, "assistant": full_reply})
        else:
            full_response = self.sampler.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mode=mode,
                stream=False,
                verbose=False,
            )
            # Extracted assistant response (remove the prefilled prompt context)
            reply = full_response[len(prompt):].strip()
            self.history.append({"user": user_message, "assistant": reply})
            yield reply


# ═══════════════════════════════════════════════════════════════════════
# Console Chat Session
# ═══════════════════════════════════════════════════════════════════════

def run_chat_cli(chatbot: UltraGPTChatbot):
    """Run an interactive chat session in the terminal."""
    print("=" * 60)
    print(" 💬 Welcome to the UltraGPT Interactive Chatbot Session!")
    print(f"    (Model context window size: {chatbot.config.block_size} tokens)")
    print("    Type '/clear' to reset history, '/exit' to quit.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("\033[94m\033[1mUser:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Chatbot] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("[Chatbot] Goodbye!")
            break

        if user_input.lower() == "/clear":
            chatbot.clear_history()
            print("\033[93m[Chatbot] Conversation history cleared.\033[0m\n")
            continue

        print("\033[92m\033[1mAssistant:\033[0m ", end="", flush=True)

        try:
            # Stream response to the terminal
            response_generator = chatbot.respond(
                user_message=user_input,
                max_new_tokens=chatbot.config.max_gen_length,
                temperature=chatbot.config.temperature,
                top_k=chatbot.config.top_k,
                top_p=chatbot.config.top_p,
            )
            for token in response_generator:
                print(token, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n\033[91m[Error during generation]: {e}\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Run UltraGPT Chatbot CLI")
    parser.add_argument("--preset", choices=["toy", "small", "medium"], default="toy")
    parser.add_argument("--weights", required=True, help="Path to weights file (.weights.h5)")
    parser.add_argument("--system-prompt", default=None, help="System instructions")
    args = parser.parse_args()

    # Load configuration
    config_map = {"toy": toy_config, "small": small_config, "medium": medium_config}
    config = config_map[args.preset]()

    print("[Chatbot] Initializing model structure...")
    model = UltraGPT(config)
    
    # Dummy forward pass to construct model variables before loading weights
    dummy_input = tf.zeros((1, config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    
    print(f"[Chatbot] Loading weights from {args.weights}...")
    model.load_weights(args.weights)

    tokenizer = TiktokenWrapper()
    sampler = UltraGPTSampler(model, tokenizer, config)
    chatbot = UltraGPTChatbot(sampler, system_prompt=args.system_prompt)

    run_chat_cli(chatbot)


if __name__ == "__main__":
    main()
