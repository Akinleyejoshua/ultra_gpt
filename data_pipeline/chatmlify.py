"""
ChatML Converter Utility
=========================
Converts raw text datasets (Q/A or conversational turns) into the industry-standard ChatML format:
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
"""

import argparse
import os
import re

def chatmlify_text(raw_text: str) -> str:
    """Parse raw conversational text and convert it to ChatML format.
    
    Supports formats:
      - Turn prefixes: "User: [text]" / "Assistant: [text]"
      - Q&A prefixes: "Q: [text]" / "A: [text]" or "Question: [text]" / "Answer: [text]"
      - Question mark separator: "hi? hello" (converts Q? A to ChatML turns)
    """
    lines = raw_text.strip().split("\n")
    chatml_turns = []
    
    # 1. First, check if the text contains standard User/Assistant or Q/A prefixes
    has_user_prefix = any(re.match(r"^(User|System|Assistant|Q:|A:|Question:|Answer:)", line, re.IGNORECASE) for line in lines if line.strip())
    
    if has_user_prefix:
        current_role = None
        current_content = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Match role prefix
            match = re.match(r"^(User|System|Assistant|Q|A|Question|Answer)\s*:\s*(.*)", line_str, re.IGNORECASE)
            if match:
                # If there was a previous turn, save it
                if current_role and current_content:
                    chatml_turns.append(f"<|im_start|>{current_role}\n{' '.join(current_content)}<|im_end|>")
                    current_content = []
                
                role_str = match.group(1).lower()
                # Normalize role
                if role_str in ["user", "q", "question"]:
                    current_role = "user"
                elif role_str in ["assistant", "a", "answer"]:
                    current_role = "assistant"
                elif role_str in ["system"]:
                    current_role = "system"
                else:
                    current_role = "user"
                    
                current_content.append(match.group(2).strip())
            else:
                # Continuation of the current role's turn
                if current_role:
                    current_content.append(line_str)
                    
        # Add the last turn
        if current_role and current_content:
            chatml_turns.append(f"<|im_start|>{current_role}\n{' '.join(current_content)}<|im_end|>")
            
    else:
        # 2. Check if we have simple "Question? Answer" style lines (e.g. "hi? hello")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            
            if "?" in line_str:
                parts = line_str.split("?", 1)
                question = parts[0].strip() + "?"
                answer = parts[1].strip()
                
                chatml_turns.append(f"<|im_start|>user\n{question}<|im_end|>")
                chatml_turns.append(f"<|im_start|>assistant\n{answer}<|im_end|>")
            else:
                # Default to user message if no question mark
                chatml_turns.append(f"<|im_start|>user\n{line_str}<|im_end|>")

    return "\n".join(chatml_turns) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Convert raw text datasets to ChatML format")
    parser.add_argument("--input", default="data_pipeline/dataset.txt", help="Input raw dataset text file")
    parser.add_argument("--output", default="data_pipeline/dataset_chatml.txt", help="Output ChatML formatted text file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return
        
    with open(args.input, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    chatml_content = chatmlify_text(raw_content)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(chatml_content)
        
    print(f"Successfully converted '{args.input}' to ChatML format and saved to '{args.output}'!")

if __name__ == "__main__":
    main()
