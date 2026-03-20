#!/usr/bin/env python3
"""
Verify how Qwen2.5-Coder tokenizer renders tool-calling messages.

What this script does:
1) Loads Qwen/Qwen2.5-Coder-1.5B-Instruct tokenizer
2) Prints tokenizer.chat_template (so you can see the roles / special tokens)
3) Builds:
   - an OpenAI-compatible tool-calling message sequence (tool_calls + tool_call_id)
   - a "bad" XML-tag style <tool_call>... example for comparison
4) Uses tokenizer.apply_chat_template(..., tokenize=False) to show the EXACT rendered text
5) Tokenizes the rendered text and prints:
   - input_ids length
   - first N token IDs
   - first N tokens (strings)
   - a small slice around where the JSON tool call appears

Run:
  pip install -U transformers accelerate
  python verify_qwen25_tool_tokenizer.py
"""

from __future__ import annotations

import json
from typing import List, Dict, Any

from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def find_substring_spans(text: str, needle: str) -> List[int]:
    """Return all start indices where needle occurs in text."""
    starts = []
    i = 0
    while True:
        j = text.find(needle, i)
        if j == -1:
            break
        starts.append(j)
        i = j + max(1, len(needle))
    return starts


def main() -> None:
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("\n===== tokenizer.special_tokens_map =====")
    print(pretty(tokenizer.special_tokens_map))

    print("\n===== tokenizer.chat_template =====")
    # Some tokenizers may have None; Qwen should have it.
    print(tokenizer.chat_template)

    # ---------------------------------------------------------------------
    # A) OpenAI-compatible tool-calling style messages (recommended)
    # ---------------------------------------------------------------------
    messages_good: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个代码智能体。需要实时信息时请调用工具。"},
        {"role": "user", "content": "巴黎现在多少度？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "arguments": '{"location":"Paris, France"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18°C"},
        {"role": "assistant", "content": "巴黎现在是 18 摄氏度。"},
    ]

    print("\n\n==============================")
    print("A) GOOD messages (tool_calls + tool_call_id)")
    print("==============================")
    print(pretty(messages_good))

    rendered_good = tokenizer.apply_chat_template(
        messages_good, tokenize=False, add_generation_prompt=False
    )

    print("\n===== Rendered text (GOOD) =====")
    print(rendered_good)

    # Tokenize the rendered text to see actual tokens
    enc_good = tokenizer(rendered_good, return_tensors="pt")
    ids_good = enc_good["input_ids"][0].tolist()
    print("\n===== Tokenization stats (GOOD) =====")
    print("Total tokens:", len(ids_good))

    N = 120
    print(f"\nFirst {N} token IDs:")
    print(ids_good[:N])

    tokens_good = tokenizer.convert_ids_to_tokens(ids_good)
    print(f"\nFirst {N} tokens:")
    print(tokens_good[:N])

    # Try to locate a likely JSON marker in the rendered text for local slice viewing
    # Depending on the template, the JSON may appear in assistant block as text.
    # We'll search for '"get_current_temperature"' or '"name"' patterns.
    needles = ['"get_current_temperature"', '"name"', "get_current_temperature"]
    for needle in needles:
        spans = find_substring_spans(rendered_good, needle)
        if spans:
            print(f"\nFound '{needle}' in rendered text at positions: {spans[:5]} (showing up to 5)")
            # Show a short slice around the first occurrence
            s = spans[0]
            left = max(0, s - 160)
            right = min(len(rendered_good), s + 200)
            print("\n----- Rendered slice around first match -----")
            print(rendered_good[left:right])
            break
    else:
        print("\nCould not find JSON needle in rendered text (template may render tool calls differently).")

    # ---------------------------------------------------------------------
    # B) "Bad" / non-native style: XML tags inside assistant content
    # ---------------------------------------------------------------------
    messages_bad: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个代码智能体。"},
        {"role": "user", "content": "巴黎现在多少度？"},
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n"
                '{"arguments": {"location": "Paris, France"}, "name": "get_current_temperature"}\n'
                "</tool_call>"
            ),
        },
    ]

    print("\n\n==============================")
    print("B) BAD messages (<tool_call> XML tag style)")
    print("==============================")
    print(pretty(messages_bad))

    rendered_bad = tokenizer.apply_chat_template(
        messages_bad, tokenize=False, add_generation_prompt=False
    )

    print("\n===== Rendered text (BAD) =====")
    print(rendered_bad)

    enc_bad = tokenizer(rendered_bad, return_tensors="pt")
    ids_bad = enc_bad["input_ids"][0].tolist()
    print("\n===== Tokenization stats (BAD) =====")
    print("Total tokens:", len(ids_bad))

    tokens_bad = tokenizer.convert_ids_to_tokens(ids_bad)
    print(f"\nFirst {N} tokens (BAD):")
    print(tokens_bad[:N])

    # Compare whether '<tool_call>' is a single token or split into many (it will be split)
    tag = "<tool_call>"
    if tag in rendered_bad:
        print(f"\n'{tag}' appears in rendered BAD text. Checking how it tokenizes...")
        # Find the first occurrence in token strings (not perfect, but indicative)
        # We'll print a window around the first token that contains "tool" or "<"
        idx = None
        for i, t in enumerate(tokens_bad):
            if "tool" in t or "<" in t:
                idx = i
                break
        if idx is not None:
            w_left = max(0, idx - 20)
            w_right = min(len(tokens_bad), idx + 40)
            print("\n----- Token window around tag-ish tokens -----")
            print(tokens_bad[w_left:w_right])
        else:
            print("Could not locate tag-ish tokens in token list (unexpected).")

    print("\n\nDONE.")
    print("If you see that tool call JSON is just normal text tokens, that's the expected result for Qwen2.5.")
    print("The only real 'special' tokens are the chat delimiters like <|im_start|>/<|im_end|>.")


if __name__ == "__main__":
    main()
