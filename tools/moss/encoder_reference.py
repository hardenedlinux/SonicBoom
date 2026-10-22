#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Official reference dump for the MOSS-Image text encoder (S2 gate).

Runs the Hugging Face Qwen3.5 text model on CPU in bfloat16 -- the dtype the
checkpoint stores -- and writes the last hidden state in the same MOSSHST1
container sb-moss writes, so tools/moss/compare_hidden.py can diff them.

The token ids are printed as well; they must match `sb-moss tokens`.
"""

import argparse
import os
import struct
import sys

import torch
from transformers import AutoModel, AutoTokenizer


def tokenizer_dir(model_dir: str) -> str:
    """The MOSS-Image layout keeps the tokenizer beside the text encoder: the
    text_encoder directory holds config + shards only, the tokenizer files are
    in ../tokenizer. Point AutoTokenizer at whichever one actually has them
    (sb-moss's loader uses the same fallback)."""
    if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
        return model_dir
    sibling = os.path.join(os.path.dirname(os.path.abspath(model_dir)),
                           "tokenizer")
    if os.path.exists(os.path.join(sibling, "tokenizer.json")):
        return sibling
    return model_dir


def chat_prompt(text: str) -> str:
    return (
        "<|im_start|>user\n"
        + text
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="text_encoder directory")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir(args.model))
    text = args.text if args.raw else chat_prompt(args.text)
    inputs = tokenizer(text, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    print(" ".join(str(i) for i in ids), flush=True)

    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        out = model(**inputs, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state[0].float().numpy()
    tokens, width = hidden.shape
    with open(args.out, "wb") as f:
        f.write(b"MOSSHST1")
        f.write(struct.pack("<qq", tokens, width))
        f.write(hidden.astype("<f4").tobytes())
    print(f"wrote {args.out} ({tokens}x{width})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
