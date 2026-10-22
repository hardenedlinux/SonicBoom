#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Tokenize the frozen real-text prompt used by the C.4 prefix-cache tests, with
# the llama.cpp tokenizer for the GGUF under test, and emit a space-separated
# token-id file. The text is a committed fixture (tests/data/real_prompt.txt):
# it was once a concatenation of SonicBoom's own docs and source, and freezing
# it keeps the workload comparable now that the docs it used have moved.
#
#   tests/data/real_prompt.txt  committed fixture (the raw concatenated text)
#   real_prompt.tokens          committed fixture (space-separated token ids)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SONICBOOM_ROOT:-$(cd "$DIR/../.." && pwd)}"

MODEL="${SONICBOOM_GEMMA_GGUF:-$ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf}"
if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL (set SONICBOOM_GEMMA_GGUF)" >&2
  exit 1
fi

TEXT="$ROOT/tests/data/real_prompt.txt"
if [ ! -f "$TEXT" ]; then
  echo "frozen prompt not found: $TEXT" >&2
  exit 1
fi
TOKENS="$DIR/real_prompt.tokens"

"$DIR/tokenize_text" "$MODEL" "$TEXT" > "$TOKENS.full"

# Truncate to a committed fixture size large enough for the longest acceptance
# case (prefix 8193 + new 600 + 32 continuation steps + margin).
MAX_TOKENS="${REAL_PROMPT_MAX_TOKENS:-9000}"
cut -d' ' -f1-"$MAX_TOKENS" "$TOKENS.full" > "$TOKENS"
rm -f "$TOKENS.full"

n=$(wc -w < "$TOKENS")
echo "wrote $TOKENS ($n tokens)"
