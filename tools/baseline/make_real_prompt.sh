#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Tokenize the frozen real-text prompt used by the prefix-cache tests and the
# canonical bench, with the llama.cpp tokenizer for the GGUF under test, and
# emit a space-separated token-id file. The text is a frozen fixture kept with
# the private engine suites (private tests/data/real_prompt.txt): it was once a
# concatenation of SonicBoom's own docs and source, and freezing it keeps the
# workload comparable now that the docs it used have moved. An older checkout
# keeps the text under tests/data/; both are accepted and the private copy wins.
#
#   <private>/tests/data/real_prompt.txt  frozen fixture (raw concatenated text)
#   real_prompt.tokens                    this tool's output (space-separated ids)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SONICBOOM_ROOT:-$(cd "$DIR/../.." && pwd)}"

MODEL="${SONICBOOM_GEMMA_GGUF:-$ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf}"
if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL (set SONICBOOM_GEMMA_GGUF)" >&2
  exit 1
fi

PRIVATE_ROOT="${SONICBOOM_PRIVATE_ROOT:-$ROOT/../sonicboom-private}"
TEXT="$PRIVATE_ROOT/tests/data/real_prompt.txt"
[ -f "$TEXT" ] || TEXT="$ROOT/tests/data/real_prompt.txt"
if [ ! -f "$TEXT" ]; then
  # The private suites are optional: without them there is no frozen prompt to
  # tokenize, and no private consumer to serve. Say so and leave the existing
  # tokens file untouched rather than failing a bench setup.
  echo "SKIP: frozen prompt not found (looked in $PRIVATE_ROOT/tests/data and" \
       "$ROOT/tests/data; set SONICBOOM_PRIVATE_ROOT)" >&2
  exit 0
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
