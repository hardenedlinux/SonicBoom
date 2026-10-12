#!/usr/bin/env bash
# Generate a real-text prompt for the C.4 prefix-cache tests: concatenate a
# fixed set of SonicBoom's own docs + source (deterministic order), tokenize it
# with the llama.cpp tokenizer for the GGUF under test, and emit a
# space-separated token-id file. The tokens file is committed as a test fixture
# so the correctness/acceptance tests do not depend on llama.cpp at test time.
#
#   real_prompt.txt    gitignored intermediate (the raw concatenated text)
#   real_prompt.tokens committed fixture (space-separated token ids)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SONICBOOM_ROOT:-$(cd "$DIR/../.." && pwd)}"

MODEL="${SONICBOOM_GEMMA_GGUF:-$ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf}"
if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL (set SONICBOOM_GEMMA_GGUF)" >&2
  exit 1
fi

# Fixed, ordered file list. Do not reorder without regenerating the committed
# fixture. Comment/blank lines are kept as-is (real text).
FILES=(
  "$ROOT/CLAUDE.md"
  "$ROOT/design/backend-hal.md"
  "$ROOT/design/gemma4-semantics.md"
  "$ROOT/design/native-torch-migration.md"
  "$ROOT/core/src/model/gemma4_plan.cpp"
  "$ROOT/core/src/planner/sonic_cuda_backend.cpp"
  "$ROOT/core/src/nn/cuda_elementwise.cu"
  "$ROOT/core/src/model/exec.cpp"
)

TEXT="$DIR/real_prompt.txt"
TOKENS="$DIR/real_prompt.tokens"

: > "$TEXT"
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "missing source file: $f" >&2
    exit 1
  fi
  cat "$f" >> "$TEXT"
  printf '\n' >> "$TEXT"
done

"$DIR/tokenize_text" "$MODEL" "$TEXT" > "$TEXT.tokens.full"

# Truncate to a committed fixture size large enough for the longest acceptance
# case (prefix 4097 + new 600 + 32 continuation steps + margin).
MAX_TOKENS="${REAL_PROMPT_MAX_TOKENS:-8000}"
cut -d' ' -f1-"$MAX_TOKENS" "$TEXT.tokens.full" > "$TOKENS"
rm -f "$TEXT.tokens.full"

n=$(wc -w < "$TOKENS")
echo "wrote $TOKENS ($n tokens)"
