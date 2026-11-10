#!/bin/bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C

root=$(cd "$(dirname "$0")/.." && pwd)
export LD_LIBRARY_PATH="$root/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec guile -L "$root/bindings/guile/scheme" \
     "$root/bindings/guile/scheme/sonicboom/serve/app.scm" \
     --port 8000 --model "$root/models/qwen2.5-7b-instruct-1m-q4_k_m.gguf" \
     --model-id qwen2.5-7b --context-length 16384 --max-sessions 1
