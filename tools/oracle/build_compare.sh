#!/usr/bin/env bash
# Build the dump-comparison tool. This is a pure C++ tool with no llama.cpp or
# SonicBoom dependency; it only reads the text dump format.
set -euo pipefail

CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 "$(dirname "$0")/compare_dumps.cpp" \
  -o "$(dirname "$0")/compare_dumps"

echo "built $(dirname "$0")/compare_dumps"
