#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/plan_diff. It is a standalone text comparison and links nothing —
# not even libsonicboom.so — so it builds anywhere and can be used to diff a plan
# recorded on another machine.
set -euo pipefail

CXX="${CXX:-g++-13}"

"$CXX" -std=c++23 -O2 "$(dirname "$0")/plan_diff.cpp" -o "$(dirname "$0")/plan_diff"

echo "built $(dirname "$0")/plan_diff"
