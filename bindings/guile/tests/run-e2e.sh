#!/bin/sh
# Orchestrates the Guile → C++ end-to-end loop:
#   1. the Guile exporter trains a model and writes model.sx + weights.bin +
#      expected.bin into a scratch directory,
#   2. the independent C++ loader loads that artifact and checks its forward
#      outputs match expected.bin.
# Exits non-zero if either half fails.
#
# Usage: run-e2e.sh <guile> <scheme-dir> <exporter.scm> <loader-bin> <lib-dir>

set -eu

GUILE="$1"
SCHEME_DIR="$2"
EXPORTER="$3"
LOADER="$4"
LIB_DIR="$5"

export GUILE_AUTO_COMPILE=0
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DIR="$(mktemp -d "${TMPDIR:-/tmp}/sonicboom-e2e.XXXXXX")"
trap 'rm -rf "$DIR"' EXIT

"$GUILE" --no-auto-compile -L "$SCHEME_DIR" "$EXPORTER" "$DIR"
"$LOADER" "$DIR"

echo "guile-cpp-e2e OK"
