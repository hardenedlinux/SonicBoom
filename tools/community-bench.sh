#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# community-bench.sh — the community run in one command, one log.
#
# A community submission is the steps in README, "Reproduce the table": the
# hardware/build record, the llama.cpp commit, the code-editing comparison and
# the extended-context tiers, plus the server's streamed code-edit rate with
# --serve. Run by hand they leave separate terminal outputs; this script runs
# the same commands in order, writes every line of stdout and stderr to one log
# as it goes, and shows it on the terminal too.
#
# It does no measurement of its own: each step is the command from the README,
# unchanged (the server step is tools/bench-serve.sh), and any failure stops the
# run with a non-zero status.
#
# The SonicBoom side is the default build (CUDA 13.0; 12.0 is the minimum
# supported toolkit) and the llama.cpp side is that commit's build-cuda13, so
# both arms are built with the same toolkit.
#
# Usage: tools/community-bench.sh <model.gguf> <llama.cpp/build/bin>
#                                 [--runs N] [--no-llama] [--out FILE] [--serve]
#                                 [--toolcall]
#        <llama.cpp/build/bin> may be omitted with --no-llama.
#        --runs N   independent process runs per metric (default 3).
#        --no-llama measure SonicBoom alone: skip the llama.cpp commit and pass
#                   --no-llama to bench.sh (every llama.cpp cell reads skipped).
#        --out FILE write the log here (default sonicboom-bench-<datetime>.log).
#        --serve    add a step, tools/bench-serve.sh <model> --runs N
#                   (the server's streamed code-edit rate).
#        --toolcall add a step, tools/bench-toolcall.sh <model> --runs N
#                   (constrained JSON tool calls against llama-server).
# Env:   CUDA_VISIBLE_DEVICES defaults to 0, as in the README block.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS=3
NO_LLAMA=0
GGUF=""
LLAMA=""
OUT=""
SERVE=0
TOOLCALL=0

usage() {
  sed -n '/^# Usage:/,/^#$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="${2:-}"; shift 2 ;;
    --no-llama) NO_LLAMA=1; shift ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --serve) SERVE=1; shift ;;
    --toolcall) TOOLCALL=1; shift ;;
    -h|--help) usage ;;
    -*) echo "community-bench: unknown option $1" >&2; usage ;;
    *)
      if [ -z "$GGUF" ]; then GGUF="$1"
      elif [ -z "$LLAMA" ]; then LLAMA="$1"
      else echo "community-bench: unexpected argument $1" >&2; usage
      fi
      shift ;;
  esac
done

case "$RUNS" in
  ''|*[!0-9]*) echo "community-bench: --runs must be a positive integer" >&2
               exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "community-bench: --runs must be >= 1" >&2; exit 2; }
[ -n "$GGUF" ] || usage
[ -f "$GGUF" ] || { echo "community-bench: model not found: $GGUF" >&2; exit 2; }
GGUF="$(readlink -f "$GGUF")"

# Fail before any measurement: a run with llama.cpp needs a build directory
# with the two programs bench.sh calls.
if [ "$NO_LLAMA" = 0 ]; then
  [ -n "$LLAMA" ] || usage
  [ -d "$LLAMA" ] || { echo "community-bench: not a directory: $LLAMA" >&2; exit 2; }
  LLAMA="$(readlink -f "$LLAMA")"
  for prog in llama-bench llama-server; do
    [ -x "$LLAMA/$prog" ] ||
      { echo "community-bench: $LLAMA/$prog is missing or not executable" >&2
        exit 2; }
  done
elif [ -n "$LLAMA" ]; then
  [ -d "$LLAMA" ] ||
    { echo "community-bench: not a directory: $LLAMA" >&2; exit 2; }
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [ -z "$OUT" ]; then
  OUT="sonicboom-bench-$(date +%Y%m%d-%H%M%S).log"
fi
case "$OUT" in
  /*) ;;
  *) OUT="$PWD/$OUT" ;;
esac

# One log, in order, on the terminal at the same time.
exec > >(tee "$OUT") 2>&1

STEP=0
step() { # TITLE
  STEP=$((STEP + 1))
  echo
  echo "== $STEP/$TOTAL $1"
  echo
}

TOTAL=4
[ "$SERVE" = 1 ] && TOTAL=$((TOTAL + 1))
[ "$TOOLCALL" = 1 ] && TOTAL=$((TOTAL + 1))

step "hardware and build record"
echo "model:  $GGUF"
echo "runs:   $RUNS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if ! "$ROOT/tools/sb-report.sh" --model "$GGUF" --no-bench; then
  echo
  echo "community-bench: step 1 failed; the log is $OUT" >&2
  exit 1
fi

if [ "$NO_LLAMA" = 0 ]; then
  step "llama.cpp commit"
  COMMIT="$(git -C "$LLAMA" rev-parse --short HEAD 2>/dev/null || true)"
  echo "llama.cpp: $LLAMA"
  echo "commit:    ${COMMIT:-unknown}"
else
  step "llama.cpp commit (skipped: --no-llama)"
fi

LLAMA_ARGS=()
[ "$NO_LLAMA" = 1 ] && LLAMA_ARGS+=(--no-llama)

step "code-editing comparison"
if ! LLAMA_BIN="$LLAMA" "$ROOT/tools/bench.sh" "$GGUF" --runs "$RUNS" --spec \
     "${LLAMA_ARGS[@]}"; then
  echo
  echo "community-bench: step 3 failed; the log is $OUT" >&2
  exit 1
fi

step "extended-context tiers"
if ! LLAMA_BIN="$LLAMA" "$ROOT/tools/bench.sh" "$GGUF" --runs "$RUNS" --long \
     "${LLAMA_ARGS[@]}"; then
  echo
  echo "community-bench: step 4 failed; the log is $OUT" >&2
  exit 1
fi

if [ "$SERVE" = 1 ]; then
  step "server streaming"
  if ! "$ROOT/tools/bench-serve.sh" "$GGUF" --runs "$RUNS"; then
    echo
    echo "community-bench: the server-streaming step failed; the log is $OUT" >&2
    exit 1
  fi
fi

if [ "$TOOLCALL" = 1 ]; then
  step "constrained tool calls"
  if ! LLAMA_BIN="$LLAMA" "$ROOT/tools/bench-toolcall.sh" "$GGUF" --runs "$RUNS"; then
    echo
    echo "community-bench: the tool-call step failed; the log is $OUT" >&2
    exit 1
  fi
fi

echo
echo "== done"
echo
echo "log: $OUT"
echo
echo "Paste this file into the community benchmark issue."
