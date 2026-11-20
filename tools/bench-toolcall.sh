#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# The constrained tool-call benchmark: one JSON tool call per prompt, greedy,
# the same GGUF on both sides.
#
# Ten frozen prompts (tools/bench/toolcall) ask for a tool call; every reply is
# constrained to schema.json. SonicBoom decodes through its Guile C-ABI binding
# with the schema installed on the session; llama.cpp decodes through
# llama-server /completion with `json_schema`. Both are greedy (temperature 0,
# top_k 1), and both use whole-model-in-VRAM flags (llama.cpp -ngl 99), with
# the routed-expert offload tools/bench.sh uses for a model that does not fit.
#
# The SonicBoom side runs twice: once with its forced-token runs and gathered
# lm head on (the defaults), once with both ablations off
# (SONICBOOM_GRAMMAR_JUMP=0 SONICBOOM_GRAMMAR_GATHER=0).
#
# Usage:
#   tools/bench-toolcall.sh <model.gguf> [--runs N] [--port P]
#
# LLAMA_BIN (a llama.cpp build directory) or LLAMA_SERVER names llama-server;
# without it the llama.cpp side is skipped and the table says so. Runs must be
# serial on an idle GPU; the script takes /tmp/sb-bench.lock for the duration.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=3
PORT=18080
GGUF=""

while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) sed -n '3,30p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) GGUF="$1"; shift ;;
  esac
done

[ -n "$GGUF" ] || { echo "usage: tools/bench-toolcall.sh <model.gguf> [--runs N] [--port P]" >&2; exit 2; }
GGUF="$(readlink -f "$GGUF")"
[ -f "$GGUF" ] || { echo "model not found: $GGUF" >&2; exit 2; }
case "$RUNS" in ''|*[!0-9]*) echo "runs must be a positive integer" >&2; exit 2 ;; esac
[ "$RUNS" -ge 1 ] || { echo "runs must be >= 1" >&2; exit 2; }

LLAMA_SERVER="${LLAMA_SERVER:-}"
if [ -z "$LLAMA_SERVER" ] && [ -n "${LLAMA_BIN:-}" ]; then
  if [ -f "${LLAMA_BIN}" ]; then
    LLAMA_SERVER="$(dirname "$LLAMA_BIN")/llama-server"
  else
    LLAMA_SERVER="$LLAMA_BIN/llama-server"
  fi
fi
HAVE_LLAMA=0
if [ -n "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ]; then
  HAVE_LLAMA=1
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  exec 3>&-
  echo "bench-toolcall: port $PORT is already in use" >&2
  exit 2
fi

LIB="$ROOT/build/core/libsonicboom.so"
[ -f "$LIB" ] || { echo "bench-toolcall: $LIB is missing; build first" >&2; exit 1; }
DATA="$ROOT/tools/bench/toolcall"
CLIENT="$ROOT/tools/bench/toolcall.scm"

# The GPU is measured serially; a second benchmark must not overlap this one.
exec 9>"/tmp/sb-bench.lock"
flock 9

median() {
  awk '{ v[NR] = $1 } END {
         if (NR == 0) { print "nan"; exit }
         n = NR
         for (i = 1; i <= n; i++)
           for (j = i + 1; j <= n; j++)
             if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
         if (n % 2) print v[(n + 1) / 2]
         else printf "%.6g\n", (v[n / 2] + v[n / 2 + 1]) / 2
       }'
}

# expert_count is read from the GGUF key-value block (uint64 key length, key
# bytes, uint32 value type, value); a routed-expert model is offloaded the way
# tools/bench.sh does: the smallest --n-cpu-moe count llama-server allocates
# with, so the most weights stay in VRAM. Dense models add no argument.
gguf_u32() { # FILE KEY-SUFFIX
  local file=$1 suffix=$2 m off key klen ty
  m=$(LC_ALL=C grep -aboE "[a-z0-9_]+\.$suffix" "$file" 2>/dev/null |
      sed -n '1p') || true
  [ -n "$m" ] || return 0
  off=${m%%:*}; key=${m#*:}; klen=${#key}
  ty=$(od -An -t u4 -j $((off + klen)) -N 4 "$file" 2>/dev/null | tr -d ' ')
  [ "$ty" = 4 ] || return 0
  od -An -t u4 -j $((off + klen + 4)) -N 4 "$file" 2>/dev/null | tr -d ' '
}

server_healthy() {
  guile --no-auto-compile -c "
    (use-modules (web client) (web response) (web uri))
    (define code
      (catch #t
        (lambda ()
          (call-with-values
            (lambda () (http-get (build-uri 'http #:host \"127.0.0.1\"
                                            #:port $PORT #:path \"/health\")))
            (lambda (resp body) (response-code resp))))
        (lambda _ 0)))
    (exit (if (= code 200) 0 1))" >/dev/null 2>&1
}

# probe_moe L: 0 when llama-server loads and answers /health with that count.
probe_moe() { # L
  local l=$1 log pid waited=0
  log=$(mktemp "${TMPDIR:-/tmp}/boom-moe-probe.XXXXXX")
  "$LLAMA_SERVER" -m "$GGUF" -ngl 99 -fa on -np 1 --no-webui \
    -c 2048 -ncmoe "$l" --port "$PORT" >"$log" 2>&1 &
  pid=$!
  while [ "$waited" -lt 300 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$log"
      return 1
    fi
    if server_healthy; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      rm -f "$log"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
  return 1
}

MOE_ARGS=""
if [ "$HAVE_LLAMA" = 1 ]; then
  experts=$(gguf_u32 "$GGUF" expert_count)
  if [ -n "$experts" ] && [ "$experts" -gt 0 ]; then
    blocks=$(gguf_u32 "$GGUF" block_count)
    [ -n "$blocks" ] || blocks=$experts
    if "$LLAMA_SERVER" --help 2>&1 | grep -q -- '--n-cpu-moe'; then
      # The smallest count that loads leaves the most weights in VRAM.
      for ((l=0; l<=blocks; l+=4)); do
        if probe_moe "$l"; then
          MOE_ARGS="-ncmoe $l"
          break
        fi
      done
    elif "$LLAMA_SERVER" --help 2>&1 | grep -q -- '--override-tensor'; then
      MOE_ARGS='-ot \.ffn_(up|down|gate)_exps\.=CPU'
    fi
  fi
fi

echo "==== constrained JSON output, greedy, same GGUF: $(basename "$GGUF") ===="
echo "model: $GGUF"
echo "runs: $RUNS, prompts: 10, n_predict: 256, schema: $DATA/schema.json"
if [ "$HAVE_LLAMA" = 1 ]; then
  echo "llama-server: $LLAMA_SERVER${MOE_ARGS:+ ($MOE_ARGS)}"
else
  echo "llama-server: not found; the llama.cpp side is skipped"
fi
echo

WORK=$(mktemp -d "${TMPDIR:-/tmp}/boom-toolcall.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/boom" "$WORK/boom-ablate" "$WORK/llama"

run_side() { # SIDE OUTDIR [env assignments...]
  local side=$1 out=$2; shift 2
  echo "toolcall: SonicBoom side ($side), $RUNS reps" >&2
  env -u SONICBOOM_SPEC "$@" \
      LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      GUILE_AUTO_COMPILE=0 \
      guile --no-auto-compile -L "$ROOT/bindings/guile/scheme" "$CLIENT" \
        --side boom --model "$GGUF" --data "$DATA" --runs "$RUNS" \
        --out "$out" >"$out/run.txt"
}

run_side boom "$WORK/boom"
run_side boom-ablate "$WORK/boom-ablate" \
  SONICBOOM_GRAMMAR_JUMP=0 SONICBOOM_GRAMMAR_GATHER=0

if [ "$HAVE_LLAMA" = 1 ]; then
  echo "toolcall: llama-server side, $RUNS reps" >&2
  moe_opt=()
  [ -z "$MOE_ARGS" ] || moe_opt=(--moe-args "$MOE_ARGS")
  env LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      GUILE_AUTO_COMPILE=0 guile --no-auto-compile \
      -L "$ROOT/bindings/guile/scheme" \
    "$CLIENT" --side llama --model "$GGUF" --data "$DATA" --runs "$RUNS" \
    --out "$WORK/llama" --server "$LLAMA_SERVER" --port "$PORT" \
    "${moe_opt[@]}" >"$WORK/llama/run.txt"
fi

side_tok_s() { sed -n 's/^rep=[0-9]* side=[^ ]* tok_s=//p' "$1/run.txt" | median; }
side_field() { # FILE FIELD -> median
  sed -n "s/.* $2=\\([0-9.]*\\).*/\\1/p" "$1/run.txt" | median
}
side_valid() { # FILE -> "k/total"
  local k total
  k=$(grep -c 'valid=1$' "$1/run.txt" || true)
  total=$(grep -c '^turn=' "$1/run.txt" || true)
  echo "$k/$total"
}

compare_turns() { # DIR-A DIR-B PREFIX-A PREFIX-B -> "k/total"
  local a=$1 b=$2 pa=$3 pb=$4 k=0 total=0 r t fa fb
  for ((r=0; r<RUNS; r++)); do
    for t in 01 02 03 04 05 06 07 08 09 10; do
      fa="$a/$pa.rep$r.$t.txt"; fb="$b/$pb.rep$r.$t.txt"
      total=$((total + 1))
      if cmp -s "$fa" "$fb"; then k=$((k + 1)); fi
    done
  done
  echo "$k/$total"
}

ident_llama="—"
[ "$HAVE_LLAMA" = 1 ] && ident_llama=$(compare_turns "$WORK/boom" "$WORK/llama" boom llama)
ablate_ident="—"
[ "$HAVE_LLAMA" = 1 ] && ablate_ident=$(compare_turns "$WORK/boom-ablate" "$WORK/llama" boom llama)

printf '| side                        | tok/s   | prefill ms | tokens | valid JSON | identical to llama-server |\n'
printf '|-----------------------------|---------|------------|--------|------------|---------------------------|\n'
printf '| SonicBoom                   | %-7s | %-10s | %-6s | %-10s | %-25s |\n' \
  "$(side_tok_s "$WORK/boom")" "$(side_field "$WORK/boom" prefill_ms)" \
  "$(side_field "$WORK/boom" tokens)" "$(side_valid "$WORK/boom")" "$ident_llama"
printf '| SonicBoom gather+jump off   | %-7s | %-10s | %-6s | %-10s | %-25s |\n' \
  "$(side_tok_s "$WORK/boom-ablate")" "$(side_field "$WORK/boom-ablate" prefill_ms)" \
  "$(side_field "$WORK/boom-ablate" tokens)" "$(side_valid "$WORK/boom-ablate")" "$ablate_ident"
if [ "$HAVE_LLAMA" = 1 ]; then
  printf '| llama-server                | %-7s | %-10s | %-6s | %-10s | %-25s |\n' \
    "$(side_tok_s "$WORK/llama")" "$(side_field "$WORK/llama" prefill_ms)" \
    "$(side_field "$WORK/llama" tokens)" "$(side_valid "$WORK/llama")" "—"
else
  printf '| llama-server                | skipped |            |        |            |                           |\n'
fi

echo
echo "RESULT toolcall model=$(basename "$GGUF" .gguf) runs=$RUNS boom_tok_s=$(side_tok_s "$WORK/boom") boom_ablate_tok_s=$(side_tok_s "$WORK/boom-ablate") llama_tok_s=$( [ "$HAVE_LLAMA" = 1 ] && side_tok_s "$WORK/llama" || echo nan) identical=$ident_llama"
