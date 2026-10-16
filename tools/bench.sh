#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# bench.sh — the one canonical SonicBoom benchmark.
#
# Runs the canonical metric set for a GGUF and prints one table per model
# (SonicBoom, and llama.cpp + ratio when LLAMA_BENCH is set). Every metric is the
# median of `--runs` (default 3) independent process runs; each underlying tool
# ends its output with a single machine-readable `RESULT ...` line, which is what
# this script parses (never a free-text grep).
#
# Canonical set (CUDA spine, whole model resident on the GPU — llama.cpp -ngl 99):
#   tg64    decode 64 tokens from position 0        (llama-bench -p 0 -n 64 -d 0)
#   d512    decode 64 tokens at KV depth 512        (llama-bench -p 0 -n 64 -d 512)
#   pp512   prefill a 512-token prompt, marginal    (llama-bench -p 512 -n 0)
#   reuse   512-token prefix + 64-token continuation, first-token latency
#           (SonicBoom's prefix cache; llama-bench has no equivalent)
#
# The decode tools warm the exact (model, n_ctx) runtime before the timed region,
# so the one-time graph/plan build + CUDA-graph capture is never counted. That is
# why tg64 reads ~51.6 tok/s through this harness and not the ~47.5 tok/s a
# warm-up keyed on a different n_ctx used to report (the capture cost leaked into
# the depth-0 tail). pp512 reports pp_marginal (the per-token prefill cost, the
# llama-bench `-p n -n 0` semantics); pp_wall is a labelled legacy metric.
#
# A/B mode — compare two builds of libsonicboom.so on the same canonical set:
#
#   tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]
#
# BASE_LIB / CUR_LIB are libsonicboom.so paths (or the directories holding one).
# Each library is staged into its own directory and selected with
# LD_LIBRARY_PATH, so the same tools run against either. The two are run in
# *alternating* order (A,B on odd repetitions; B,A on even), so a slow drift in
# the machine — clocks, thermals, page cache — lands on both sides instead of on
# whichever library happens to run second, and the reported deltas are
# medians-of-N per side. `--runs` is per side, so `--runs 3` is 6 runs a metric.
#
# Usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]
#        tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]
# Env:   LLAMA_BENCH=/path/to/llama-bench   (optional; enables the llama.cpp column)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE="$ROOT/tools/baseline"
RUNS=3
BUILD=1
LABEL=""
GGUF=""
AB_BASE=""
AB_CUR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --runs)  RUNS="$2";  shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --ab)
      AB_BASE="${2:-}"; AB_CUR="${3:-}"
      { [ -n "$AB_BASE" ] && [ -n "$AB_CUR" ]; } || {
        echo "usage: --ab BASE_LIB CUR_LIB" >&2; exit 2; }
      shift 3 ;;
    -h|--help) sed -n '3,40p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  GGUF="$1"; shift ;;
  esac
done

if [ -z "$GGUF" ]; then
  echo "usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]" >&2
  echo "       tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]" >&2
  exit 2
fi
GGUF="$(readlink -f "$GGUF")"
[ -f "$GGUF" ] || { echo "model not found: $GGUF" >&2; exit 2; }
[ -n "$LABEL" ] || LABEL="$(basename "$GGUF" .gguf)"

export LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SONICBOOM_GEMMA_GGUF="$GGUF"   # the loader entry point, arch-generic

if [ "$BUILD" = 1 ]; then
  for b in longctx_bench prefill_bench prefix_cache_bench; do
    bash "$BASELINE/build_${b}.sh" >/dev/null
  done
fi

# --- helpers ---------------------------------------------------------------
res_field() { # KEY ; RESULT line on stdin -> value of KEY
  awk -v k="$1" '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]==k){print a[2]; exit}}}'
}

median() { # numbers on stdin -> median
  sort -n | awk '{a[NR]=$1} END{
    if(NR==0){print "nan"} else if(NR%2){print a[(NR+1)/2]} else {printf "%.3f", (a[NR/2]+a[NR/2+1])/2}}'
}

bench_runs() { # KEY CMD... ; median of RUNS process runs of CMD's RESULT KEY
  local key=$1; shift
  local vals="" v
  for ((i=1;i<=RUNS;i++)); do
    v=$("$@" 2>/dev/null | grep '^RESULT ' | res_field "$key") || v=""
    [ -n "$v" ] && vals="$vals$v\n"
  done
  printf '%b' "$vals" | median
}

ratio() { # SB LL -> "1.03x" or "n/a"
  awk -v a="$1" -v b="$2" 'BEGIN{
    if(a=="nan"||b=="nan"||b+0==0){print "n/a"} else {printf "%.2fx", a/b}}'
}

fmt() { # number|nan|n/a -> one decimal
  awk -v v="$1" 'BEGIN{ if(v=="nan"||v=="n/a") print v; else printf "%.1f", v }'
}

row() { # LABEL SB LL RATIO UNIT
  printf '| %-14s | %10s %-5s | %10s %-5s | %6s |\n' \
    "$1" "$(fmt "$2")" "$5" "$(fmt "$3")" "$5" "$4"
}

# --- A/B mode --------------------------------------------------------------
# Placed after the helpers so it reuses median/fmt/res_field, and after the build
# step so `--ab X build/core/libsonicboom.so` compares against the build just
# made rather than a stale copy of it.
if [ -n "$AB_BASE" ]; then
  resolve_lib() { # PATH -> absolute path to a libsonicboom.so
    local p="$1"
    [ -d "$p" ] && p="$p/libsonicboom.so"
    [ -f "$p" ] || { echo "not a libsonicboom.so: $1" >&2; exit 2; }
    readlink -f "$p"
  }
  base_so="$(resolve_lib "$AB_BASE")"
  cur_so="$(resolve_lib "$AB_CUR")"

  ab_tmp="$(mktemp -d)"
  trap 'rm -rf "$ab_tmp"' EXIT
  mkdir -p "$ab_tmp/base" "$ab_tmp/cur"
  cp -L "$base_so" "$ab_tmp/base/libsonicboom.so"
  cp -L "$cur_so"  "$ab_tmp/cur/libsonicboom.so"

  ab_run() { # SIDE KEY CMD... ; one run against SIDE, KEY's RESULT value or ""
    local side=$1 key=$2; shift 2
    LD_LIBRARY_PATH="$ab_tmp/$side:$ROOT/build/core" "$@" 2>/dev/null \
      | grep '^RESULT ' | res_field "$key" || true
  }

  ab_metric() { # KEY CMD... -> "<base median> <cur median>", alternating order
    local key=$1; shift
    local base_vals="" cur_vals="" v
    for ((i=1;i<=RUNS;i++)); do
      if (( i % 2 )); then
        v=$(ab_run base "$key" "$@"); [ -n "$v" ] && base_vals="$base_vals$v\n"
        v=$(ab_run cur  "$key" "$@"); [ -n "$v" ] && cur_vals="$cur_vals$v\n"
      else
        v=$(ab_run cur  "$key" "$@"); [ -n "$v" ] && cur_vals="$cur_vals$v\n"
        v=$(ab_run base "$key" "$@"); [ -n "$v" ] && base_vals="$base_vals$v\n"
      fi
    done
    printf '%s %s\n' "$(printf '%b' "$base_vals" | median)" \
                     "$(printf '%b' "$cur_vals" | median)"
  }

  delta_pct() { # BASE CUR -> "+1.23%" / "-0.45%" / "n/a"
    awk -v a="$1" -v b="$2" 'BEGIN{
      if(a=="nan"||b=="nan"||a+0==0){print "n/a"} else {printf "%+.2f%%", (b-a)/a*100}}'
  }

  ab_row() { # LABEL METRIC CMD...
    local label=$1 metric=$2; shift 2
    local pair base cur
    pair="$(ab_metric "$metric" "$@")"
    base="${pair%% *}"; cur="${pair##* }"
    printf '| %-14s | %12s | %12s | %8s |\n' "$label" "$(fmt "$base")" \
      "$(fmt "$cur")" "$(delta_pct "$base" "$cur")"
  }

  echo
  echo "==== $LABEL (A/B) ===="
  echo "base: $base_so"
  echo "cur:  $cur_so"
  echo "median of $RUNS runs per side, alternating order; CUDA spine"
  echo
  echo "| metric         |         base |          cur |    delta |"
  echo "|----------------|--------------|--------------|----------|"
  TOKFILE=$(mktemp)
  seq 2 1000 | tr '\n' ' ' >"$TOKFILE"
  ab_row "tg64 (depth 0)" tok_s "$BASELINE/longctx_bench" --depth 0 --steps 64
  ab_row "d512"           tok_s "$BASELINE/longctx_bench" --depth 512 --steps 64
  ab_row "pp512"          pp_marginal_tok_s "$BASELINE/prefill_bench" \
         --prompt 512 --reps 3
  ab_row "reuse 512+64"   reuse_ms "$BASELINE/prefix_cache_bench" \
         --prefix 512 --new 64 --steps 4 --reps 3 --tokens-file "$TOKFILE"
  rm -f "$TOKFILE"
  echo
  echo "RESULT ab model=$LABEL base_so=$base_so cur_so=$cur_so"
  exit 0
fi

# --- SonicBoom -------------------------------------------------------------
sb_tg64=$(bench_runs tok_s "$BASELINE/longctx_bench" --depth 0 --steps 64)
sb_d512=$(bench_runs tok_s "$BASELINE/longctx_bench" --depth 512 --steps 64)
sb_pp512=$(bench_runs pp_marginal_tok_s "$BASELINE/prefill_bench" --prompt 512 --reps 3)
# prefix_cache_bench needs prompt token ids that are valid for the model under
# test: real_prompt.tokens is a Gemma-vocab fixture, and feeding it to Qwen
# (vocab 151936) indexes past the embedding table (CUDA kernel failure). Token
# *values* do not affect any kernel cost here (the prompt is shape-only), so
# synthesize a deterministic in-vocab prompt — ids in [2, 1000) are below every
# supported model's vocab.
TOKFILE=$(mktemp)
seq 2 1000 | tr '\n' ' ' >"$TOKFILE"
sb_reuse=$(bench_runs reuse_ms "$BASELINE/prefix_cache_bench" \
             --prefix 512 --new 64 --steps 4 --reps 3 \
             --tokens-file "$TOKFILE")
rm -f "$TOKFILE"

# --- llama.cpp (optional) --------------------------------------------------
ll_tg64="nan"; ll_d512="nan"; ll_pp512="nan"
if [ -n "${LLAMA_BENCH:-}" ] && [ -x "${LLAMA_BENCH}" ]; then
  csvs=()
  for ((i=1;i<=RUNS;i++)); do
    f=$(mktemp)
    "$LLAMA_BENCH" -m "$GGUF" -p 0,512 -n 64,0 -d 0,512 -ngl 99 -r 3 -o csv \
      >"$f" 2>/dev/null || true
    csvs+=("$f")
  done
  lb_ts() { # PROMPT GEN DEPTH -> median avg_ts across the RUNS csv files
    local p=$1 g=$2 d=$3
    for f in "${csvs[@]}"; do
      awk -F, -v P="$p" -v G="$g" -v D="$d" '
        {gsub(/"/,"")}   # llama-bench CSV quotes every field
        NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next}
        h["avg_ts"] && $(h["n_prompt"])==P && $(h["n_gen"])==G && $(h["n_depth"])==D \
          {print $(h["avg_ts"])}
      ' "$f"
    done | median
  }
  ll_tg64=$(lb_ts 0 64 0)
  ll_d512=$(lb_ts 0 64 512)
  ll_pp512=$(lb_ts 512 0 0)
  rm -f "${csvs[@]}"
else
  [ -n "${LLAMA_BENCH:-}" ] && echo "note: LLAMA_BENCH=$LLAMA_BENCH is not executable; llama.cpp column skipped" >&2 || true
fi

# --- table -----------------------------------------------------------------
echo
echo "==== $LABEL ===="
echo "median of $RUNS runs, CUDA spine, whole model in VRAM (llama.cpp -ngl 99)"
echo
echo "| metric         | SonicBoom      | llama.cpp      |  ratio |"
echo "|----------------|----------------|----------------|--------|"
row "tg64 (depth 0)" "$sb_tg64" "$ll_tg64" "$(ratio "$sb_tg64" "$ll_tg64")" "tok/s"
row "d512"           "$sb_d512" "$ll_d512" "$(ratio "$sb_d512" "$ll_d512")" "tok/s"
row "pp512"          "$sb_pp512" "$ll_pp512" "$(ratio "$sb_pp512" "$ll_pp512")" "tok/s"
row "reuse 512+64"   "$sb_reuse" "n/a" "n/a" "ms"
echo
echo "RESULT bench model=$LABEL tg64=$sb_tg64 d512=$sb_d512 pp512=$sb_pp512 reuse_ms=$sb_reuse"
