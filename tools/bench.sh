#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
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
# why tg64 reads the steady-state rate through this harness and not the lower
# number a warm-up keyed on a different n_ctx used to report (the capture cost
# leaked into the depth-0 tail). pp512 reports pp_marginal (the per-token prefill
# cost, the llama-bench `-p n -n 0` semantics); pp_wall is a labelled legacy
# metric.
#
# Measurement conditions: the GPU clock is pinned for the duration of the run
# (`nvidia-smi -lgc`, released on exit with `-rgc`) — an unpinned clock drifts by
# more than the effects being compared. `--gpu-clock MHZ` picks the value (the
# device's max graphics clock by default) and `--no-lock-clocks` skips it. Pinning
# needs root, so when it is not possible the run is *not* clock-homogeneous and
# the number this script can give instead is the clock it actually ran at: the SM
# clock is sampled once a second for the duration (plain `nvidia-smi`, no root)
# and reported as min/median/max on exit. The CPU governor is only *reported*;
# this script never changes a CPU setting.
#
# Every metric is the median of independent process runs, and the spread is
# reported beside it: the median alone cannot say whether a later comparison's
# difference is bigger than this harness's own repeatability.
#
# --long adds the extended-context tiers (M3): decode at depth 4K/8K/16K/32K and
# prefill of 4K/8K/16K/32K tokens, each against llama-bench at the same depth
# (`-d` / `-p`). It is opt-in because the deep tiers cost minutes per run. The
# decode tiers reach the depth with the chunked prefill path (longctx_bench
# --prefill) rather than decoding to it; the subtraction cancels the prefill,
# so the number is the steady decode rate with a depth-deep KV cache, the same
# thing llama-bench `-d` measures. Tiers whose n_ctx exceeds a model's declared
# context are reported as rejected by the runtime rather than as a number.
#
# Usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]
#                                       [--gpu-clock MHZ] [--no-lock-clocks]
#                                       [--long]
# Env:   LLAMA_BENCH=/path/to/llama-bench   (optional; enables the llama.cpp column)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE="$ROOT/tools/baseline"
RUNS=3
BUILD=1
LABEL=""
GGUF=""
CLOCKS=1
CLOCK_MHZ=""
LONG=0
LONG_DEPTHS="4096 8192 16384 32768"
LONG_STEPS=64

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --runs)  RUNS="$2";  shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --gpu-clock) CLOCK_MHZ="$2"; shift 2 ;;
    --no-lock-clocks) CLOCKS=0; shift ;;
    --long) LONG=1; shift ;;
    -h|--help) awk '/^# SPDX/{f=1;next} /^set -euo pipefail/{f=0} f' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  GGUF="$1"; shift ;;
  esac
done

# The run count drives the reported spread, so a non-numeric one is an error,
# not a default.
case "$RUNS" in
  ''|*[!0-9]*) echo "runs must be a positive integer (got '$RUNS')" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "runs must be >= 1" >&2; exit 2; }

if [ -z "$GGUF" ]; then
  echo "usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]" >&2
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

# --- measurement conditions -------------------------------------------------
# Pinned clocks, and a governor that is reported but never touched. The point of
# the pin is not that a fixed clock is the "real" clock but that it is the clock
# every run in the session reports at; clocks move more than the differences
# between sessions.

CLOCK_LOCKED=""
sm_samples=""
sm_sampler_pid=""

sm_clock_mhz() { # current SM clock in MHz, empty when unreadable
  # sed reads the whole stream, unlike `head -1`: under `set -o pipefail` an
  # early-exiting head turns nvidia-smi's SIGPIPE into a 141 status and, on a
  # Blackwell card (whose clocks.max.gr query prints a cosmetic warning), that
  # used to kill the script from the EXIT trap.
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null |
    sed -n '1p' | tr -d ' '
}

# One reading a second for the whole run, in the background. The sampler is a
# plain nvidia-smi query (no root, no device state change): the point is to have
# a record of the clock the numbers were produced at, which is what an unlocked
# run otherwise leaves unstated.
start_sm_sampler() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  [ -n "$(sm_clock_mhz)" ] || return 0
  sm_samples=$(mktemp "${TMPDIR:-/tmp}/sb-sm-clock.XXXXXX") || return 0
  (
    while :; do
      nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits \
        2>/dev/null >> "$sm_samples" || true
      sleep 1
    done
  ) &
  sm_sampler_pid=$!
}

report_sm_clock() {
  [ -n "$sm_sampler_pid" ] && kill "$sm_sampler_pid" 2>/dev/null || true
  [ -n "$sm_samples" ] || return 0
  # The last sample can land while the process is being torn down; it is one
  # reading, so it is dropped rather than averaged over.
  awk -F, '
    NR>1 { n++; s=$1+0; if (n==1 || s<min) min=s; if (n==1 || s>max) max=s
           v[n]=s; m=$2+0; if (n==1 || m<memmin) memmin=m; if (n==1 || m>memmax) memmax=m }
    END {
      if (n==0) exit
      for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (v[i]>v[j]) {t=v[i];v[i]=v[j];v[j]=t}
      printf "gpu clock: SM %d-%d MHz (median %d) over %d samples, mem %d-%d MHz\n",
             min, max, v[int((n+1)/2)], n, memmin, memmax
    }' "$sm_samples"
  rm -f "$sm_samples"
  sm_samples=""
}

restore_clocks() {
  report_sm_clock
  if [ -n "$CLOCK_LOCKED" ]; then
    nvidia-smi -rgc >/dev/null 2>&1 || true
    echo "gpu clock: released (nvidia-smi -rgc)" >&2
  fi
}
cleanup() {
  restore_clocks
  return 0
}
trap cleanup EXIT

lock_clocks() {
  if [ "$CLOCKS" != 1 ]; then
    echo "gpu clock: not pinned (--no-lock-clocks: readings are unlocked)"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu clock: no NVIDIA device visible — readings are unlocked"
    return 0
  fi
  if [ -z "$CLOCK_MHZ" ]; then
    # `|| true`: the script runs under `set -e`, and a driver that cannot
    # answer clocks.max.gr (or a non-root query that fails) must not abort the
    # whole benchmark — that failure mode is exactly what this branch handles.
    CLOCK_MHZ=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits \
                  2>/dev/null | sed -n '1p' | tr -d ' ' || true)
  fi
  case "$CLOCK_MHZ" in
    ''|*[!0-9]*)
      echo "gpu clock: could not read clocks.max.gr — readings are unlocked"
      return 0 ;;
  esac
  if nvidia-smi -lgc "$CLOCK_MHZ" >/dev/null 2>&1; then
    CLOCK_LOCKED=1
    echo "gpu clock: pinned to ${CLOCK_MHZ} MHz (nvidia-smi -lgc), released on exit"
  else
    echo "gpu clock: could not pin to ${CLOCK_MHZ} MHz — needs root, readings are unlocked"
    echo "           to pin it yourself: sudo nvidia-smi -lgc ${CLOCK_MHZ}"
    # Without the pin the "same clock on both sides" property is not something
    # this script enforced, so say what the clock actually is: what it reads at
    # this moment, and (at exit) the range it ran over. Both probes are
    # best-effort: `|| true` keeps `set -e` from turning an unreadable clock
    # into a failed run (the sampler below is the authoritative record).
    local now
    now=$(sm_clock_mhz || true)
    if [ -n "$now" ]; then
      echo "           SM clock now ${now} MHz (idle); min/median/max over the run reported at exit"
    else
      echo "           SM clock unreadable now; min/median/max over the run reported at exit"
    fi
  fi
}

report_governor() {
  local f=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor g
  if [ -r "$f" ]; then
    g=$(cat "$f")
    echo "cpu governor: $g"
    if [ "$g" != performance ]; then
      echo "           (not 'performance'; this script never changes it — if you want a"
      echo "            quieter comparison: sudo cpupower frequency-set -g performance)"
    fi
  else
    echo "cpu governor: no cpufreq interface — the CPU clock is governed by firmware"
  fi
}

lock_clocks
start_sm_sampler
report_governor

# --- helpers ---------------------------------------------------------------
res_field() { # KEY ; RESULT line on stdin -> value of KEY
  awk -v k="$1" '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]==k){print a[2]; exit}}}'
}

median() { # numbers on stdin -> median
  sort -n | awk '{a[NR]=$1} END{
    if(NR==0){print "nan"} else if(NR%2){print a[(NR+1)/2]} else {printf "%.3f", (a[NR/2]+a[NR/2+1])/2}}'
}

# Sample standard deviation (n-1). "nan" below two samples: a spread is not
# defined there rather than zero, and a fake 0 would read as perfect
# repeatability.
stdev() { # numbers on stdin -> standard deviation or "nan"
  sort -n | awk '{a[NR]=$1; s+=$1} END{
    if(NR<2){print "nan"; exit}
    m=s/NR; t=0
    for(i=1;i<=NR;i++) t+=(a[i]-m)^2
    printf "%.4f", sqrt(t/(NR-1))}'
}

stats() { # numbers on stdin -> "<median> <stdev>"
  local vals
  vals=$(cat)
  printf '%s %s' "$(printf '%s\n' "$vals" | median)" "$(printf '%s\n' "$vals" | stdev)"
}

# "52.5 ± 0.058" — a value with its spread, the form every row reports. %g, not
# a fixed number of decimals: the metrics differ by three orders of magnitude
# (52.5 tok/s against 1692 tok/s a step), and neither a fixed decimal count nor
# an exponent suits both.
pm() { # MEDIAN SD -> "M ± SD" / "n/a"
  awk -v m="$1" -v s="$2" 'BEGIN{
    if(m=="nan"||m=="n/a"){print "n/a"; exit}
    if(s=="nan"||s=="n/a"){printf "%g", m; exit}
    printf "%g ± %g", m, s}'
}

bench_stats() { # KEY CMD... ; "<median> <stdev>" of RUNS runs of CMD's RESULT KEY
  local key=$1; shift
  local vals="" v
  for ((i=1;i<=RUNS;i++)); do
    v=$("$@" 2>/dev/null | grep '^RESULT ' | res_field "$key") || v=""
    [ -n "$v" ] && vals="$vals$v\n"
  done
  printf '%b' "$vals" | stats
}

ratio() { # SB LL -> "1.03x" or "n/a"
  awk -v a="$1" -v b="$2" 'BEGIN{
    if(a=="nan"||b=="nan"||b+0==0){print "n/a"} else {printf "%.2fx", a/b}}'
}

fmt() { # number|nan|n/a -> one decimal
  awk -v v="$1" 'BEGIN{ if(v=="nan"||v=="n/a") print v; else printf "%.1f", v }'
}

row() { # LABEL SB_MEDIAN SB_SD LL RATIO UNIT
  printf '| %-14s | %18s %-5s | %10s %-5s | %6s |\n' \
    "$1" "$(pm "$2" "$3")" "$6" "$(fmt "$4")" "$6" "$5"
}

# --- SonicBoom -------------------------------------------------------------
# Each metric reports the median of RUNS runs *and* their spread: the median
# alone cannot say whether a later comparison's difference is bigger than this
# harness's own repeatability.
sb_tg64_stats=$(bench_stats tok_s "$BASELINE/longctx_bench" --depth 0 --steps 64)
sb_d512_stats=$(bench_stats tok_s "$BASELINE/longctx_bench" --depth 512 --steps 64)
sb_pp512_stats=$(bench_stats pp_marginal_tok_s "$BASELINE/prefill_bench" --prompt 512 --reps 3)
# prefix_cache_bench needs prompt token ids that are valid for the model under
# test: real_prompt.tokens is a Gemma-vocab fixture, and feeding it to Qwen
# (vocab 151936) indexes past the embedding table (CUDA kernel failure). Token
# *values* do not affect any kernel cost here (the prompt is shape-only), so
# synthesize a deterministic in-vocab prompt — ids in [2, 1000) are below every
# supported model's vocab.
TOKFILE=$(mktemp)
seq 2 1000 | tr '\n' ' ' >"$TOKFILE"
sb_reuse_stats=$(bench_stats reuse_ms "$BASELINE/prefix_cache_bench" \
                   --prefix 512 --new 64 --steps 4 --reps 3 \
                   --tokens-file "$TOKFILE")
rm -f "$TOKFILE"

sb_tg64="${sb_tg64_stats%% *}"; sb_tg64_sd="${sb_tg64_stats##* }"
sb_d512="${sb_d512_stats%% *}"; sb_d512_sd="${sb_d512_stats##* }"
sb_pp512="${sb_pp512_stats%% *}"; sb_pp512_sd="${sb_pp512_stats##* }"
sb_reuse="${sb_reuse_stats%% *}"; sb_reuse_sd="${sb_reuse_stats##* }"

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
echo "median ± sd of $RUNS runs, CUDA spine, whole model in VRAM (llama.cpp -ngl 99)"
echo
echo "| metric         | SonicBoom          | llama.cpp      |  ratio |"
echo "|----------------|--------------------|----------------|--------|"
row "tg64 (depth 0)" "$sb_tg64" "$sb_tg64_sd" "$ll_tg64" "$(ratio "$sb_tg64" "$ll_tg64")" "tok/s"
row "d512"           "$sb_d512" "$sb_d512_sd" "$ll_d512" "$(ratio "$sb_d512" "$ll_d512")" "tok/s"
row "pp512"          "$sb_pp512" "$sb_pp512_sd" "$ll_pp512" "$(ratio "$sb_pp512" "$ll_pp512")" "tok/s"
row "reuse 512+64"   "$sb_reuse" "$sb_reuse_sd" "n/a" "n/a" "ms"
echo
echo "RESULT bench model=$LABEL tg64=$sb_tg64 d512=$sb_d512 pp512=$sb_pp512 reuse_ms=$sb_reuse"

# --- extended-context tiers (--long) ---------------------------------------
# One llama-bench invocation per run with -r 1: the deep tiers are long enough
# that llama-bench's internal repeats would put a single cell at tens of
# minutes. The median over RUNS process runs is the reported value, as in the
# canonical table.
if [ "$LONG" = 1 ]; then
  lb_long_ts() { # PROMPT GEN DEPTH -> median avg_ts, or "" when llama-bench absent
    local p=$1 g=$2 d=$3
    [ -n "${LLAMA_BENCH:-}" ] && [ -x "${LLAMA_BENCH:-}" ] || return 0
    local vals="" i f
    for ((i=1;i<=RUNS;i++)); do
      f=$(mktemp)
      "$LLAMA_BENCH" -m "$GGUF" -p "$p" -n "$g" -d "$d" -ngl 99 -r 1 -o csv \
        >"$f" 2>/dev/null || true
      local v
      v=$(awk -F, -v P="$p" -v G="$g" -v D="$d" '
            {gsub(/"/,"")}
            NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next}
            h["avg_ts"] && $(h["n_prompt"])==P && $(h["n_gen"])==G && $(h["n_depth"])==D \
              {print $(h["avg_ts"])}
          ' "$f" | sed -n '1p')
      rm -f "$f"
      [ -n "$v" ] && vals="$vals$v\n"
    done
    printf '%b' "$vals" | median
  }

  echo
  echo "==== $LABEL (long context) ===="
  echo "median ± sd of $RUNS runs; decode = $LONG_STEPS tokens at depth"
  echo
  echo "| tier          | SonicBoom          | llama.cpp      |  ratio |"
  echo "|---------------|--------------------|----------------|--------|"
  for d in $LONG_DEPTHS; do
    st=$(bench_stats tok_s "$BASELINE/longctx_bench" --prefill --depth "$d" \
           --steps "$LONG_STEPS")
    sb=${st%% *}; sbsd=${st##* }
    ll=$(lb_long_ts 0 "$LONG_STEPS" "$d")
    row "d${d}" "$sb" "$sbsd" "${ll:-nan}" "$(ratio "$sb" "${ll:-nan}")" "tok/s"
    echo "RESULT bench_long model=$LABEL tier=d$d tok_s=$sb llama_tok_s=${ll:-nan}"
  done
  for n in $LONG_DEPTHS; do
    st=$(bench_stats pp_marginal_tok_s "$BASELINE/prefill_bench" --prompt "$n" \
           --reps 1)
    sb=${st%% *}; sbsd=${st##* }
    ll=$(lb_long_ts "$n" 0 0)
    row "pp${n}" "$sb" "$sbsd" "${ll:-nan}" "$(ratio "$sb" "${ll:-nan}")" "tok/s"
    echo "RESULT bench_long model=$LABEL tier=pp$n tok_s=$sb llama_tok_s=${ll:-nan}"
  done
fi
