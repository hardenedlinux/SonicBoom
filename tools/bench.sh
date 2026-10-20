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
# why tg64 reads ~51.6 tok/s through this harness and not the ~47.5 tok/s a
# warm-up keyed on a different n_ctx used to report (the capture cost leaked into
# the depth-0 tail). pp512 reports pp_marginal (the per-token prefill cost, the
# llama-bench `-p n -n 0` semantics); pp_wall is a labelled legacy metric.
#
# A/B mode — compare two builds of libsonicboom.so on the same canonical set:
#
#   tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]
#                                [--no-control [--control-runs K]] [--k X]
#                                [--no-lock-clocks | --gpu-clock MHZ]
#
# BASE_LIB / CUR_LIB are libsonicboom.so paths (or the directories holding one).
# Each library is staged into its own directory and selected with
# LD_LIBRARY_PATH, so the same tools run against either. The two are run in
# *alternating* order (A,B on odd repetitions; B,A on even), so a slow drift in
# the machine — clocks, thermals, page cache — lands on both sides instead of on
# whichever library happens to run second. `--runs` is per side, so `--runs 3`
# is 6 runs a metric.
#
# Every row reports, per side, the median *and the sample standard deviation* of
# its runs, plus the SD of the paired per-repetition deltas. A delta is only
# meaningful next to that spread: a ±0.5% threshold is below this harness's own
# noise, so it is not used here. Instead the noise floor is *measured*: the same
# protocol with the *same* library on both sides, K pairs of it (`--control-runs`,
# default 5). The gate for a metric is
#
#     |median(control deltas)| + X · SD(control deltas)      (X = --k, default 2)
#
# and every real delta is reported against it. That control is on by default —
# a gate taken from a fixed number rather than from the harness's own spread is
# the thing it replaces — and it re-runs the whole metric set, so it roughly
# doubles the wall clock. `--no-control` skips it and reports spreads only.
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
# Workload mode — the two session traces, which the per-request metric set above
# cannot express:
#
#   tools/bench.sh <model.gguf> --workload [--runs N] [--no-build]
#
#   agent   a multi-turn session — a shared 1000-token system prompt, then eight
#           user turns of 57/312/140/33/401/96/210/75 tokens and 128 generated
#           tokens each, prefix cache on — reported as the median per-turn TTFT,
#           the median per-turn decode rate and the trace's wall time, beside
#           the same trace with a fresh session per turn (full re-prefill of the
#           whole history), which is the ratio prefix reuse is worth here.
#   coding  one 2048-token prompt and a 256-token generation: time to first
#           token, and the decode rate that follows it at that depth.
#
# Both are warmed up internally and run once per process; the table reports the
# median ± sd over --runs processes, the same statistic as the canonical rows.
# The text is the tool's built-in corpus, tokenized with the model's own
# tokenizer; tools/baseline/workload_bench.cpp takes --turns/--turn-tokens/
# --system/--gen/--prompt/--text for other shapes.
#
# Ablation mode — what each Boom IR pass group and each runtime feature is
# worth:
#
#   tools/bench.sh <model.gguf> --ablate [--qwen <model.gguf>]
#                                          [--runs N] [--no-build]
#
#   Runs the canonical set and workload A with everything enabled, then once per
#   group with that group disabled. Pass groups go in
#   SONICBOOM_IR_PASS_DISABLE (a pass plus the folds that only match the IR it
#   produces, or a single fold); runtime features go in
#   SONICBOOM_FEATURE_DISABLE (cuda_graph_decode, cuda_graph_prefill,
#   capture_reuse, split_kv, plan_cache, q8_1_dispatch, buffer_reuse, warmup —
#   see <sonicboom/runtime_features.h>). The table's contribution is the
#   fraction of the group-off value the group is worth: throughput is
#   (on - off) / off, latency (off - on) / off, so a positive number means the
#   group helps. Names that fail to parse are reported by the core and abort the
#   run rather than silently reading as "no effect".
#
#   With --qwen a second table is produced for that model, and when LLAMA_BENCH
#   is set each model's all-on canonical row is accompanied by the llama.cpp
#   value and the SonicBoom/llama.cpp ratio.
#
# Usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]
#        tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]
#                                             [--control | --no-control]
#                                             [--control-runs K] [--k X]
#                                             [--gpu-clock MHZ] [--no-lock-clocks]
#        tools/bench.sh <model.gguf> --workload [--runs N] [--no-build]
#        tools/bench.sh <model.gguf> --ablate [--qwen PATH] [--runs N] [--no-build]
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
CONTROL=1
CONTROL_RUNS=5
KSIGMA=2
CLOCKS=1
CLOCK_MHZ=""
WORKLOAD=0
ABLATE=0
QWEN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --runs)  RUNS="$2";  shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --control) CONTROL=1; shift ;;
    --no-control) CONTROL=0; shift ;;
    --control-runs) CONTROL_RUNS="$2"; shift 2 ;;
    --k) KSIGMA="$2"; shift 2 ;;
    --gpu-clock) CLOCK_MHZ="$2"; shift 2 ;;
    --no-lock-clocks) CLOCKS=0; shift ;;
    --workload) WORKLOAD=1; shift ;;
    --ablate) ABLATE=1; shift ;;
    --qwen)  QWEN="$2"; shift 2 ;;
    --ab)
      AB_BASE="${2:-}"; AB_CUR="${3:-}"
      { [ -n "$AB_BASE" ] && [ -n "$AB_CUR" ]; } || {
        echo "usage: --ab BASE_LIB CUR_LIB" >&2; exit 2; }
      shift 3 ;;
    -h|--help) sed -n '3,66p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  GGUF="$1"; shift ;;
  esac
done

# The counts drive the gates, so a non-numeric one is an error, not a default.
for pair in "runs:$RUNS" "control-runs:$CONTROL_RUNS"; do
  case "${pair#*:}" in
    ''|*[!0-9]*) echo "${pair%%:*} must be a positive integer (got '${pair#*:}')" >&2; exit 2 ;;
  esac
  [ "${pair#*:}" -ge 1 ] || { echo "${pair%%:*} must be >= 1" >&2; exit 2; }
done
awk -v k="$KSIGMA" 'BEGIN{exit !(k ~ /^[0-9]+(\.[0-9]+)?$/)}' ||
  { echo "--k must be a non-negative number (got '$KSIGMA')" >&2; exit 2; }

if [ -z "$GGUF" ]; then
  echo "usage: tools/bench.sh <model.gguf> [--label NAME] [--runs N] [--no-build]" >&2
  echo "       tools/bench.sh <model.gguf> --ab BASE_LIB CUR_LIB [--runs N] [--no-build]" >&2
  exit 2
fi
GGUF="$(readlink -f "$GGUF")"
[ -f "$GGUF" ] || { echo "model not found: $GGUF" >&2; exit 2; }
if [ -n "$QWEN" ]; then
  QWEN="$(readlink -f "$QWEN")"
  [ -f "$QWEN" ] || { echo "qwen model not found: $QWEN" >&2; exit 2; }
fi
[ -n "$LABEL" ] || LABEL="$(basename "$GGUF" .gguf)"

export LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SONICBOOM_GEMMA_GGUF="$GGUF"   # the loader entry point, arch-generic

if [ "$BUILD" = 1 ]; then
  for b in longctx_bench prefill_bench prefix_cache_bench workload_bench; do
    bash "$BASELINE/build_${b}.sh" >/dev/null
  done
fi

# --- measurement conditions -------------------------------------------------
# Pinned clocks, and a governor that is reported but never touched. The point of
# the pin is not that a fixed clock is the "real" clock but that it is the *same*
# clock across the two sides of a comparison; clocks move more than the effects
# being compared.

CLOCK_LOCKED=""
ab_tmp=""
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
  [ -n "$ab_tmp" ] && rm -rf "$ab_tmp"
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

# Sample standard deviation (n-1). "nan" below two samples, where a spread is
# not defined rather than zero — the distinction matters, because the A/B gate
# below is derived from it and a fake 0 would make the gate a point estimate.
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
  mkdir -p "$ab_tmp/base" "$ab_tmp/cur"
  cp -L "$base_so" "$ab_tmp/base/libsonicboom.so"
  cp -L "$cur_so"  "$ab_tmp/cur/libsonicboom.so"
  # The control measures the base library against itself; a second directory
  # rather than a second path, so the two sides are staged exactly like a real
  # comparison (two files, two LD_LIBRARY_PATH roots).
  mkdir -p "$ab_tmp/ctrl"
  cp -L "$base_so" "$ab_tmp/ctrl/libsonicboom.so"

  ab_run() { # SIDE KEY CMD... ; one run against SIDE, KEY's RESULT value or ""
    local side=$1 key=$2; shift 2
    LD_LIBRARY_PATH="$ab_tmp/$side:$ROOT/build/core" "$@" 2>/dev/null \
      | grep '^RESULT ' | res_field "$key" || true
  }

  # Alternating order (A,B on odd repetitions; B,A on even), so a slow drift in
  # the machine lands on both sides. Sets MA / MB / MD:
  #   MA, MB  "median stdev" per side       MD  "median stdev" of the paired
  #                                             per-repetition deltas, in %
  ab_pairs() { # SIDE_A SIDE_B REPS KEY CMD...
    local sa=$1 sb=$2 reps=$3 key=$4; shift 4
    local a_vals="" b_vals="" d_vals="" va vb d
    for ((i=1;i<=reps;i++)); do
      if (( i % 2 )); then
        va=$(ab_run "$sa" "$key" "$@"); vb=$(ab_run "$sb" "$key" "$@")
      else
        vb=$(ab_run "$sb" "$key" "$@"); va=$(ab_run "$sa" "$key" "$@")
      fi
      [ -n "$va" ] && a_vals="$a_vals$va\n"
      [ -n "$vb" ] && b_vals="$b_vals$vb\n"
      d=$(awk -v a="$va" -v b="$vb" 'BEGIN{
        if(a==""||b==""||a+0==0) exit 1; printf "%.4f", (b-a)/a*100}' 2>/dev/null) || d=""
      [ -n "$d" ] && d_vals="$d_vals$d\n"
    done
    MA="$(printf '%b' "$a_vals" | stats)"
    MB="$(printf '%b' "$b_vals" | stats)"
    MD="$(printf '%b' "$d_vals" | stats)"
  }

  delta_pct() { # BASE CUR -> "+1.23%" / "-0.45%" / "n/a"
    awk -v a="$1" -v b="$2" 'BEGIN{
      if(a=="nan"||b=="nan"||a=="n/a"||b=="n/a"||a+0==0){print "n/a"}
      else {printf "%+.2f%%", (b-a)/a*100}}'
  }

  # The gate a metric's delta is judged against, from the control's paired
  # deltas: |bias| + k·σ. The bias term matters — a harness that always reads the
  # second side 0.3% faster has a 0.3% floor, not a zero one. "n/a" when the
  # control could not produce a σ (fewer than two usable pairs).
  gate_pct() { # "median stdev" of the control deltas -> "0.42" / "n/a"
    awk -v m="${1%% *}" -v s="${1##* }" -v k="$KSIGMA" 'BEGIN{
      if(m=="nan"||s=="nan"){print "n/a"; exit}
      printf "%.3f", (m<0?-m:m) + k*s}'
  }

  # True when |delta| is inside the gate; "no gate" without a control.
  verdict() { # DELTA_STRING GATE -> one word for the table
    awk -v d="$1" -v g="$2" 'BEGIN{
      if(d=="n/a"||d=="nan"){print "n/a"; exit}
      if(g=="n/a"){print "no gate"; exit}
      v=substr(d,1,length(d)-1)+0; v=(v<0?-v:v)
      print (v<=g+0) ? "within" : "OUTSIDE"}'
  }

  ab_row() { # LABEL METRIC CMD...
    local label=$1 metric=$2; shift 2
    local gate="n/a" b_med b_sd c_med c_sd d_med d_sd
    if [ "$CONTROL" = 1 ]; then
      ab_pairs ctrl ctrl "$CONTROL_RUNS" "$metric" "$@"
      gate="$(gate_pct "$MD")"
    fi
    ab_pairs base cur "$RUNS" "$metric" "$@"
    b_med="${MA%% *}"; b_sd="${MA##* }"
    c_med="${MB%% *}"; c_sd="${MB##* }"
    d_med="${MD%% *}"; d_sd="${MD##* }"
    local dpct; dpct="$(delta_pct "$b_med" "$c_med")"
    printf '| %-14s | %20s | %20s | %8s | %7s | %-9s |\n' "$label" \
      "$(pm "$b_med" "$b_sd")" "$(pm "$c_med" "$c_sd")" "$dpct" \
      "$(printf '%s' "$d_sd" | awk '{print ($1=="nan")?"n/a":sprintf("%.2f%%",$1)}')" \
      "$(verdict "$dpct" "$gate")"
    ROW_GATES="$ROW_GATES$label=$gate "
  }

  echo
  echo "==== $LABEL (A/B) ===="
  echo "base: $base_so"
  echo "cur:  $cur_so"
  echo "median ± sd of $RUNS runs per side, alternating order; CUDA spine"
  if [ "$CONTROL" = 1 ]; then
    echo "control: $CONTROL_RUNS pairs of the base library against itself"
    echo "gate:    |median(control delta)| + ${KSIGMA}σ(control delta), per metric"
  else
    echo "gate:    none (--no-control) — spreads only, no verdict"
  fi
  echo
  echo "| metric         |        base ± sd   |         cur ± sd   |    delta |  Δ sd   | verdict   |"
  echo "|----------------|--------------------|--------------------|----------|---------|-----------|"
  ROW_GATES=""
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
  echo "gates: ${ROW_GATES% }"
  echo "RESULT ab model=$LABEL base_so=$base_so cur_so=$cur_so control=$CONTROL"
  exit 0
fi

# --- workload mode ---------------------------------------------------------
# The canonical set measures one request shape at a time; a session is a
# *sequence* of them, and what a session spends between its requests is what
# this mode exists to show. Four traces (tools/baseline/workload_bench.cpp):
#
#   agent   a multi-turn session whose prompt grows every turn (1000-token
#           system prompt, then eight turns of 57/312/140/33/401/96/210/75 user
#           tokens and 128 generated each) — a turn's prefill is the cost of the
#           tokens it adds, not of the history, when the prefix cache serves it
#           — reported beside the same trace with a fresh session per turn (full
#           re-prefill), which is the ratio prefix reuse is worth at this depth.
#   coding  one 2048-token prompt and a 256-token generation: time to first
#           token, and the decode rate that follows at that depth.
#   codeedit six independent edit requests, each pasting a 150-400 line repo
#           source file and generating the file's own token count back — the
#           copy-heavy trace speculative drafting is aimed at (B8).
#   chat    eight short unrelated user turns and 96 generated tokens each —
#           free-form text, where the drafter rarely fires, so speculation must
#           not cost anything (B8).
#
# Each is RUNS independent processes (each warmed up internally, --reps 1); the
# reported value is their median and sample sd, the same statistic every other
# row in this script reports.
if [ "$WORKLOAD" = 1 ]; then
  WB="$BASELINE/workload_bench"
  [ -x "$WB" ] || { echo "workload_bench not built: $WB" >&2; exit 2; }

  # A trace that produced no RESULT line (a refused GGUF, a failed run — the tool
  # writes why to stderr, which is left unredirected so the reason is visible)
  # must not turn into a table of empty cells: the first one to do it marks the
  # run, and the mode aborts once the subshells have finished.
  WL_FAIL="/tmp/wl-fail.$$"
  rm -f "$WL_FAIL"
  wl_stats() { # MODE FIELD-ARGS... ; "<median> <sd>" of RUNS runs
    local mode=$1 key=$2; shift 2
    local vals="" v
    for ((i=1;i<=RUNS;i++)); do
      v=$("$WB" --workload "$mode" "$@" | grep '^RESULT ' | res_field "$key") \
        || v=""
      [ -n "$v" ] && vals="$vals$v\n"
    done
    if [ -z "$vals" ]; then
      echo "workload trace '$mode' produced no result (reason above)" >&2
      : > "$WL_FAIL"
      echo "0 0"
      return
    fi
    printf '%b' "$vals" | stats
  }

  a_ttft=$(wl_stats agent ttft_ms)
  a_dec=$(wl_stats agent decode_tok_s)
  a_turn=$(wl_stats agent turn_ms)
  a_trace=$(wl_stats agent trace_ms)
  f_ttft=$(wl_stats agent ttft_ms --fresh)
  f_trace=$(wl_stats agent trace_ms --fresh)
  c_ttft=$(wl_stats coding ttft_ms)
  c_dec=$(wl_stats coding decode_tok_s)
  e_ttft=$(wl_stats codeedit ttft_ms)
  e_dec=$(wl_stats codeedit decode_tok_s)
  e_wall=$(wl_stats codeedit trace_ms)
  h_ttft=$(wl_stats chat ttft_ms)
  h_dec=$(wl_stats chat decode_tok_s)
  h_wall=$(wl_stats chat trace_ms)
  if [ -f "$WL_FAIL" ]; then rm -f "$WL_FAIL"; exit 2; fi
  rm -f "$WL_FAIL"

  echo
  echo "==== $LABEL ===="
  echo "workload traces, median ± sd of $RUNS runs, CUDA spine, whole model in VRAM"
  echo
  echo "| trace                        | value            | unit  |"
  echo "|------------------------------|------------------|-------|"
  printf '| %-28s | %16s | %5s |\n' "agent turn ttft (reuse)" \
    "$(pm "${a_ttft%% *}" "${a_ttft##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "agent turn decode (reuse)" \
    "$(pm "${a_dec%% *}" "${a_dec##* }")" "tok/s"
  printf '| %-28s | %16s | %5s |\n' "agent turn total (reuse)" \
    "$(pm "${a_turn%% *}" "${a_turn##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "agent trace wall (reuse)" \
    "$(pm "${a_trace%% *}" "${a_trace##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "agent turn ttft (fresh)" \
    "$(pm "${f_ttft%% *}" "${f_ttft##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "agent trace wall (fresh)" \
    "$(pm "${f_trace%% *}" "${f_trace##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "coding ttft (p2048)" \
    "$(pm "${c_ttft%% *}" "${c_ttft##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "coding decode" \
    "$(pm "${c_dec%% *}" "${c_dec##* }")" "tok/s"
  printf '| %-28s | %16s | %5s |\n' "code-edit turn ttft" \
    "$(pm "${e_ttft%% *}" "${e_ttft##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "code-edit turn decode" \
    "$(pm "${e_dec%% *}" "${e_dec##* }")" "tok/s"
  printf '| %-28s | %16s | %5s |\n' "code-edit trace wall" \
    "$(pm "${e_wall%% *}" "${e_wall##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "chat turn ttft" \
    "$(pm "${h_ttft%% *}" "${h_ttft##* }")" "ms"
  printf '| %-28s | %16s | %5s |\n' "chat decode" \
    "$(pm "${h_dec%% *}" "${h_dec##* }")" "tok/s"
  printf '| %-28s | %16s | %5s |\n' "chat trace wall" \
    "$(pm "${h_wall%% *}" "${h_wall##* }")" "ms"
  echo
  echo "RESULT workload model=$LABEL agent_ttft_ms=${a_ttft%% *} agent_decode_tok_s=${a_dec%% *} agent_turn_ms=${a_turn%% *} agent_trace_ms=${a_trace%% *} agent_fresh_ttft_ms=${f_ttft%% *} agent_fresh_trace_ms=${f_trace%% *} coding_ttft_ms=${c_ttft%% *} coding_decode_tok_s=${c_dec%% *} codeedit_ttft_ms=${e_ttft%% *} codeedit_decode_tok_s=${e_dec%% *} codeedit_trace_ms=${e_wall%% *} chat_ttft_ms=${h_ttft%% *} chat_decode_tok_s=${h_dec%% *} chat_trace_ms=${h_wall%% *}"
  exit 0
fi

# --- ablation mode ---------------------------------------------------------
# Three group families, all measured the same way:
#   pass groups   — "this pass and the folds it feeds": disabling
#                   QuantizeOncePass leaves the IR without the quantize nodes
#                   the quantize folds match, and the mmq folds' input pass is
#                   listed with them for the same reason.
#   feature groups — runtime switches (SONICBOOM_FEATURE_DISABLE): the decode /
#                   prefill CUDA graphs, capture reuse, split-KV attention, the
#                   plan cache, the q8_1 dispatch, device-buffer reuse, and the
#                   one-time warm-up.
#   switch groups  — a single env switch that is not a FEATURE_DISABLE bit
#                   (e.g. SONICBOOM_PREFIX_ALIGN=0).
if [ "$ABLATE" = 1 ]; then
  ABL_PASS_GROUPS=(
    "quantize-once+folds|QuantizeOncePass,RmsNormQuantizePass,GeluMulQuantizePass,SiluMulQuantizePass"
    "bias-into-gemv|BiasIntoGemvPass"
    "residual-add-into-gemv|ResidualAddIntoGemvPass"
    "mmq-folds|RmsNormMmqPass,GeluMulMmqPass,SiluMulMmqPass"
    "add-rms-norm-fusion|FuseAddRmsNormMmqPass"
    "rms-norm-add-fusion|FuseRmsNormAddPass"
    "heads-norm-rope-cast-fusion|FuseHeadsNormRopeCastPass"
    # Stage 6d: the KV append folded into that region. Disabling it restores the
    # attention op's two kv_append launches per KV layer.
    "kv-append-into-prep|FuseKvAppendIntoPrepPass"
    "slice-scale-fusion|FuseSliceScalePass"
    "gelu-mul-fusion|FuseGeluMulPass"
    # The two epilogue folds: the scalar scale after a dense projection / after
    # a generated glue region becomes an attribute of that node. Disabling one
    # restores its standalone scale_kernel launches.
    "scale-into-dense|ScaleIntoDensePass"
    "fold-scale-into-glue|FoldScaleIntoGluePass"
    "dense-layout|DenseLayoutPropagationPass"
    # Stage 7: liveness-based buffer classes and in-place elementwise results.
    # Disabling it restores one buffer per activation and no in-place writes.
    "buffer-reuse-assign|BufferReusePass"
  )
  ABL_FEATURE_GROUPS=(
    "cuda-graph-decode|cuda_graph_decode"
    "cuda-graph-prefill|cuda_graph_prefill"
    "capture-reuse|capture_reuse"
    "split-kv|split_kv"
    "plan-cache|plan_cache"
    "q8_1-dispatch|q8_1_dispatch"
    "buffer-reuse|buffer_reuse"
    "glue-codegen|glue_codegen"
    "autotune|autotune"
    "warmup|warmup"
  )
  # switch groups — env switches that are not SONICBOOM_FEATURE_DISABLE bits.
  # The entry is the literal VAR=value exported while the group is "off".
  ABL_SWITCH_GROUPS=(
    "glue-cache|SONICBOOM_GLUE_CACHE=0"
    "prefix-align|SONICBOOM_PREFIX_ALIGN=0"
    # Default is tiled; the group-off side runs the split-K schedule, so a
    # positive contribution means selecting tiled was the right call.
    "mmq-decomp|SONICBOOM_MMQ_DECOMP=stream_k"
    # Default is the per-shape "auto" pick; the group-off side pins the
    # compiler-generated FA-2 kernel for every shape it covers (others fall
    # back to the native kernel), so a positive contribution means auto's
    # per-shape choice is ahead of codegen-everywhere.
    "attn-codegen|SONICBOOM_ATTN_IMPL=codegen"
    # Default is the per-shape "auto" pick; the group-off side pins the native
    # tiled kernel (the pre-3c behavior), so a positive contribution means the
    # measured selection is ahead of always-native.
    "attn-auto|SONICBOOM_ATTN_IMPL=own"
    # Default is the cuBLASLt bf16 tensor-core kernel with a per-shape cached
    # algorithm; the group-off side runs the cublasGemmEx default algorithm.
    "dense-lt|SONICBOOM_DENSE_LT=0"
    # Default is 1: the bf16 dense projection runs the compiler-generated
    # tensor-core GEMM when the per-GPU autotune prefers it, with the folded
    # epilogue. The group-off side runs the cuBLASLt path for every shape.
    "dense-codegen|SONICBOOM_DENSE_GEMM=0"
    # Default is 1 (the per-step replay loop: measured fastest, the decode step
    # is GPU-bound so chunking's host-round-trip savings are already hidden);
    # the group-off side captures k=4 steps into one graph (device-side token
    # feedback, one host round-trip per 4 tokens), so a positive contribution
    # means the per-step default is ahead.
    "decode-chunk|SONICBOOM_DECODE_CHUNK=4"
    # With SONICBOOM_SPEC=auto exported, the all-on arm is B7's per-step
    # draft-length policy and the group-off side pins the fixed cap
    # (SONICBOOM_SPEC=on), so a positive contribution means the policy is ahead
    # of always drafting max_drafts. Without SONICBOOM_SPEC=auto both arms are
    # the plain loop and the row is zero by construction.
    "spec-policy|SONICBOOM_SPEC=on"
  )

  abl_median() { # KEY CMD... -> median of RUNS values of KEY's RESULT field
    local key=$1; shift
    local vals="" v
    for ((i=1;i<=RUNS;i++)); do
      v=$("$@" 2>/dev/null | grep '^RESULT ' | res_field "$key") || v=""
      [ -n "$v" ] && vals="$vals$v\n"
    done
    printf '%b' "$vals" | median
  }

  # The agent trace prints both fields from one run; running it twice for the
  # two columns would double the probe's cost and compare two different traces.
  abl_agent_both() { # sets ABL_AT / ABL_AW: medians of ttft_ms / trace_ms
    local tv="" wv="" out a b
    for ((i=1;i<=RUNS;i++)); do
      out=$("$BASELINE/workload_bench" --workload agent 2>/dev/null \
              | grep '^RESULT ' || true)
      a=$(printf '%s\n' "$out" | res_field ttft_ms)
      b=$(printf '%s\n' "$out" | res_field trace_ms)
      [ -n "$a" ] && tv="$tv$a\n"
      [ -n "$b" ] && wv="$wv$b\n"
    done
    ABL_AT=$(printf '%b' "$tv" | median)
    ABL_AW=$(printf '%b' "$wv" | median)
  }

  abl_contrib() { # KIND ON OFF ; positive = the group is worth this share
    awk -v k="$1" -v on="$2" -v off="$3" 'BEGIN{
      if(on==""||off==""||on=="nan"||off=="nan"||off+0==0){print "n/a"; exit}
      v = (k=="tok") ? (on-off)/off*100 : (off-on)/off*100
      printf "%+.2f%%", v }'
  }

  # llama-bench for the current model: PROMPT GEN DEPTH -> median avg_ts over
  # RUNS invocations (each invocation itself repeats -r 3 times).
  abl_llama() {
    local p=$1 g=$2 d=$3
    if [ -z "${LLAMA_BENCH:-}" ] || [ ! -x "${LLAMA_BENCH}" ]; then
      echo nan; return
    fi
    local f vals="" v
    for ((i=1;i<=RUNS;i++)); do
      f=$(mktemp)
      "$LLAMA_BENCH" -m "$ABL_MODEL" -p "$p" -n "$g" -d "$d" -ngl 99 -r 3 -o csv \
        >"$f" 2>/dev/null || true
      v=$(awk -F, -v P="$p" -v G="$g" -v D="$d" '
        {gsub(/"/,"")}
        NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next}
        h["avg_ts"] && $(h["n_prompt"])==P && $(h["n_gen"])==G && $(h["n_depth"])==D \
          {print $(h["avg_ts"])}
      ' "$f")
      rm -f "$f"
      [ -n "$v" ] && vals="$vals$v\n"
    done
    printf '%b' "$vals" | median
  }

  TOKFILE=$(mktemp)
  seq 2 1000 | tr '\n' ' ' >"$TOKFILE"
  trap 'rm -f "$TOKFILE"; cleanup' EXIT

  abl_metric() { # METRIC -> value with the current pass/feature configuration
    case "$1" in
      tg64)  abl_median tok_s "$BASELINE/longctx_bench" --depth 0 --steps 64 ;;
      d512)  abl_median tok_s "$BASELINE/longctx_bench" --depth 512 --steps 64 ;;
      pp512) abl_median pp_marginal_tok_s "$BASELINE/prefill_bench" --prompt 512 --reps 3 ;;
      reuse) abl_median reuse_ms "$BASELINE/prefix_cache_bench" \
               --prefix 512 --new 64 --steps 4 --reps 3 --tokens-file "$TOKFILE" ;;
    esac
  }

  ABL_METRICS="tg64:tok d512:tok pp512:tok reuse:ms agent_ttft:ms agent_wall:ms"

  # One complete table for one model, with the current env as the "all-on" arm.
  ablate_one() { # MODEL LABEL
    local ABL_MODEL=$1 table_label=$2
    export SONICBOOM_GEMMA_GGUF="$ABL_MODEL"
    unset SONICBOOM_IR_PASS_DISABLE SONICBOOM_FEATURE_DISABLE
    unset SONICBOOM_GLUE_CACHE SONICBOOM_PREFIX_ALIGN SONICBOOM_MMQ_DECOMP

    local -A ABL_ON
    local spec m kind off g gname glist
    for spec in $ABL_METRICS; do
      case "${spec%%:*}" in agent_*) continue ;; esac
      m=${spec%%:*}
      ABL_ON[$m]=$(abl_metric "$m")
    done
    abl_agent_both; ABL_ON[agent_ttft]=$ABL_AT; ABL_ON[agent_wall]=$ABL_AW

    echo
    echo "==== $table_label ===="
    echo "ablation, ${RUNS} run(s) per cell, CUDA spine, all groups on vs off"
    if [ -n "${LLAMA_BENCH:-}" ] && [ -x "${LLAMA_BENCH}" ]; then
      local ll_tg64 ll_d512 ll_pp512
      ll_tg64=$(abl_llama 0 64 0)
      ll_d512=$(abl_llama 0 64 512)
      ll_pp512=$(abl_llama 512 0 0)
      echo
      echo "all-on canonical vs llama.cpp (-ngl 99):"
      echo
      echo "| metric         | SonicBoom          | llama.cpp      |  ratio |"
      echo "|----------------|--------------------|----------------|--------|"
      row "tg64 (depth 0)" "${ABL_ON[tg64]}" "nan" "$ll_tg64" \
          "$(ratio "${ABL_ON[tg64]}" "$ll_tg64")" "tok/s"
      row "d512" "${ABL_ON[d512]}" "nan" "$ll_d512" \
          "$(ratio "${ABL_ON[d512]}" "$ll_d512")" "tok/s"
      row "pp512" "${ABL_ON[pp512]}" "nan" "$ll_pp512" \
          "$(ratio "${ABL_ON[pp512]}" "$ll_pp512")" "tok/s"
    fi

    echo
    echo "| group                  | metric      |           all-on |      group-off | contribution |"
    echo "|------------------------|-------------|------------------|----------------|--------------|"
    for g in "${ABL_PASS_GROUPS[@]}"; do
      gname=${g%%|*}; glist=${g##*|}
      export SONICBOOM_IR_PASS_DISABLE="$glist"
      abl_agent_both
      for spec in $ABL_METRICS; do
        m=${spec%%:*}; kind=${spec##*:}
        case "$m" in
          agent_ttft) off=$ABL_AT ;;
          agent_wall) off=$ABL_AW ;;
          *) off=$(abl_metric "$m") ;;
        esac
        printf '| %-22s | %-11s | %16s | %14s | %12s |\n' "$gname" "$m" \
          "$(pm "${ABL_ON[$m]}" nan)" "$(pm "$off" nan)" \
          "$(abl_contrib "$kind" "${ABL_ON[$m]}" "$off")"
      done
      unset SONICBOOM_IR_PASS_DISABLE
    done
    for g in "${ABL_FEATURE_GROUPS[@]}"; do
      gname=${g%%|*}; glist=${g##*|}
      export SONICBOOM_FEATURE_DISABLE="$glist"
      abl_agent_both
      for spec in $ABL_METRICS; do
        m=${spec%%:*}; kind=${spec##*:}
        case "$m" in
          agent_ttft) off=$ABL_AT ;;
          agent_wall) off=$ABL_AW ;;
          *) off=$(abl_metric "$m") ;;
        esac
        printf '| %-22s | %-11s | %16s | %14s | %12s |\n' "$gname" "$m" \
          "$(pm "${ABL_ON[$m]}" nan)" "$(pm "$off" nan)" \
          "$(abl_contrib "$kind" "${ABL_ON[$m]}" "$off")"
      done
      unset SONICBOOM_FEATURE_DISABLE
    done
    for g in "${ABL_SWITCH_GROUPS[@]}"; do
      gname=${g%%|*}; glist=${g##*|}
      export "$glist"
      abl_agent_both
      for spec in $ABL_METRICS; do
        m=${spec%%:*}; kind=${spec##*:}
        case "$m" in
          agent_ttft) off=$ABL_AT ;;
          agent_wall) off=$ABL_AW ;;
          *) off=$(abl_metric "$m") ;;
        esac
        printf '| %-22s | %-11s | %16s | %14s | %12s |\n' "$gname" "$m" \
          "$(pm "${ABL_ON[$m]}" nan)" "$(pm "$off" nan)" \
          "$(abl_contrib "$kind" "${ABL_ON[$m]}" "$off")"
      done
      unset "${glist%%=*}"
    done
    echo
    echo "groups (contribution is the share of the group-off value the group is worth;"
    echo "        positive = the group helps, negative = it costs that share):"
    for g in "${ABL_PASS_GROUPS[@]}"; do
      printf '  pass     %-24s %s\n' "${g%%|*}" "${g##*|}"
    done
    for g in "${ABL_FEATURE_GROUPS[@]}"; do
      printf '  feature  %-24s %s\n' "${g%%|*}" "${g##*|}"
    done
    for g in "${ABL_SWITCH_GROUPS[@]}"; do
      printf '  switch   %-24s %s\n' "${g%%|*}" "${g##*|}"
    done
    echo "RESULT ablate model=$table_label runs=$RUNS $(for spec in $ABL_METRICS; do m=${spec%%:*}; printf '%s_on=%s ' "$m" "${ABL_ON[$m]}"; done)"
  }

  ablate_one "$GGUF" "$LABEL"
  if [ -n "$QWEN" ]; then
    ablate_one "$QWEN" "$(basename "$QWEN" .gguf)"
  fi
  exit 0
fi

# --- SonicBoom -------------------------------------------------------------
# Each metric reports the median of RUNS runs *and* their spread: the median
# alone cannot say whether a later comparison's difference is bigger than this
# harness's own repeatability (see --control in the header).
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
