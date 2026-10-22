#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# bench-image.sh — the canonical MOSS-Image benchmark.
#
# Measures the end-to-end pipeline (conditioning is precomputed and passed with
# --embeds, so the numbers are the DiT + VAE path): seconds per image and
# images per minute at each requested batch size, peak device memory, and the
# cold-start cost (the first process pays weight load, Q8_0 quantization and
# upload; later runs are the steady state). Every measurement is one independent
# process run and the reported number is the median over `--runs` runs, with the
# spread beside it. Each run ends with one machine-readable
# `RESULT sample batch=... steps=... forwards=... load_s=... sample_s=...
# decode_s=... total_s=...` line, which is what this script parses.
#
# --verify adds the determinism checks the M2 gate requires: the same seed twice
# produces byte-identical latents, and a batch-4 run's first image equals the
# batch-1 image for the same seed (batch invariance).
#
# The SM clock is sampled for the duration and reported on exit (pinning needs
# root; see tools/bench.sh for the same convention). Device memory is sampled
# four times a second inside each timed run; the reported peak is the maximum
# device `memory.used` observed, idle baseline included, so it is a whole-device
# number and comparable across runs on a quiet machine.
#
# Usage: tools/bench-image.sh --model DIR --embeds FILE --neg-embeds FILE
#          [--sizes 512x512,1024x1024] [--batches 1,4] [--steps 20]
#          [--runs 3] [--guidance 3] [--seed 42] [--verify] [--no-build]
#          [--out DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOL="$ROOT/tools/sb-moss"
MODEL=""
EMBEDS=""
NEG_EMBEDS=""
SIZES="512x512"
BATCHES="1,4"
STEPS=20
GUIDANCE=3
SEED=42
RUNS=3
VERIFY=0
BUILD=1
OUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --embeds) EMBEDS="$2"; shift 2 ;;
    --neg-embeds) NEG_EMBEDS="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --batches) BATCHES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --verify) VERIFY=1; shift ;;
    --no-build) BUILD=0; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help) awk '/^# SPDX/{f=1;next} /^set -euo pipefail/{f=0} f' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$RUNS" in
  ''|*[!0-9]*) echo "runs must be a positive integer (got '$RUNS')" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "runs must be >= 1" >&2; exit 2; }
if [ -z "$MODEL" ] || [ -z "$EMBEDS" ] || [ -z "$NEG_EMBEDS" ]; then
  echo "usage: tools/bench-image.sh --model DIR --embeds FILE --neg-embeds FILE" >&2
  exit 2
fi
MODEL="$(readlink -f "$MODEL")"
EMBEDS="$(readlink -f "$EMBEDS")"
NEG_EMBEDS="$(readlink -f "$NEG_EMBEDS")"
[ -d "$MODEL" ] || { echo "model dir not found: $MODEL" >&2; exit 2; }
[ -f "$EMBEDS" ] || { echo "embeds not found: $EMBEDS" >&2; exit 2; }
[ -f "$NEG_EMBEDS" ] || { echo "negative embeds not found: $NEG_EMBEDS" >&2; exit 2; }

export LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "$BUILD" = 1 ]; then
  bash "$ROOT/tools/build_sb_moss.sh" >/dev/null
fi
[ -x "$TOOL" ] || { echo "sb-moss not built: $TOOL" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sb-bench-image.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  REPORT="$OUT_DIR/bench-image.md"
  : > "$REPORT"
else
  REPORT=/dev/null
fi

say() { echo "$@" | tee -a "$REPORT"; }

say "cpu governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
idle_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' | tr -d ' ')"
say "device memory idle: ${idle_mib:-?} MiB"

# One run of the pipeline at a fixed size/batch. Prints the RESULT line and the
# peak device memory observed while the process ran to $WORK/peak.
run_once() { # height width batch out_tag
  local h="$1" w="$2" b="$3" tag="$4"
  local img="$WORK/$tag.img"
  local peak="$WORK/$tag.peak"
  rm -f "$peak"
  local sampler=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 \
        > "$peak.msg" 2>/dev/null ) &
    sampler=$!
  fi
  local line
  line="$("$TOOL" sample --model "$MODEL" --embeds "$EMBEDS" \
            --neg-embeds "$NEG_EMBEDS" --height "$h" --width "$w" \
            --steps "$STEPS" --guidance "$GUIDANCE" --batch "$b" \
            --seed "$SEED" --image-out "$img" 2>"$WORK/$tag.log" |
          grep '^RESULT ' || true)"
  if [ -n "$sampler" ]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    tr -d ' ' < "$peak.msg" | sort -n | sed -n '$p' > "$peak" || true
  fi
  if [ -z "$line" ]; then
    echo "run failed ($tag):" >&2
    tail -20 "$WORK/$tag.log" >&2
    exit 1
  fi
  echo "$line" > "$WORK/$tag.result"
  echo "${line#RESULT sample }"
}

field() { # file key
  sed -n "s/.*[ ]$2=\([^ ]*\).*/\1/p" "$1"
}

median() { sort -n | awk '{a[NR]=$1} END { if (NR%2) print a[(NR+1)/2]; else printf "%.6f", (a[NR/2]+a[NR/2+1])/2 }'; }
spread() { sort -n | awk '{a[NR]=$1} END { if (NR<2) {print 0; exit} s=0; m=0; for(i=1;i<=NR;i++) m+=a[i]; m/=NR; for(i=1;i<=NR;i++) s+=(a[i]-m)^2; printf "%.6f", sqrt(s/(NR-1)) }'; }

for size in ${SIZES//,/ }; do
  h="${size%x*}"; w="${size#*x}"
  say ""
  say "==== MOSS-Image ${h}x${w}, ${STEPS} steps, CFG ${GUIDANCE} ===="
  say ""
  say "| batch | runs | load s | gen s (median ± sd) | gen s/image | images/min | peak VRAM MiB | total s |"
  say "|-------|------|--------|---------------------|-------------|------------|---------------|---------|"
  for b in ${BATCHES//,/ }; do
    cold=""
    files=()
    for i in $(seq 1 "$RUNS"); do
      tag="s${h}b${b}r${i}"
      run_once "$h" "$w" "$b" "$tag" >/dev/null
      if [ -z "$cold" ]; then
        cold="$(field "$WORK/$tag.result" total_s)"
      fi
      files+=("$WORK/$tag.result")
    done
    load_s=$(for f in "${files[@]}"; do field "$f" load_s; done | sort -n)
    gen_s=$(for f in "${files[@]}"; do
              awk -v a="$(field "$f" sample_s)" -v d="$(field "$f" decode_s)" \
                'BEGIN {print a+d}'
            done | sort -n)
    total_s=$(for f in "${files[@]}"; do field "$f" total_s; done | sort -n)
    peak=$(for f in "${files[@]}"; do tr -d ' ' < "${f%.result}.peak" 2>/dev/null; done | sort -n | sed -n '$p')
    med_load=$(printf '%s\n' "$load_s" | median)
    med_gen=$(printf '%s\n' "$gen_s" | median)
    sd_gen=$(printf '%s\n' "$gen_s" | spread)
    med_total=$(printf '%s\n' "$total_s" | median)
    per_image=$(awk -v t="$med_gen" -v b="$b" 'BEGIN {printf "%.3f", t/b}')
    per_min=$(awk -v t="$med_gen" -v b="$b" 'BEGIN {printf "%.2f", 60*b/t}')
    say "| $b | $RUNS | $(printf '%.3f' "$med_load") | $(printf '%.3f' "$med_gen") ± $(printf '%.3f' "$sd_gen") | $per_image | $per_min | ${peak:-?} | $(printf '%.3f' "$med_total") (cold $(printf '%.3f' "$cold")) |"
    # Machine-readable form of the row, for tools/sb-report.sh and scripts.
    echo "RESULT image_bench size=${h}x${w} batch=$b runs=$RUNS" \
         "load_s=$(printf '%.3f' "$med_load")" \
         "gen_s=$(printf '%.3f' "$med_gen")" \
         "gen_sd=$(printf '%.3f' "$sd_gen")" \
         "image_s=$per_image images_min=$per_min peak_mib=${peak:-?}" \
         "total_s=$(printf '%.3f' "$med_total")" \
         "cold_s=$(printf '%.3f' "$cold")"
  done

  if [ "$VERIFY" = 1 ]; then
    say ""
    d1="$WORK/verify-a.img"; d2="$WORK/verify-b.img"; d4="$WORK/verify-b4.img"
    "$TOOL" sample --model "$MODEL" --embeds "$EMBEDS" --neg-embeds "$NEG_EMBEDS" \
      --height "$h" --width "$w" --steps "$STEPS" --guidance "$GUIDANCE" \
      --batch 1 --seed "$SEED" --image-out "$d1" >/dev/null 2>&1
    "$TOOL" sample --model "$MODEL" --embeds "$EMBEDS" --neg-embeds "$NEG_EMBEDS" \
      --height "$h" --width "$w" --steps "$STEPS" --guidance "$GUIDANCE" \
      --batch 1 --seed "$SEED" --image-out "$d2" >/dev/null 2>&1
    "$TOOL" sample --model "$MODEL" --embeds "$EMBEDS" --neg-embeds "$NEG_EMBEDS" \
      --height "$h" --width "$w" --steps "$STEPS" --guidance "$GUIDANCE" \
      --batch 4 --seed "$SEED" --image-out "$d4" >/dev/null 2>&1
    if cmp -s "$d1" "$d2"; then det="identical"; else det="DIFFER"; fi
    payload=$((3 * h * w * 4))     # MOSIMG01: 40-byte header then image 0
    if cmp -s -n "$payload" -i 40:40 "$d1" "$d4"; then bi="identical"; else bi="DIFFER"; fi
    say "determinism (same seed, twice): $det; batch invariance (B=1 vs B=4 image 0): $bi"
  fi
done

# The clock the numbers were produced at, same convention as bench.sh.
if command -v nvidia-smi >/dev/null 2>&1; then
  say "SM clock now: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' | tr -d ' ') MHz"
fi
