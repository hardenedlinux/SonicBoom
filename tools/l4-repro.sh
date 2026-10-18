#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# L4 (sm_89) prefix-cache divergence reproduction.
#
# What this reproduces
# --------------------
# A prefix-cache reuse run must produce the same greedy tokens as a full
# prefill of the same tokens. The extended test_prefix_cache set walks the
# segment boundaries (4095/4096/4097/4098 and 8191/8192/8193 around the 512-token
# sliding-window chunk). On sm_86 the 8193 case used to diverge at step 6 when
# the prefix-reuse boundary was taken literally: a prefix of length 1 mod 512
# ends its seed prefill with a one-token final segment, which runs the
# single-column kernels instead of the batched ones, so the KV row written for
# that token differs (layer-0 V by ~3e-2; K bit-identical) from the same
# position computed inside a multi-token segment by a full prefill.
#
# The fix (gemma4_plan.cpp, generate_prefill_spine_cuda_prefix) holds that token
# back: the suffix starts on the chunk boundary and is segmented exactly like a
# full prefill. SONICBOOM_PREFIX_ALIGN=0 restores the old boundary, so this
# script can show the divergence and the fixed behaviour side by side. The
# isolate tool (`prefix_cache_isolate --prefix ... --new ...`) additionally
# prints the first differing KV element, the per-layer maxima, and a
# last-prefix-token breakdown that localizes the difference to the seed's final
# segment.
#
# Arch selections
# ---------------
# SonicBoom core has no __CUDA_ARCH__-conditional *numerics* selection that can
# be forced on an sm_86 device to emulate sm_89: the only device-attribute
# selections are the glue-JIT PTX target (compilation, not a runtime choice;
# compute_89 PTX cannot run on sm_86), the decode split-KV chunk count
# (sm_count*4/n_heads_kv, capped at 16 on both a 20-SM 3050 and a 58-SM L4),
# and a launch-grid cap (performance only). The L4 must therefore run the code
# natively; there is nothing meaningful to force.
#
# Usage: tools/l4-repro.sh [--no-build] [--build-dir DIR] [--model PATH]
# Env:   SONICBOOM_GEMMA_GGUF, SONICBOOM_ENABLE_MLIR, LLVM_DIR, MLIR_DIR
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=1
BUILD_DIR="${SONICBOOM_L4_BUILD_DIR:-$ROOT/build-l4}"
MODEL="${SONICBOOM_GEMMA_GGUF:-$ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf}"
TOKENS="$ROOT/tools/baseline/real_prompt.tokens"
RUN_DIR="${TMPDIR:-/tmp}/sb-l4-repro"

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '3,45p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_DIR"

# --- what we are running on -------------------------------------------------
echo "== host =="
printf '  %-14s %s\n' "git" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
printf '  %-14s %s\n' "build dir" "$BUILD_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r gname gdrv gmem gcap < <(nvidia-smi \
    --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null | head -n1)
  printf '  %-14s %s\n' "gpu" "${gname:-?} (${gmem:-?}, compute ${gcap:-?})"
  case "${gcap:-}" in
    8.9) ;;
    *) echo "  note: this script targets an L4 (sm_89); this device reports ${gcap:-unknown}" ;;
  esac
else
  echo "  no nvidia-smi"
fi
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 2; }
[ -f "$TOKENS" ] || { echo "tokens fixture not found: $TOKENS" >&2; exit 2; }

# --- build ------------------------------------------------------------------
TEST="$BUILD_DIR/tests/core/test_prefix_cache"
if [ "$BUILD" = 1 ]; then
  echo
  echo "== build =="
  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DSONICBOOM_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      ${SONICBOOM_ENABLE_MLIR:+-DSONICBOOM_ENABLE_MLIR="$SONICBOOM_ENABLE_MLIR"} \
      ${LLVM_DIR:+-DLLVM_DIR="$LLVM_DIR"} ${MLIR_DIR:+-DMLIR_DIR="$MLIR_DIR"} \
      > "$RUN_DIR/configure.log" 2>&1 \
      || { echo "configure failed (see $RUN_DIR/configure.log)"; tail -n 20 "$RUN_DIR/configure.log"; exit 1; }
  fi
  ninja -C "$BUILD_DIR" test_prefix_cache > "$RUN_DIR/build.log" 2>&1 \
    || { echo "build failed (see $RUN_DIR/build.log)"; tail -n 20 "$RUN_DIR/build.log"; exit 1; }
  echo "  ok ($TEST)"
fi
[ -x "$TEST" ] || { echo "test binary not found: $TEST (build it, or pass --build-dir)" >&2; exit 2; }

export SONICBOOM_GEMMA_GGUF="$MODEL" SONICBOOM_PREFIX_CACHE_TOKENS="$TOKENS"

# --- variant runs -----------------------------------------------------------
# Label | expected outcome | command. "pass" runs must pass; "diverge" runs are
# the pre-fix boundary and are only recorded (a "diverge" run that passes is a
# finding — the divergence is no longer reproducible — not a regression).
results=()
run_variant() {
  local label="$1" expect="$2"; shift 2
  local out="$RUN_DIR/${label// /_}.log"
  "$@" > "$out" 2>&1
  local rc=$?
  local detail
  detail="$(grep -E 'DIVERGE|FAIL|reuse identical' "$out" | tail -n 3 | tr '\n' ';' || true)"
  [ -n "$detail" ] || detail="$(tail -n 1 "$out" | cut -c1-120)"
  local verdict="ok"
  if [ "$expect" = pass ] && [ "$rc" != 0 ]; then verdict="REGRESSION"; fi
  if [ "$expect" = diverge ] && [ "$rc" = 0 ]; then verdict="expected-divergence-not-seen"; fi
  printf '  %-34s rc=%-2s %-26s %s\n' "$label" "$rc" "$verdict" "$detail"
  results+=("$label|$rc|$verdict")
}

echo
echo "== extended prefix set (32 continuation steps) =="
run_variant "aligned boundary"        pass    "$TEST"
run_variant "aligned boundary, 8K"    pass    "$TEST" --long
run_variant "pre-fix boundary, 8K"    diverge env SONICBOOM_PREFIX_ALIGN=0 "$TEST" --long
# With mmq_min_n=1 the one-token final segment stays on the batched MMQ path,
# so it matches the full prefill: this pass is the root-cause check, not an
# expected divergence.
run_variant "pre-fix, min_n=1, 8K"    pass    env SONICBOOM_PREFIX_ALIGN=0 SONICBOOM_MMQ_MIN_N=1 "$TEST" --long
run_variant "pre-fix, no autotune, 8K" diverge env SONICBOOM_PREFIX_ALIGN=0 \
  SONICBOOM_FEATURE_DISABLE=autotune "$TEST" --long

# --- KV localization --------------------------------------------------------
# The isolate tool reads the device KV buffers of a full prefill and of a
# prefix-reuse run and prints the first differing element plus the last-prefix
# -token breakdown. It needs the test build's libsonicboom.so.
echo
echo "== KV localization (isolate tool, 8 continuation steps) =="
if SONICBOOM_BUILD_DIR="$BUILD_DIR" "$ROOT/tools/baseline/build_prefix_cache_isolate.sh" \
     > "$RUN_DIR/isolate-build.log" 2>&1; then
  ISO="$ROOT/tools/baseline/prefix_cache_isolate"
  for p in 4097 8193; do
    for align in 1 0; do
      label="isolate p=$p align=$align"
      out="$RUN_DIR/${label// /_}.log"
      SONICBOOM_PREFIX_ALIGN="$align" \
        timeout 900 "$ISO" --prefix "$p" --new 100 --steps 8 \
                --tokens-file "$TOKENS" > "$out" 2>&1
      rc=$?
      grep -E 'tokens \(a full|tokens \(b reuse|a vs c|last prefix token|layer  0: last|layer 23: last' \
        "$out" | sed 's/^/    /'
      printf '    (%s rc=%s, log %s)\n' "$label" "$rc" "$out"
    done
  done
else
  echo "  isolate tool build failed (see $RUN_DIR/isolate-build.log); skipping"
fi

# --- summary ----------------------------------------------------------------
echo
echo "== summary =="
fail=0
for r in "${results[@]}"; do
  IFS='|' read -r label rc verdict <<< "$r"
  printf '  %-34s %-26s rc=%s\n' "$label" "$verdict" "$rc"
  [ "$verdict" = REGRESSION ] && fail=1
done
echo
if [ "$fail" = 0 ]; then
  echo "aligned-boundary runs pass; the pre-fix divergence is reproducible with SONICBOOM_PREFIX_ALIGN=0"
else
  echo "REGRESSION: an aligned-boundary run failed; see $RUN_DIR"
fi
exit "$fail"
