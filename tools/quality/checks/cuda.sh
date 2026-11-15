#!/usr/bin/env bash
# Optional CUDA Compute Sanitizer check.
#
# This check is deliberately separate from the CPU gate and is never run by
# `run.sh ci` or the default mode. It requires:
#   * a CUDA device (nvidia-smi)
#   * compute-sanitizer
#   * an existing CUDA-enabled build tree supplied by the developer through
#     SONICBOOM_QUALITY_CUDA_BUILD (the gate never builds a CUDA tree)
#
# If any requirement is absent the check reports SKIP with the reason. It never
# reports PASS without running the tool.
#
# racecheck covers shared-memory hazards; it is not a general global-memory
# race detector.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=cuda

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  boom_record SKIP "$CHECK/gpu" "no usable CUDA device"
  exit 0
fi
if [ -z "${BOOM_COMPUTE_SANITIZER:-}" ] || [ ! -x "$BOOM_COMPUTE_SANITIZER" ]; then
  boom_record SKIP "$CHECK/tool" "compute-sanitizer not found"
  exit 0
fi
if [ -z "$BOOM_CUDA_BUILD" ] || [ ! -f "$BOOM_CUDA_BUILD/CMakeCache.txt" ]; then
  boom_record SKIP "$CHECK/build" \
    "set SONICBOOM_QUALITY_CUDA_BUILD to an existing CUDA-enabled build tree"
  exit 0
fi
if ! grep -q '^SONICBOOM_USE_CUDA:BOOL=ON' "$BOOM_CUDA_BUILD/CMakeCache.txt"; then
  boom_record SKIP "$CHECK/build" "build tree is not CUDA-enabled: $BOOM_CUDA_BUILD"
  exit 0
fi

# Default suites that exercise the CUDA boundary. A developer can override the
# list with SONICBOOM_QUALITY_CUDA_TESTS.
read -r -a tests <<< "${SONICBOOM_QUALITY_CUDA_TESTS:-test_backward_ops test_clip_grad_norm}"

export LD_LIBRARY_PATH="$BOOM_CUDA_BUILD/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
lock="${SONICBOOM_QUALITY_GPU_LOCK:-/tmp/sonicboom-quality-gpu.lock}"

found=0
rc=0
for t in "${tests[@]}"; do
  bin="$BOOM_CUDA_BUILD/tests/core/$t"
  if [ ! -x "$bin" ]; then
    boom_record SKIP "$CHECK/$t" "test binary not built: $bin"
    continue
  fi
  found=1
  for tool in memcheck initcheck synccheck racecheck; do
    log="$BOOM_LOG_DIR/cuda-$t-$tool.log"
    echo "-- compute-sanitizer --tool $tool $t"
    if flock "$lock" "$BOOM_COMPUTE_SANITIZER" --tool "$tool" \
         --error-exitcode 99 "$bin" > "$log" 2>&1; then
      boom_record PASS "$CHECK/$t/$tool" "no errors"
    else
      summary="$(grep -E 'ERROR SUMMARY|========= ERROR' "$log" | tail -2 | tr '\n' ' ')"
      boom_record FAIL "$CHECK/$t/$tool" "see cuda-$t-$tool.log${summary:+; $summary}"
      rc=1
    fi
  done
done

if [ "$found" -eq 0 ]; then
  boom_record SKIP "$CHECK/tests" "none of the configured CUDA tests were built"
  exit 0
fi
exit "$rc"
