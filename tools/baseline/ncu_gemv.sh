#!/usr/bin/env bash
# ncu profile of the SonicBoom K-quant gemv kernels, to decide whether the
# cooperative remap (gemv_q*_K_coop) is now compute-bound (dp4a viable) or still
# L1/LSU-bound (dp4a dead end, same as the one-thread-per-block finding).
#
# Usage:
#   ./ncu_gemv.sh          # profile gemv_q3_K_coop (ffn_down, bpr=40)
#   ./ncu_gemv.sh base     # profile gemv_q3_K_base (ffn_gate_up, bpr=10)
#
# Needs sudo for --clock-control base on GeForce (locks to base clock so the
# %-of-peak numbers are comparable across runs).
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="$ROOT/tools/baseline/gemv_bench"

KERNEL="gemv_q3_K_coop"                 # ffn_down, bpr=40 (the remap that won)
if [ "${1:-coop}" = "base" ]; then
  KERNEL="gemv_q3_K_base"               # ffn_gate_up, bpr=10
fi

METRICS="l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"

echo "profiling kernel: ${KERNEL}"
sudo ncu --clock-control base \
  --kernel-name "regex:${KERNEL}" \
  --launch-skip 5 --launch-count 3 \
  --metrics "$METRICS" \
  "$BIN"
