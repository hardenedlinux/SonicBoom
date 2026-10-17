#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# ncu profile of the SonicBoom q8_1 K-quant gemv kernels (the default decode
# path, 64% of GPU time), to decide whether the mmvq port is now DRAM-bound
# (matmul floor -> stop) or still L1/LSU-bound (shared-memory tiling is the
# remaining rewrite). Profiling the q8_1 kernels, NOT the fp32 base/coop kernels
# (see ncu_gemv.sh for those).
#
# Usage:
#   ./ncu_gemv_q8_1.sh          # gemv_q3_K_q8_1 (ffn_gate/up, bpr=10)
#   ./ncu_gemv_q8_1.sh q4       # gemv_q4_K_q8_1 (ffn_down, bpr=40)
#
# Needs sudo for --clock-control base on GeForce (locks to base clock so the
# %-of-peak numbers are comparable across runs). Without sudo, drop the flag:
#   sudo ncu --clock-control base ...   ->   ncu ...
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="$ROOT/tools/baseline/gemv_bench"

KERNEL="gemv_q3_K_q8_1"
if [ "${1:-q3}" = "q4" ]; then
  KERNEL="gemv_q4_K_q8_1"
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
