#!/usr/bin/env bash
# SonicBoom unified quality gate.
#
# Usage:
#   tools/quality/run.sh                 # default CPU gate (tests + sanitizers + static)
#   tools/quality/run.sh ci              # same checks, CI-oriented environment
#   tools/quality/run.sh static
#   tools/quality/run.sh sanitizers
#   tools/quality/run.sh tests
#   tools/quality/run.sh concurrency     # optional: TSan
#   tools/quality/run.sh memory          # optional: Valgrind Memcheck
#   tools/quality/run.sh cuda            # optional: CUDA Compute Sanitizer
#   tools/quality/run.sh all             # default + all optional checks
#
# Exit status is non-zero if any check FAILs. Optional checks that cannot run
# report SKIP and do not fail the gate; required checks that cannot run FAIL.

set -uo pipefail

SB_Q_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB_ROOT_TMP="$(cd "$SB_Q_DIR/../.." && pwd)"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

MODE="${1:-default}"
case "$MODE" in
  -h|--help|help) usage; exit 0 ;;
  default|ci|static|sanitizers|tests|concurrency|memory|cuda|all) ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

# CI mode differs only in reporting context; the checks themselves are shared.
export SB_CI=0
[ "$MODE" = "ci" ] && export SB_CI=1

SB_LOG_ROOT_TMP="${SONICBOOM_QUALITY_LOG_ROOT:-$SB_ROOT_TMP/.quality/logs}"
SB_LOG_DIR="$SB_LOG_ROOT_TMP/$(date +%Y%m%d-%H%M%S)-$MODE"
mkdir -p "$SB_LOG_DIR"
export SB_LOG_DIR

# shellcheck source=config.sh
source "$SB_Q_DIR/config.sh"

export SB_RESULTS_FILE="$SB_LOG_DIR/results.tsv"
: > "$SB_RESULTS_FILE"

echo "SonicBoom quality gate"
echo "  mode:     $MODE"
echo "  repo:     $SB_ROOT"
echo "  logs:     $SB_LOG_DIR"
echo "  jobs:     $SB_JOBS"
echo

sb_write_environment

case "$MODE" in
  default|ci) CHECKS=(tests sanitizers static) ;;
  static)     CHECKS=(static) ;;
  sanitizers) CHECKS=(sanitizers) ;;
  tests)      CHECKS=(tests) ;;
  concurrency) CHECKS=(concurrency) ;;
  memory)     CHECKS=(memory) ;;
  cuda)       CHECKS=(cuda) ;;
  all)        CHECKS=(tests sanitizers static concurrency memory cuda) ;;
esac

overall=0
for check in "${CHECKS[@]}"; do
  echo
  echo "======================================================================"
  echo "== $check"
  echo "======================================================================"
  if ! bash "$SB_Q_DIR/checks/$check.sh"; then
    overall=1
  fi
done

echo
echo "======================================================================"
echo "== summary"
echo "======================================================================"
if [ -s "$SB_RESULTS_FILE" ]; then
  # Stable column layout; tab-separated input.
  awk -F'\t' '{ printf "%-4s  %-34s %s\n", $1, $2, $3 }' "$SB_RESULTS_FILE"
else
  echo "no results were recorded"
  overall=1
fi

if grep -q $'^FAIL\t' "$SB_RESULTS_FILE"; then
  overall=1
fi

echo
echo "logs: $SB_LOG_DIR"
if [ "$overall" -eq 0 ]; then
  echo "quality gate: PASS"
else
  echo "quality gate: FAIL"
fi
exit "$overall"
