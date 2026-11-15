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

BOOM_Q_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOM_ROOT_TMP="$(cd "$BOOM_Q_DIR/../.." && pwd)"

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
export BOOM_CI=0
[ "$MODE" = "ci" ] && export BOOM_CI=1

BOOM_LOG_ROOT_TMP="${SONICBOOM_QUALITY_LOG_ROOT:-$BOOM_ROOT_TMP/.quality/logs}"
BOOM_LOG_DIR="$BOOM_LOG_ROOT_TMP/$(date +%Y%m%d-%H%M%S)-$MODE"
mkdir -p "$BOOM_LOG_DIR"
export BOOM_LOG_DIR

# shellcheck source=config.sh
source "$BOOM_Q_DIR/config.sh"

export BOOM_RESULTS_FILE="$BOOM_LOG_DIR/results.tsv"
: > "$BOOM_RESULTS_FILE"

echo "SonicBoom quality gate"
echo "  mode:     $MODE"
echo "  repo:     $BOOM_ROOT"
echo "  logs:     $BOOM_LOG_DIR"
echo "  jobs:     $BOOM_JOBS"
echo

boom_write_environment

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
  if ! bash "$BOOM_Q_DIR/checks/$check.sh"; then
    overall=1
  fi
done

echo
echo "======================================================================"
echo "== summary"
echo "======================================================================"
if [ -s "$BOOM_RESULTS_FILE" ]; then
  # Stable column layout; tab-separated input.
  awk -F'\t' '{ printf "%-4s  %-34s %s\n", $1, $2, $3 }' "$BOOM_RESULTS_FILE"
else
  echo "no results were recorded"
  overall=1
fi

if grep -q $'^FAIL\t' "$BOOM_RESULTS_FILE"; then
  overall=1
fi

echo
echo "logs: $BOOM_LOG_DIR"
if [ "$overall" -eq 0 ]; then
  echo "quality gate: PASS"
else
  echo "quality gate: FAIL"
fi
exit "$overall"
