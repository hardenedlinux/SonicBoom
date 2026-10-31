#!/usr/bin/env bash
# CPU test check: configure and build the normal (non-sanitized) CPU tree, then
# run the complete CTest suite. No test is skipped to obtain a green result.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=tests

if ! sb_required_tools cmake ctest ninja "$SB_CC" "$SB_CXX" guile; then
  sb_record FAIL "$CHECK/toolchain" "required tools missing (see stderr)"
  exit 1
fi
if ! sb_require_mlir; then
  sb_record FAIL "$CHECK/mlir" "MLIR prefix unavailable"
  exit 1
fi
if [ -z "${SB_GUILD:-}" ]; then
  sb_record FAIL "$CHECK/soniccross" "guild not found (required build-time dependency)"
  exit 1
fi

echo "configuring $SB_BASE_BUILD (CPU-only, $SB_BUILD_TYPE)"
if ! sb_configure "$SB_BASE_BUILD" "" "$SB_BUILD_TYPE" \
     > "$SB_LOG_DIR/tests-configure.log" 2>&1; then
  sb_record FAIL "$CHECK/configure" "see tests-configure.log"
  exit 1
fi
sb_record PASS "$CHECK/configure" "$SB_BASE_BUILD"

echo "building $SB_BASE_BUILD"
if ! sb_build "$SB_BASE_BUILD" "$SB_LOG_DIR/tests-build.log"; then
  sb_record FAIL "$CHECK/build" "see tests-build.log"
  exit 1
fi
sb_record PASS "$CHECK/build" "complete"

echo "running ctest"
if sb_ctest "$SB_BASE_BUILD" "$SB_LOG_DIR/tests-ctest.log" --no-tests=error; then
  summary="$(grep -E 'tests passed|tests failed' "$SB_LOG_DIR/tests-ctest.log" | tail -1)"
  sb_record PASS "$CHECK/ctest" "${summary:-all tests passed}"
  exit 0
fi
failed="$(grep -E '^\s*[0-9]+ - ' "$SB_LOG_DIR/tests-ctest.log" | head -10 | tr '\n' ' ')"
sb_record FAIL "$CHECK/ctest" "see tests-ctest.log${failed:+; failed: $failed}"
exit 1
