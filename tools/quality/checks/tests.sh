#!/usr/bin/env bash
# CPU test check: configure and build the normal (non-sanitized) CPU tree, then
# run the complete CTest suite. No test is skipped to obtain a green result.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=tests

if ! boom_required_tools cmake ctest ninja "$BOOM_CC" "$BOOM_CXX" guile; then
  boom_record FAIL "$CHECK/toolchain" "required tools missing (see stderr)"
  exit 1
fi
if ! boom_require_mlir; then
  boom_record FAIL "$CHECK/mlir" "MLIR prefix unavailable"
  exit 1
fi
if [ -z "${BOOM_GUILD:-}" ]; then
  boom_record FAIL "$CHECK/soniccross" "guild not found (required build-time dependency)"
  exit 1
fi

echo "configuring $BOOM_BASE_BUILD (CPU-only, $BOOM_BUILD_TYPE)"
if ! boom_configure "$BOOM_BASE_BUILD" "" "$BOOM_BUILD_TYPE" \
     > "$BOOM_LOG_DIR/tests-configure.log" 2>&1; then
  boom_record FAIL "$CHECK/configure" "see tests-configure.log"
  exit 1
fi
boom_record PASS "$CHECK/configure" "$BOOM_BASE_BUILD"

echo "building $BOOM_BASE_BUILD"
if ! boom_build "$BOOM_BASE_BUILD" "$BOOM_LOG_DIR/tests-build.log"; then
  boom_record FAIL "$CHECK/build" "see tests-build.log"
  exit 1
fi
boom_record PASS "$CHECK/build" "complete"

echo "running ctest"
if boom_ctest "$BOOM_BASE_BUILD" "$BOOM_LOG_DIR/tests-ctest.log" --no-tests=error; then
  summary="$(grep -E 'tests passed|tests failed' "$BOOM_LOG_DIR/tests-ctest.log" | tail -1)"
  boom_record PASS "$CHECK/ctest" "${summary:-all tests passed}"
  exit 0
fi
failed="$(grep -E '^\s*[0-9]+ - ' "$BOOM_LOG_DIR/tests-ctest.log" | head -10 | tr '\n' ' ')"
boom_record FAIL "$CHECK/ctest" "see tests-ctest.log${failed:+; failed: $failed}"
exit 1
