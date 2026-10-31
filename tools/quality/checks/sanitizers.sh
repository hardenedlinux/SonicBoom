#!/usr/bin/env bash
# ASan + UBSan check in a private build directory, followed by LeakSanitizer on
# the native C++ suites.
#
# Scope and limits:
#   * SONICBOOM_SANITIZE instruments SonicBoom-owned targets only. native-torch,
#     the installed MLIR/LLVM, and the Guile runtime are not instrumented; the
#     ASan runtime still intercepts the process-wide heap.
#   * Guile tests are run with leak detection off: libguile is uninstrumented
#     and Guile guardians are not guaranteed to run at process exit, so their
#     reports would mix runtime shutdown behavior with SonicBoom ownership.
#     Leak detection is therefore a separate run over the native C++ suites.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=sanitizers

if ! sb_required_tools cmake ctest ninja "$SB_CC" "$SB_CXX" guile; then
  sb_record FAIL "$CHECK/toolchain" "required tools missing (see stderr)"
  exit 1
fi
if [ ! -f "$SB_ASAN_LIB" ]; then
  sb_record FAIL "$CHECK/runtime" "libasan not found via $SB_CC -print-file-name"
  exit 1
fi
if ! sb_require_mlir; then
  sb_record FAIL "$CHECK/mlir" "MLIR prefix unavailable"
  exit 1
fi

echo "configuring $SB_ASAN_BUILD (ASan+UBSan, CPU-only, $SB_ASAN_BUILD_TYPE)"
if ! sb_configure "$SB_ASAN_BUILD" "address;undefined" "$SB_ASAN_BUILD_TYPE" \
     > "$SB_LOG_DIR/sanitizers-configure.log" 2>&1; then
  sb_record FAIL "$CHECK/configure" "see sanitizers-configure.log"
  exit 1
fi
sb_record PASS "$CHECK/configure" "$SB_ASAN_BUILD"

echo "building $SB_ASAN_BUILD"
if ! sb_build "$SB_ASAN_BUILD" "$SB_LOG_DIR/sanitizers-build.log"; then
  sb_record FAIL "$CHECK/build" "see sanitizers-build.log"
  exit 1
fi
sb_record PASS "$CHECK/build" "complete"

asan_ok=1
lsan_ok=1

echo "running ASan+UBSan over the full CPU CTest suite"
if LD_PRELOAD="$SB_ASAN_LIB" \
   ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:print_stacktrace=1:strict_string_checks=1" \
   UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
   sb_ctest "$SB_ASAN_BUILD" "$SB_LOG_DIR/sanitizers-asan-ctest.log"; then
  summary="$(grep -E 'tests passed|tests failed' "$SB_LOG_DIR/sanitizers-asan-ctest.log" | tail -1)"
  sb_record PASS "$CHECK/asan-ubsan" "${summary:-all tests passed}"
else
  failed="$(grep -E '^\s*[0-9]+ - ' "$SB_LOG_DIR/sanitizers-asan-ctest.log" | head -10 | tr '\n' ' ')"
  sb_record FAIL "$CHECK/asan-ubsan" "see sanitizers-asan-ctest.log${failed:+; failed: $failed}"
  asan_ok=0
fi

echo "running LeakSanitizer over native C++ suites"
# Restrict to the native suites. test_adamw_lr_input is a Guile script (its
# libguile allocations are process-lifetime and uninstrumented), and the
# ResNet-18 fixture dependency runs a Python generator; neither is a native
# C++ suite, and both would report leaks owned by their runtime. `-E` also
# prevents CTest from pulling the resnet18_artifacts fixture in for a test
# that was filtered out.
lsan_filter=(-R '^test_' -E 'resnet18|adamw_lr_input')
lsan_supp="$SB_Q_DIR/lsan.supp"
lsan_opts="print_suppressions=0"
if [ -f "$lsan_supp" ]; then
  lsan_opts="suppressions=$lsan_supp:$lsan_opts"
fi
if LD_PRELOAD="$SB_ASAN_LIB" \
   ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:print_stacktrace=1:print_summary=1" \
   UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
   LSAN_OPTIONS="$lsan_opts" \
   sb_ctest "$SB_ASAN_BUILD" "$SB_LOG_DIR/sanitizers-lsan-ctest.log" "${lsan_filter[@]}"; then
  sb_record PASS "$CHECK/lsan-native" "native C++ suites (see suppressions file)"
else
  leaks="$(grep -E 'SUMMARY: AddressSanitizer' "$SB_LOG_DIR/sanitizers-lsan-ctest.log" \
           | head -5 | tr '\n' ' ')"
  sb_record FAIL "$CHECK/lsan-native" "see sanitizers-lsan-ctest.log${leaks:+; $leaks}"
  lsan_ok=0
fi

if [ "$asan_ok" -eq 1 ] && [ "$lsan_ok" -eq 1 ]; then
  exit 0
fi
exit 1
