#!/usr/bin/env bash
# Optional ThreadSanitizer check in its own build directory. TSan is never
# combined with ASan.
#
# Scope limits (reported, not hidden):
#   * SONICBOOM_SANITIZE=thread instruments SonicBoom-owned code only.
#     native-torch, MLIR/LLVM, and libguile are uninstrumented; races whose
#     both accesses are in those trees are invisible to TSan.
#   * The Guile concurrency test is run as supplementary evidence. Guile's own
#     mutex/finalizer machinery is uninstrumented, so its result cannot be read
#     as a complete Scheme-level race audit.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=concurrency

if ! sb_required_tools cmake ninja "$SB_CC" "$SB_CXX" guile; then
  sb_record SKIP "$CHECK/toolchain" "required TSan tools missing"
  exit 0
fi
if [ ! -f "$SB_TSAN_LIB" ]; then
  sb_record SKIP "$CHECK/runtime" "libtsan not found via $SB_CC -print-file-name"
  exit 0
fi
if ! sb_require_mlir; then
  sb_record SKIP "$CHECK/mlir" "MLIR prefix unavailable"
  exit 0
fi

echo "configuring $SB_TSAN_BUILD (TSan, CPU-only, $SB_TSAN_BUILD_TYPE)"
if ! sb_configure "$SB_TSAN_BUILD" "thread" "$SB_TSAN_BUILD_TYPE" \
     > "$SB_LOG_DIR/concurrency-configure.log" 2>&1; then
  sb_record FAIL "$CHECK/configure" "see concurrency-configure.log"
  exit 1
fi

targets=(sonicboom test_tensor test_s_expr test_s_expr_exec test_persistence
         test_capi test_capi_multi test_capi_parameter_view)
echo "building TSan targets: ${targets[*]}"
if ! sb_build "$SB_TSAN_BUILD" "$SB_LOG_DIR/concurrency-build.log" "${targets[@]}"; then
  sb_record FAIL "$CHECK/build" "see concurrency-build.log"
  exit 1
fi
sb_record PASS "$CHECK/build" "${targets[*]}"

export TSAN_OPTIONS="halt_on_error=1:second_deadlock_stack=1:print_stacktrace=1"
export LD_LIBRARY_PATH="$SB_TSAN_BUILD/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Newer kernels randomize the address space with more entropy than TSan's
# shadow-memory layout can tolerate ("unexpected memory mapping"). Disabling
# ASLR for the test process is the documented workaround and does not change
# the code under test.
aslr_off=()
if command -v setarch >/dev/null 2>&1 && setarch "$(uname -m)" -R true 2>/dev/null; then
  aslr_off=(setarch "$(uname -m)" -R)
fi

run_test() {
  local name="$1"
  shift
  local log="$SB_LOG_DIR/concurrency-$name.log"
  "${aslr_off[@]}" "$@" > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    sb_record PASS "$CHECK/$name" "no TSan report"
  else
    local report
    report="$(grep -E 'WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer' "$log" | head -3 | tr '\n' ' ')"
    sb_record FAIL "$CHECK/$name" "rc=$rc${report:+; $report}"
  fi
  return "$rc"
}

rc=0
for t in test_tensor test_s_expr test_s_expr_exec test_persistence \
         test_capi test_capi_multi test_capi_parameter_view; do
  run_test "$t" "$SB_TSAN_BUILD/tests/core/$t" || rc=1
done

# Supplementary: Guile's single-flight model compilation path. Libguile is
# uninstrumented, so this observes instrumented SonicBoom code only.
guile_log="$SB_LOG_DIR/concurrency-guile-model-concurrency.log"
if "${aslr_off[@]}" env \
     LD_PRELOAD="$SB_TSAN_LIB" \
     TSAN_OPTIONS="$TSAN_OPTIONS" \
     GUILE_AUTO_COMPILE=0 \
     guile --no-auto-compile \
     -L "$SB_ROOT/bindings/guile/scheme" \
     "$SB_ROOT/bindings/guile/tests/test-model-concurrency.scm" \
     > "$guile_log" 2>&1; then
  sb_record PASS "$CHECK/guile-model-concurrency" "supplementary; libguile uninstrumented"
else
  report="$(grep -E 'WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer' "$guile_log" | head -3 | tr '\n' ' ')"
  sb_record FAIL "$CHECK/guile-model-concurrency" "supplementary; see log${report:+; $report}"
  rc=1
fi

exit "$rc"
