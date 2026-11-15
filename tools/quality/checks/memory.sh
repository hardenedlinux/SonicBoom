#!/usr/bin/env bash
# Optional Valgrind Memcheck pass over selected native CPU suites.
#
# Valgrind is a complementary check: it observes the uninstrumented trees
# (native-torch, MLIR/LLVM, libguile) that ASan cannot instrument, at the cost
# of a large slowdown. Only a focused set of lifetime/execution suites runs
# here; the complete suite runs under ASan/UBSan in checks/sanitizers.sh.

set -uo pipefail
# shellcheck source=../config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CHECK=memory

if [ -z "${BOOM_VALGRIND:-}" ]; then
  boom_record SKIP "$CHECK/valgrind" "valgrind not installed (optional check)"
  exit 0
fi
if ! boom_required_tools cmake ninja "$BOOM_CC" "$BOOM_CXX"; then
  boom_record SKIP "$CHECK/toolchain" "build tools missing (optional check)"
  exit 0
fi
if ! boom_require_mlir; then
  boom_record SKIP "$CHECK/mlir" "MLIR prefix unavailable (optional check)"
  exit 0
fi

# Reuse the normal CPU tree; build only the suites this check runs.
if [ ! -f "$BOOM_BASE_BUILD/CMakeCache.txt" ]; then
  if ! boom_configure "$BOOM_BASE_BUILD" "" "$BOOM_BUILD_TYPE" \
       > "$BOOM_LOG_DIR/memory-configure.log" 2>&1; then
    boom_record FAIL "$CHECK/configure" "see memory-configure.log"
    exit 1
  fi
fi

tests=(test_tensor test_s_expr_exec test_persistence test_capi
       test_capi_multi test_capi_parameter_view)
echo "building Valgrind targets: ${tests[*]}"
if ! boom_build "$BOOM_BASE_BUILD" "$BOOM_LOG_DIR/memory-build.log" \
     sonicboom "${tests[@]}"; then
  boom_record FAIL "$CHECK/build" "see memory-build.log"
  exit 1
fi

export LD_LIBRARY_PATH="$BOOM_BASE_BUILD/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
supp="$BOOM_Q_DIR/valgrind.supp"
supp_args=()
[ -f "$supp" ] && supp_args=(--suppressions="$supp")

rc=0
for t in "${tests[@]}"; do
  bin="$BOOM_BASE_BUILD/tests/core/$t"
  [ -x "$bin" ] || { boom_record FAIL "$CHECK/$t" "test binary missing: $bin"; rc=1; continue; }
  log="$BOOM_LOG_DIR/memory-$t.log"
  echo "-- valgrind $t"
  if "$BOOM_VALGRIND" --tool=memcheck \
       --error-exitcode=99 \
       --leak-check=full \
       --show-leak-kinds=definite,indirect \
       --errors-for-leak-kinds=definite \
       --track-origins=yes \
       --num-callers=30 \
       --child-silent-after-fork=yes \
       "${supp_args[@]}" \
       "$bin" > "$log" 2>&1; then
    boom_record PASS "$CHECK/$t" "no definite errors/leaks"
  else
    n="$("$BOOM_VALGRIND" --version >/dev/null 2>&1; grep -E 'ERROR SUMMARY: [0-9]+ errors' "$log" | tail -1)"
    boom_record FAIL "$CHECK/$t" "see memory-$t.log${n:+; $n}"
    rc=1
  fi
done

exit "$rc"
