#!/usr/bin/env bash
# Shared configuration and helpers for the SonicBoom quality gate.
#
# Sourced by tools/quality/run.sh and tools/quality/checks/*.sh. This file does
# not run any check by itself.
#
# Design rules:
#   * Paths are derived from this file's location, so the gate works from any
#     current directory.
#   * Build directories are private to the gate (default: .quality/build/...).
#     An existing developer build directory (build/, build-asan/, ...) is never
#     used as an output directory.
#   * A missing required tool is a FAIL, never a silent PASS. Optional tools are
#     reported as SKIP with the reason.

# No `set -e`: check scripts collect results and decide their own exit status.
set -uo pipefail

BOOM_Q_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOM_ROOT="$(cd "$BOOM_Q_DIR/../.." && pwd)"

# --- Locations --------------------------------------------------------------

BOOM_QUALITY_DIR="${SONICBOOM_QUALITY_DIR:-$BOOM_ROOT/.quality}"
BOOM_BUILD_ROOT="${SONICBOOM_QUALITY_BUILD_ROOT:-$BOOM_QUALITY_DIR/build}"
BOOM_LOG_ROOT="${SONICBOOM_QUALITY_LOG_ROOT:-$BOOM_QUALITY_DIR/logs}"

# run.sh sets BOOM_LOG_DIR before sourcing this file through the environment.
# A check script invoked directly gets a timestamped directory of its own.
if [ -z "${BOOM_LOG_DIR:-}" ]; then
  BOOM_LOG_DIR="$BOOM_LOG_ROOT/$(date +%Y%m%d-%H%M%S)-manual"
fi
mkdir -p "$BOOM_LOG_DIR"
export BOOM_LOG_DIR
export BOOM_QUALITY_DIR BOOM_BUILD_ROOT BOOM_LOG_ROOT

BOOM_RESULTS_FILE="${BOOM_RESULTS_FILE:-$BOOM_LOG_DIR/results.tsv}"
# Do not truncate: a check script sources this file in its own process and must
# append to the table created by run.sh.
touch "$BOOM_RESULTS_FILE"
export BOOM_RESULTS_FILE

# --- Toolchain and build configuration --------------------------------------

BOOM_CC="${SONICBOOM_QUALITY_CC:-gcc-13}"
BOOM_CXX="${SONICBOOM_QUALITY_CXX:-g++-13}"
BOOM_JOBS="${SONICBOOM_QUALITY_JOBS:-$(nproc 2>/dev/null || echo 2)}"
BOOM_GENERATOR="${SONICBOOM_QUALITY_GENERATOR:-Ninja}"

BOOM_BUILD_TYPE="${SONICBOOM_QUALITY_BUILD_TYPE:-Release}"
BOOM_ASAN_BUILD_TYPE="${SONICBOOM_QUALITY_ASAN_BUILD_TYPE:-Debug}"
BOOM_TSAN_BUILD_TYPE="${SONICBOOM_QUALITY_TSAN_BUILD_TYPE:-Debug}"

BOOM_MLIR_PREFIX="${SONICBOOM_MLIR_PREFIX:-$BOOM_ROOT/build/llvm-mlir-install}"
BOOM_PRIVATE_TESTS_DIR="${SONICBOOM_PRIVATE_TESTS_DIR:-}"

BOOM_BASE_BUILD="${SONICBOOM_QUALITY_BASE_BUILD:-$BOOM_BUILD_ROOT/base}"
BOOM_ASAN_BUILD="${SONICBOOM_QUALITY_ASAN_BUILD:-$BOOM_BUILD_ROOT/asan}"
BOOM_TSAN_BUILD="${SONICBOOM_QUALITY_TSAN_BUILD:-$BOOM_BUILD_ROOT/tsan}"

# CUDA checks are separate and use an existing CUDA-enabled build tree; the
# gate never builds one (CPU CI must not depend on a GPU toolchain). Point this
# at a developer build, e.g. /path/to/SonicBoom/build.
BOOM_CUDA_BUILD="${SONICBOOM_QUALITY_CUDA_BUILD:-}"

# Optional private suites are opt-in through the normal project variable.
BOOM_PRIVATE_TESTS_DIR="${SONICBOOM_PRIVATE_TESTS_DIR:-}"

# SonicCross (`guild`) is a required build-time dependency. Accept the same
# override the project's CMake does.
BOOM_GUILD="${SONICCROSS_GUILD:-$(command -v guild 2>/dev/null || true)}"
BOOM_SONICCROSS_VERSION="${SONICBOOM_QUALITY_SONICCROSS_VERSION:-v0.0.2}"

# --- Tool discovery ---------------------------------------------------------

# boom_find_tool NAME... -> prints the first executable found on PATH.
boom_find_tool() {
  local t p
  for t in "$@"; do
    p="$(command -v "$t" 2>/dev/null || true)"
    if [ -n "$p" ] && [ -x "$p" ]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

# clang++ (not the C driver): the analyzer compiles C++ and needs the C++
# standard-library include paths that the clang C driver does not add.
#
# Versioned names are preferred over the unversioned ones. Clang 18's frontend
# cannot parse libstdc++ 13's <expected> (the header is gated on
# __cpp_concepts >= 202002L; clang 18 reports 201907L), and that header is used
# throughout core/. Clang 19 reports the required value and parses it. The
# analyzer and clang-tidy must therefore come from a clang >= 19 installation.
BOOM_CLANG="$(boom_find_tool clang++-20 clang++-19 clang++-18 clang++-17 clang++ || true)"
if [ -z "$BOOM_CLANG" ]; then
  BOOM_CLANG="$(boom_find_tool clang-20 clang-19 clang-18 clang || true)"
fi
BOOM_CLANG_TIDY="$(boom_find_tool clang-tidy-20 clang-tidy-19 clang-tidy-18 clang-tidy || true)"
# The GCC installation belonging to the formal compiler. Clang otherwise picks
# the newest GCC installation on the host; when that version's libstdc++ is
# not installed (a GCC 14 driver with only libstdc++-13-dev here), every C++
# TU fails with "cstdint not found". Pin clang to the project's GCC 13.
BOOM_GCC_INSTALL_DIR=""
if [ -n "${BOOM_CXX:-}" ]; then
  _boom_libgcc="$("$BOOM_CXX" -print-libgcc-file-name 2>/dev/null || true)"
  if [ -n "$_boom_libgcc" ]; then
    BOOM_GCC_INSTALL_DIR="$(dirname "$_boom_libgcc")"
  fi
fi
BOOM_SCAN_BUILD="$(boom_find_tool scan-build scan-build-19 scan-build-18 || true)"
BOOM_VALGRIND="$(boom_find_tool valgrind || true)"
BOOM_ASAN_LIB="$("$BOOM_CC" -print-file-name=libasan.so 2>/dev/null || true)"
BOOM_TSAN_LIB="$("$BOOM_CC" -print-file-name=libtsan.so 2>/dev/null || true)"
BOOM_COMPUTE_SANITIZER="${SONICBOOM_QUALITY_COMPUTE_SANITIZER:-}"
if [ -z "$BOOM_COMPUTE_SANITIZER" ]; then
  BOOM_COMPUTE_SANITIZER="$(boom_find_tool compute-sanitizer || true)"
fi
# The /usr/bin copy on some hosts is a 2022 build whose injection library is
# missing. Prefer a versioned CUDA toolkit installation when one exists.
if [ -x /usr/local/cuda-13.4/compute-sanitizer/compute-sanitizer ]; then
  BOOM_COMPUTE_SANITIZER="${SONICBOOM_QUALITY_COMPUTE_SANITIZER:-/usr/local/cuda-13.4/compute-sanitizer/compute-sanitizer}"
fi

# --- Result reporting -------------------------------------------------------

# boom_record STATUS CHECK DETAIL
#   STATUS is PASS, FAIL, or SKIP. The line is printed and appended to the
#   per-run results table consumed by run.sh.
boom_record() {
  local status="$1" check="$2" detail="${3:-}"
  printf '%-4s  %-34s %s\n' "$status" "$check" "$detail"
  printf '%s\t%s\t%s\n' "$status" "$check" "$detail" >> "$BOOM_RESULTS_FILE"
}

boom_required_tools() {
  local missing=0 t p
  for t in "$@"; do
    p="$(boom_find_tool "$t" || true)"
    if [ -z "$p" ]; then
      echo "missing required tool: $t" >&2
      missing=1
    fi
  done
  return "$missing"
}

# boom_require_mlir: configure input that CMake requires. Never downloads.
boom_require_mlir() {
  if [ ! -f "$BOOM_MLIR_PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "MLIR not found at $BOOM_MLIR_PREFIX" >&2
    echo "Build it from the in-tree submodule (no download from this gate):" >&2
    echo "  git submodule update --init third_party/llvm-project" >&2
    echo "  tools/mlir/build-mlir.sh" >&2
    echo "or set SONICBOOM_MLIR_PREFIX to an existing install." >&2
    return 1
  fi
}

# --- Commands ---------------------------------------------------------------

boom_log_command() {
  printf '+ ' >> "$BOOM_LOG_DIR/commands.log"
  printf '%q ' "$@" >> "$BOOM_LOG_DIR/commands.log"
  printf '\n' >> "$BOOM_LOG_DIR/commands.log"
}

# boom_configure BUILD_DIR SANITIZE BUILD_TYPE [EXTRA_CMAKE_ARGS...]
boom_configure() {
  local build="$1" sanitize="$2" btype="$3"
  shift 3
  local cmd=(
    cmake -S "$BOOM_ROOT" -B "$build" -G "$BOOM_GENERATOR"
    -DCMAKE_BUILD_TYPE="$btype"
    -DCMAKE_C_COMPILER="$BOOM_CC"
    -DCMAKE_CXX_COMPILER="$BOOM_CXX"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DSONICBOOM_USE_CUDA=OFF
    -DSONICBOOM_SANITIZE="$sanitize"
    -DMLIR_INSTALL_PREFIX="$BOOM_MLIR_PREFIX"
    -DSONICBOOM_PRIVATE_TESTS_DIR="$BOOM_PRIVATE_TESTS_DIR"
  )
  if [ -n "${BOOM_GUILD:-}" ]; then
    cmd+=(-DSONICCROSS_GUILD="$BOOM_GUILD")
  fi
  if [ "${SONICBOOM_QUALITY_CCACHE:-1}" != "0" ] && command -v ccache >/dev/null 2>&1; then
    cmd+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  fi
  cmd+=("$@")
  boom_log_command "${cmd[@]}"
  "${cmd[@]}"
}

# boom_build BUILD_DIR LOG [TARGET...]
boom_build() {
  local build="$1" log="$2"
  shift 2
  local targets=()
  [ "$#" -gt 0 ] && targets=(--target "$@")
  boom_log_command cmake --build "$build" --parallel "$BOOM_JOBS" "${targets[@]}"
  cmake --build "$build" --parallel "$BOOM_JOBS" "${targets[@]}" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

# boom_ctest BUILD_DIR LOG [CTEST_ARGS...]
boom_ctest() {
  local build="$1" log="$2"
  shift 2
  boom_log_command ctest --test-dir "$build" --output-on-failure "$@"
  ctest --test-dir "$build" --output-on-failure "$@" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

# --- Environment record -----------------------------------------------------

boom_write_environment() {
  local f="$BOOM_LOG_DIR/environment.txt"
  {
    echo "date: $(date -Is)"
    echo "host: $(uname -a)"
    echo "cwd: $(pwd)"
    echo "repo: $BOOM_ROOT"
    echo "git-branch: $(git -C "$BOOM_ROOT" branch --show-current 2>/dev/null || true)"
    echo "git-rev: $(git -C "$BOOM_ROOT" rev-parse HEAD 2>/dev/null || true)"
    echo "git-status:"
    git -C "$BOOM_ROOT" status --short 2>/dev/null || true
    echo
    echo "gcc: $("$BOOM_CC" --version 2>/dev/null | head -1 || true)"
    echo "g++: $("$BOOM_CXX" --version 2>/dev/null | head -1 || true)"
    echo "clang: $("${BOOM_CLANG:-clang}" --version 2>/dev/null | head -1 || true)"
    echo "clang-tidy: $("${BOOM_CLANG_TIDY:-clang-tidy}" --version 2>/dev/null | head -1 || true)"
    echo "clang gcc-install-dir: ${BOOM_GCC_INSTALL_DIR:-<unset>}"
    echo "scan-build: ${BOOM_SCAN_BUILD:-missing}"
    echo "valgrind: $("${BOOM_VALGRIND:-valgrind}" --version 2>/dev/null | head -1 || true)"
    echo "cmake: $(cmake --version 2>/dev/null | head -1 || true)"
    echo "ctest: $(ctest --version 2>/dev/null | head -1 || true)"
    echo "ninja: $(ninja --version 2>/dev/null || true)"
    echo "ccache: $(ccache --version 2>/dev/null | head -1 || true)"
    echo "guile: $(guile --version 2>/dev/null | head -1 || true)"
    echo "guild: ${BOOM_GUILD:-missing}"
    echo "compute-sanitizer: ${BOOM_COMPUTE_SANITIZER:-missing}"
    echo
    echo "SONICBOOM_SANITIZE base: (none)"
    echo "build type: $BOOM_BUILD_TYPE"
    echo "jobs: $BOOM_JOBS"
    echo "generator: $BOOM_GENERATOR"
    echo "MLIR prefix: $BOOM_MLIR_PREFIX"
    echo "MLIR version: $(grep -h 'LLVM_PACKAGE_VERSION ' \
      "$BOOM_MLIR_PREFIX/lib/cmake/llvm/LLVMConfig.cmake" 2>/dev/null | head -1 || echo unknown)"
    echo "private tests dir: ${BOOM_PRIVATE_TESTS_DIR:-<none>}"
    echo "base build: $BOOM_BASE_BUILD"
    echo "asan build: $BOOM_ASAN_BUILD"
    echo "tsan build: $BOOM_TSAN_BUILD"
    echo "cuda build: ${BOOM_CUDA_BUILD:-<none>}"
    echo
    echo "disk:"
    df -h "$BOOM_ROOT" 2>/dev/null | tail -2 || true
  } > "$f"
  echo "Environment recorded in $f"
}
