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

SB_Q_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB_ROOT="$(cd "$SB_Q_DIR/../.." && pwd)"

# --- Locations --------------------------------------------------------------

SB_QUALITY_DIR="${SONICBOOM_QUALITY_DIR:-$SB_ROOT/.quality}"
SB_BUILD_ROOT="${SONICBOOM_QUALITY_BUILD_ROOT:-$SB_QUALITY_DIR/build}"
SB_LOG_ROOT="${SONICBOOM_QUALITY_LOG_ROOT:-$SB_QUALITY_DIR/logs}"

# run.sh sets SB_LOG_DIR before sourcing this file through the environment.
# A check script invoked directly gets a timestamped directory of its own.
if [ -z "${SB_LOG_DIR:-}" ]; then
  SB_LOG_DIR="$SB_LOG_ROOT/$(date +%Y%m%d-%H%M%S)-manual"
fi
mkdir -p "$SB_LOG_DIR"
export SB_LOG_DIR
export SB_QUALITY_DIR SB_BUILD_ROOT SB_LOG_ROOT

SB_RESULTS_FILE="${SB_RESULTS_FILE:-$SB_LOG_DIR/results.tsv}"
# Do not truncate: a check script sources this file in its own process and must
# append to the table created by run.sh.
touch "$SB_RESULTS_FILE"
export SB_RESULTS_FILE

# --- Toolchain and build configuration --------------------------------------

SB_CC="${SONICBOOM_QUALITY_CC:-gcc-13}"
SB_CXX="${SONICBOOM_QUALITY_CXX:-g++-13}"
SB_JOBS="${SONICBOOM_QUALITY_JOBS:-$(nproc 2>/dev/null || echo 2)}"
SB_GENERATOR="${SONICBOOM_QUALITY_GENERATOR:-Ninja}"

SB_BUILD_TYPE="${SONICBOOM_QUALITY_BUILD_TYPE:-Release}"
SB_ASAN_BUILD_TYPE="${SONICBOOM_QUALITY_ASAN_BUILD_TYPE:-Debug}"
SB_TSAN_BUILD_TYPE="${SONICBOOM_QUALITY_TSAN_BUILD_TYPE:-Debug}"

SB_MLIR_PREFIX="${SONICBOOM_MLIR_PREFIX:-$SB_ROOT/build/llvm-mlir-install}"
SB_PRIVATE_TESTS_DIR="${SONICBOOM_PRIVATE_TESTS_DIR:-}"

SB_BASE_BUILD="${SONICBOOM_QUALITY_BASE_BUILD:-$SB_BUILD_ROOT/base}"
SB_ASAN_BUILD="${SONICBOOM_QUALITY_ASAN_BUILD:-$SB_BUILD_ROOT/asan}"
SB_TSAN_BUILD="${SONICBOOM_QUALITY_TSAN_BUILD:-$SB_BUILD_ROOT/tsan}"

# CUDA checks are separate and use an existing CUDA-enabled build tree; the
# gate never builds one (CPU CI must not depend on a GPU toolchain). Point this
# at a developer build, e.g. /path/to/SonicBoom/build.
SB_CUDA_BUILD="${SONICBOOM_QUALITY_CUDA_BUILD:-}"

# Optional private suites are opt-in through the normal project variable.
SB_PRIVATE_TESTS_DIR="${SONICBOOM_PRIVATE_TESTS_DIR:-}"

# SonicCross (`guild`) is a required build-time dependency. Accept the same
# override the project's CMake does.
SB_GUILD="${SONICCROSS_GUILD:-$(command -v guild 2>/dev/null || true)}"
SB_SONICCROSS_VERSION="${SONICBOOM_QUALITY_SONICCROSS_VERSION:-v0.0.2}"

# --- Tool discovery ---------------------------------------------------------

# sb_find_tool NAME... -> prints the first executable found on PATH.
sb_find_tool() {
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
SB_CLANG="$(sb_find_tool clang++-20 clang++-19 clang++-18 clang++-17 clang++ || true)"
if [ -z "$SB_CLANG" ]; then
  SB_CLANG="$(sb_find_tool clang-20 clang-19 clang-18 clang || true)"
fi
SB_CLANG_TIDY="$(sb_find_tool clang-tidy-20 clang-tidy-19 clang-tidy-18 clang-tidy || true)"
# The GCC installation belonging to the formal compiler. Clang otherwise picks
# the newest GCC installation on the host; when that version's libstdc++ is
# not installed (a GCC 14 driver with only libstdc++-13-dev here), every C++
# TU fails with "cstdint not found". Pin clang to the project's GCC 13.
SB_GCC_INSTALL_DIR=""
if [ -n "${SB_CXX:-}" ]; then
  _sb_libgcc="$("$SB_CXX" -print-libgcc-file-name 2>/dev/null || true)"
  if [ -n "$_sb_libgcc" ]; then
    SB_GCC_INSTALL_DIR="$(dirname "$_sb_libgcc")"
  fi
fi
SB_SCAN_BUILD="$(sb_find_tool scan-build scan-build-19 scan-build-18 || true)"
SB_VALGRIND="$(sb_find_tool valgrind || true)"
SB_ASAN_LIB="$("$SB_CC" -print-file-name=libasan.so 2>/dev/null || true)"
SB_TSAN_LIB="$("$SB_CC" -print-file-name=libtsan.so 2>/dev/null || true)"
SB_COMPUTE_SANITIZER="${SONICBOOM_QUALITY_COMPUTE_SANITIZER:-}"
if [ -z "$SB_COMPUTE_SANITIZER" ]; then
  SB_COMPUTE_SANITIZER="$(sb_find_tool compute-sanitizer || true)"
fi
# The /usr/bin copy on some hosts is a 2022 build whose injection library is
# missing. Prefer a versioned CUDA toolkit installation when one exists.
if [ -x /usr/local/cuda-13.4/compute-sanitizer/compute-sanitizer ]; then
  SB_COMPUTE_SANITIZER="${SONICBOOM_QUALITY_COMPUTE_SANITIZER:-/usr/local/cuda-13.4/compute-sanitizer/compute-sanitizer}"
fi

# --- Result reporting -------------------------------------------------------

# sb_record STATUS CHECK DETAIL
#   STATUS is PASS, FAIL, or SKIP. The line is printed and appended to the
#   per-run results table consumed by run.sh.
sb_record() {
  local status="$1" check="$2" detail="${3:-}"
  printf '%-4s  %-34s %s\n' "$status" "$check" "$detail"
  printf '%s\t%s\t%s\n' "$status" "$check" "$detail" >> "$SB_RESULTS_FILE"
}

sb_required_tools() {
  local missing=0 t p
  for t in "$@"; do
    p="$(sb_find_tool "$t" || true)"
    if [ -z "$p" ]; then
      echo "missing required tool: $t" >&2
      missing=1
    fi
  done
  return "$missing"
}

# sb_require_mlir: configure input that CMake requires. Never downloads.
sb_require_mlir() {
  if [ ! -f "$SB_MLIR_PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "MLIR not found at $SB_MLIR_PREFIX" >&2
    echo "Build it from the in-tree submodule (no download from this gate):" >&2
    echo "  git submodule update --init third_party/llvm-project" >&2
    echo "  tools/mlir/build-mlir.sh" >&2
    echo "or set SONICBOOM_MLIR_PREFIX to an existing install." >&2
    return 1
  fi
}

# --- Commands ---------------------------------------------------------------

sb_log_command() {
  printf '+ ' >> "$SB_LOG_DIR/commands.log"
  printf '%q ' "$@" >> "$SB_LOG_DIR/commands.log"
  printf '\n' >> "$SB_LOG_DIR/commands.log"
}

# sb_configure BUILD_DIR SANITIZE BUILD_TYPE [EXTRA_CMAKE_ARGS...]
sb_configure() {
  local build="$1" sanitize="$2" btype="$3"
  shift 3
  local cmd=(
    cmake -S "$SB_ROOT" -B "$build" -G "$SB_GENERATOR"
    -DCMAKE_BUILD_TYPE="$btype"
    -DCMAKE_C_COMPILER="$SB_CC"
    -DCMAKE_CXX_COMPILER="$SB_CXX"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DSONICBOOM_USE_CUDA=OFF
    -DSONICBOOM_SANITIZE="$sanitize"
    -DMLIR_INSTALL_PREFIX="$SB_MLIR_PREFIX"
    -DSONICBOOM_PRIVATE_TESTS_DIR="$SB_PRIVATE_TESTS_DIR"
  )
  if [ -n "${SB_GUILD:-}" ]; then
    cmd+=(-DSONICCROSS_GUILD="$SB_GUILD")
  fi
  if [ "${SONICBOOM_QUALITY_CCACHE:-1}" != "0" ] && command -v ccache >/dev/null 2>&1; then
    cmd+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  fi
  cmd+=("$@")
  sb_log_command "${cmd[@]}"
  "${cmd[@]}"
}

# sb_build BUILD_DIR LOG [TARGET...]
sb_build() {
  local build="$1" log="$2"
  shift 2
  local targets=()
  [ "$#" -gt 0 ] && targets=(--target "$@")
  sb_log_command cmake --build "$build" --parallel "$SB_JOBS" "${targets[@]}"
  cmake --build "$build" --parallel "$SB_JOBS" "${targets[@]}" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

# sb_ctest BUILD_DIR LOG [CTEST_ARGS...]
sb_ctest() {
  local build="$1" log="$2"
  shift 2
  sb_log_command ctest --test-dir "$build" --output-on-failure "$@"
  ctest --test-dir "$build" --output-on-failure "$@" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

# --- Environment record -----------------------------------------------------

sb_write_environment() {
  local f="$SB_LOG_DIR/environment.txt"
  {
    echo "date: $(date -Is)"
    echo "host: $(uname -a)"
    echo "cwd: $(pwd)"
    echo "repo: $SB_ROOT"
    echo "git-branch: $(git -C "$SB_ROOT" branch --show-current 2>/dev/null || true)"
    echo "git-rev: $(git -C "$SB_ROOT" rev-parse HEAD 2>/dev/null || true)"
    echo "git-status:"
    git -C "$SB_ROOT" status --short 2>/dev/null || true
    echo
    echo "gcc: $("$SB_CC" --version 2>/dev/null | head -1 || true)"
    echo "g++: $("$SB_CXX" --version 2>/dev/null | head -1 || true)"
    echo "clang: $("${SB_CLANG:-clang}" --version 2>/dev/null | head -1 || true)"
    echo "clang-tidy: $("${SB_CLANG_TIDY:-clang-tidy}" --version 2>/dev/null | head -1 || true)"
    echo "clang gcc-install-dir: ${SB_GCC_INSTALL_DIR:-<unset>}"
    echo "scan-build: ${SB_SCAN_BUILD:-missing}"
    echo "valgrind: $("${SB_VALGRIND:-valgrind}" --version 2>/dev/null | head -1 || true)"
    echo "cmake: $(cmake --version 2>/dev/null | head -1 || true)"
    echo "ctest: $(ctest --version 2>/dev/null | head -1 || true)"
    echo "ninja: $(ninja --version 2>/dev/null || true)"
    echo "ccache: $(ccache --version 2>/dev/null | head -1 || true)"
    echo "guile: $(guile --version 2>/dev/null | head -1 || true)"
    echo "guild: ${SB_GUILD:-missing}"
    echo "compute-sanitizer: ${SB_COMPUTE_SANITIZER:-missing}"
    echo
    echo "SONICBOOM_SANITIZE base: (none)"
    echo "build type: $SB_BUILD_TYPE"
    echo "jobs: $SB_JOBS"
    echo "generator: $SB_GENERATOR"
    echo "MLIR prefix: $SB_MLIR_PREFIX"
    echo "MLIR version: $(grep -h 'LLVM_PACKAGE_VERSION ' \
      "$SB_MLIR_PREFIX/lib/cmake/llvm/LLVMConfig.cmake" 2>/dev/null | head -1 || echo unknown)"
    echo "private tests dir: ${SB_PRIVATE_TESTS_DIR:-<none>}"
    echo "base build: $SB_BASE_BUILD"
    echo "asan build: $SB_ASAN_BUILD"
    echo "tsan build: $SB_TSAN_BUILD"
    echo "cuda build: ${SB_CUDA_BUILD:-<none>}"
    echo
    echo "disk:"
    df -h "$SB_ROOT" 2>/dev/null | tail -2 || true
  } > "$f"
  echo "Environment recorded in $f"
}
