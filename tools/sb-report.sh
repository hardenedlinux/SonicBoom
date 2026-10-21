#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# sb-report.sh — the one-command "what did this run on" report.
#
# A SonicBoom number means nothing without the machine and the binary behind it,
# so this prints both: the host, the CPU and its SIMD capability, the GPU, the
# toolchain, the LLVM/MLIR linked in, and how the loaded libsonicboom.so was
# built (sb_build_info() through the Guile binding — see bindings/guile/README.md,
# "(build-info)"). Paste it next to a benchmark result or a bug report.
#
# The report also *checks* the build it describes. With a model it runs the
# golden checks through ctest (token goldens, prefix cache, plan cache,
# elementwise, and the four golden plans) and the canonical benchmark through
# tools/bench.sh, which is the same measurement a release reports. A model is
# therefore required: a report that skips the numbers is the one thing this
# script exists to prevent.
#
# Everything else is best-effort: a section whose tool is absent says so, but a
# failed golden check makes the script exit 1 (a host with no Guile still exits
# 0). --no-golden / --no-bench skip the two sections when they are not wanted.
#
# Usage: tools/sb-report.sh --gemma PATH [--qwen PATH] [--deepseek PATH]
#                           [--json] [--lib PATH] [--runs N] [--build]
#                           [--no-golden] [--no-bench]
# Env:   SONICBOOM_LIB=/path/to/libsonicboom.so   (default <repo>/build/core/libsonicboom.so)
#        LLAMA_BENCH=/path/to/llama-bench        (passed through; adds llama.cpp)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JSON=0
LIB="${SONICBOOM_LIB:-}"
GEMMA=""
QWEN=""
DEEPSEEK=""
RUNS=3
BUILD=0
GOLDEN=1
BENCH=1

while [ $# -gt 0 ]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --lib)  LIB="$2"; shift 2 ;;
    --gemma) GEMMA="$2"; shift 2 ;;
    --qwen)  QWEN="$2"; shift 2 ;;
    --deepseek) DEEPSEEK="$2"; shift 2 ;;
    --runs)  RUNS="$2"; shift 2 ;;
    --build) BUILD=1; shift ;;
    --no-golden) GOLDEN=0; shift ;;
    --no-bench)  BENCH=0; shift ;;
    -h|--help) sed -n '3,29p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$LIB" ] || LIB="$ROOT/build/core/libsonicboom.so"

# --- key/value collection ---------------------------------------------------
# Every fact goes through emit so --json and the text rendering cannot drift
# apart. Keys are dotted paths: build.git_commit, host.cpu, gpu.name.
KEYS=()
VALS=()
emit() { KEYS+=("$1"); VALS+=("$2"); }
have() { command -v "$1" >/dev/null 2>&1; }
# First line of a command's output, or the fallback if it does not exist/run.
probe() { local fb="$1"; shift; if have "$1"; then local out; out="$("$@" 2>/dev/null | head -n1)"; [ -n "$out" ] && { echo "$out"; return; }; fi; echo "$fb"; }
# One KEY=VALUE field of a `RESULT ...` line (bench.sh's machine-readable form).
field() { awk -v k="$1" '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]==k){print a[2]; exit}}}'; }

# --- the SonicBoom library --------------------------------------------------
# The artifact itself first (identifies the file even when Guile cannot read it),
# then what it says about itself.
if [ -f "$LIB" ]; then
  emit build.library "$LIB"
  emit build.size_bytes "$(stat -c %s "$LIB" 2>/dev/null || echo '?')"
  digest="$(sha256sum "$LIB" 2>/dev/null)"; emit build.sha256 "${digest%% *}"
  stamp="$(stat -c %y "$LIB" 2>/dev/null)"; emit build.built_at "${stamp%%.*}"
  if have guile; then
    # (build-info) returns a symbol-keyed alist; flatten the two nested alists
    # (cuda, llvm) into dotted keys, so `build.cuda.archs` comes out as a
    # comma-separated list and booleans as true/false.
    info="$(SONICBOOM_LIB="$LIB" LD_LIBRARY_PATH="$(dirname "$LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      guile --no-auto-compile -L "$ROOT/bindings/guile/scheme" -c '
        (use-modules (sonicboom ffi))
        (define (fmt v)
          (cond ((list? v) (string-join (map fmt v) ","))
                ((boolean? v) (if v "true" "false"))
                (else (format #f "~a" v))))
        ;; A load failure is the most actionable line in the report: print
        ;; the dynamic-linker/exception text instead of exiting with no output.
        (catch #t
          (lambda ()
            (for-each (lambda (kv)
                        (if (pair? (cdr kv))
                            (for-each (lambda (kv2)
                                        (format #t "build.~a.~a=~a~%" (car kv) (car kv2) (fmt (cdr kv2))))
                                      (cdr kv))
                            (format #t "build.~a=~a~%" (car kv) (fmt (cdr kv)))))
                      (build-info)))
          (lambda (key . args)
            (format #t "build.load_error=~a~%" (exception-text key args))))' 2>/dev/null)"
    if [ -n "$info" ]; then
      while IFS='=' read -r k v; do [ -n "$k" ] && emit "$k" "$v"; done <<<"$info"
    else
      emit build.git_commit "unknown (Guile could not load $LIB)"
    fi
  else
    emit build.git_commit "unknown (Guile not installed; see build.library)"
  fi
else
  emit build.library "not found ($LIB); pass --lib PATH"
fi

# --- host -------------------------------------------------------------------
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  emit host.os "${PRETTY_NAME:-unknown}"
fi
emit host.kernel "$(uname -srm)"
cpu="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n1)"
# aarch64 lscpu has no "Model name"; there the architecture line is the closest
# thing to a CPU identity, so fall back to it rather than to nothing.
[ -n "$cpu" ] || cpu="$(probe unknown lscpu)"
emit host.cpu "$cpu"
if [ -r /proc/cpuinfo ]; then
  emit host.cpu_cores "$(grep -c ^processor /proc/cpuinfo 2>/dev/null)"
  # x86 reports avx2/avx512f; aarch64 reports asimd (NEON). This is what the
  # q*_K AVX2 kernels in core/src/quant/quantized_matmul.cpp gate on, so it is
  # exactly what a performance comparison needs.
  flags="$(awk -F: '/^flags|^Features/{print $2; exit}' /proc/cpuinfo)"
  simd=""
  for f in avx512f avx2 fma asimd; do
    case " $flags " in *" $f "*) simd="$simd $f" ;; esac
  done
  emit host.cpu_simd "${simd# }"
fi
[ -r /proc/meminfo ] && emit host.memory "$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"

# --- GPU --------------------------------------------------------------------
if have nvidia-smi; then
  IFS=',' read -r gname gdrv gmem gcap < <(nvidia-smi \
    --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -n1)
  emit gpu.name "${gname:-unknown}"; emit gpu.driver "${gdrv:-}"; emit gpu.vram "${gmem:-}"; emit gpu.compute_cap "${gcap:-}"
else
  emit gpu.name "none (nvidia-smi not found)"
fi

# --- toolchain and LLVM/MLIR ------------------------------------------------
for c in g++-13 g++ c++; do have "$c" && { emit toolchain.compiler "$("$c" --version | head -n1)"; break; }; done
emit toolchain.cmake "$(probe none cmake --version)"
emit toolchain.ninja "$(probe none ninja --version)"
emit toolchain.linker "$(have mold && echo mold || echo 'default (ld)')"
# The nvcc on PATH — a different fact from build.cuda.toolkit, which is what the
# library was actually compiled with.
cuda_kit="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1)"
emit toolchain.cuda_toolkit "${cuda_kit:-none (nvcc not found)}"
emit toolchain.guile "$(probe none guile --version)"
# The install the build linked against (Stage F); the build-info above is the
# authoritative answer, this is the fallback when Guile could not be used.
mlir_cfg="$ROOT/build/llvm-mlir-install/lib/cmake/mlir/MLIRConfig.cmake"
if [ -r "$mlir_cfg" ]; then
  emit toolchain.mlir "${mlir_cfg%/*} (in-tree install)"
else
  emit toolchain.mlir "no in-tree install (external -DMLIR_DIR, or MLIR off)"
fi

# --- golden checks and the canonical benchmark ------------------------------
# A model is required: the report exists to bind a number to the machine and
# binary that produced it, so "no model, no numbers" is an error, not an empty
# section. --no-golden / --no-bench still require a model; they only skip work.
if [ -z "$GEMMA" ] && [ -z "$QWEN" ] && [ -z "$DEEPSEEK" ]; then
  echo "sb-report: no model given — pass --gemma PATH, --qwen PATH and/or --deepseek PATH" >&2
  echo "           the report runs the golden checks and the canonical bench against it" >&2
  exit 2
fi
for m in "$GEMMA" "$QWEN" "$DEEPSEEK"; do
  [ -z "$m" ] || [ -f "$m" ] || { echo "sb-report: model not found: $m" >&2; exit 2; }
done
[ -z "$GEMMA" ] || export SONICBOOM_GEMMA_GGUF="$GEMMA"
[ -z "$QWEN" ]  || export SONICBOOM_QWEN_GGUF="$QWEN"
[ -z "$DEEPSEEK" ] || export SONICBOOM_DEEPSEEK_GGUF="$DEEPSEEK"

GOLDEN_FAIL=0
golden_run() { # LABEL CTEST_REGEX
  local out rc
  out=$(ctest --test-dir "$ROOT/build" -R "$2" --output-on-failure 2>&1); rc=$?
  [ "$rc" = 0 ] || GOLDEN_FAIL=1
  if [ "$JSON" = 1 ]; then
    [ "$rc" = 0 ] && emit "golden.$1" pass || emit "golden.$1" fail
  fi
  if [ "$JSON" != 1 ]; then
    printf '  %-24s %s\n' "$1" "$([ "$rc" = 0 ] && echo pass || echo FAIL)"
    [ "$rc" = 0 ] || printf '%s\n' "$out" | tail -n 20
  fi
}

if [ "$GOLDEN" = 1 ]; then
  [ "$JSON" = 1 ] || echo "[golden]"
  [ -n "$GEMMA" ] && golden_run gemma_tokens '^test_generate$'
  [ -n "$GEMMA" ] && golden_run gemma_prefix_cache '^test_prefix_cache$'
  [ -n "$GEMMA" ] && golden_run gemma_plan_cache '^test_plan_cache$'
  [ -n "$GEMMA" ] && golden_run cuda_elementwise '^test_cuda_elementwise$'
  [ -n "$QWEN" ]  && golden_run qwen_tokens '^test_qwen2_generate$'
  [ -n "$DEEPSEEK" ] && golden_run deepseek_tokens '^test_deepseek_generate$'
  if [ -n "$GEMMA" ] && [ -n "$QWEN" ]; then
    golden_run golden_plans '^test_boom_conversion$'
  else
    [ "$JSON" = 1 ] && emit golden.golden_plans "skipped (needs --gemma and --qwen)"
    [ "$JSON" = 1 ] || printf '  %-24s %s\n' golden_plans "skipped (needs --gemma and --qwen)"
  fi
  [ "$JSON" = 1 ] || echo
fi

if [ "$BENCH" = 1 ]; then
  [ "$JSON" = 1 ] || echo "[bench]"
  bench_one() { # LABEL GGUF
    local args=("$ROOT/tools/bench.sh" "$2" --runs "$RUNS")
    [ "$BUILD" = 1 ] || args+=(--no-build)
    local out
    out=$(SONICBOOM_GEMMA_GGUF="$2" "${args[@]}" 2>&1) || true
    if [ "$JSON" != 1 ]; then
      printf '%s\n' "$out"
    fi
    local line; line=$(printf '%s\n' "$out" | grep '^RESULT bench ' | tail -n1)
    if [ -z "$line" ]; then
      [ "$JSON" = 1 ] && emit "bench.$1" "failed (see the bench output above)"
      [ "$JSON" = 1 ] || echo "  bench $1: FAILED (see output above)"
      return
    fi
    local k v
    for k in tg64 d512 pp512 reuse_ms; do
      v=$(printf '%s\n' "$line" | field "$k")
      [ -n "$v" ] && { [ "$JSON" = 1 ] && emit "bench.$1.$k" "$v"; }
    done
    v=$(printf '%s\n' "$out" | sed -n 's/.*gpu clock: SM [0-9]*-[0-9]* MHz (median \([0-9]*\)).*/\1/p' | tail -n1)
    [ -n "$v" ] && { [ "$JSON" = 1 ] && emit "bench.$1.sm_clock_mhz" "$v"; }
    [ "$JSON" = 1 ] || echo
  }
  [ -n "$GEMMA" ] && bench_one gemma "$GEMMA"
  [ -n "$QWEN" ]  && bench_one qwen "$QWEN"
  [ -n "$DEEPSEEK" ] && bench_one deepseek "$DEEPSEEK"
fi

# --- render -----------------------------------------------------------------
if [ "$JSON" = 1 ]; then
  printf '{'
  for i in "${!KEYS[@]}"; do
    v="${VALS[$i]//\\/\\\\}"; v="${v//\"/\\\"}"
    [ "$i" = 0 ] || printf ','
    printf '"%s":"%s"' "${KEYS[$i]}" "$v"
  done
  printf '}\n'
else
  group=""
  for i in "${!KEYS[@]}"; do
    k="${KEYS[$i]}"; g="${k%%.*}"
    if [ "$g" != "$group" ]; then printf '\n[%s]\n' "$g"; group="$g"; fi
    printf '  %-22s %s\n' "${k#*.}" "${VALS[$i]}"
  done
  echo
fi
[ "$GOLDEN_FAIL" = 0 ] || exit 1
exit 0
