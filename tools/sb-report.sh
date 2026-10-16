#!/usr/bin/env bash
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
# Everything is best-effort: a section whose tool is absent says so and the
# script still exits 0, because a CPU-only host without CUDA or Guile is a
# perfectly ordinary place to report from.
#
# Usage: tools/sb-report.sh [--json] [--lib PATH]
# Env:   SONICBOOM_LIB=/path/to/libsonicboom.so   (default <repo>/build/core/libsonicboom.so)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JSON=0
LIB="${SONICBOOM_LIB:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --lib)  LIB="$2"; shift 2 ;;
    -h|--help) sed -n '3,21p' "$0"; exit 0 ;;
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
        (for-each (lambda (kv)
                    (if (pair? (cdr kv))
                        (for-each (lambda (kv2)
                                    (format #t "build.~a.~a=~a~%" (car kv) (car kv2) (fmt (cdr kv2))))
                                  (cdr kv))
                        (format #t "build.~a=~a~%" (car kv) (fmt (cdr kv)))))
                  (build-info))' 2>/dev/null)"
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
exit 0
