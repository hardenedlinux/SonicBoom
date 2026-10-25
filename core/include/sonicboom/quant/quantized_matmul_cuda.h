// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <cstdint>
#include <span>

#include <sonicboom/quant/quantized_tensor.h>

// CUDA backend for the quantized matvec (dequantize-on-the-fly). This is a
// SonicBoom-owned kernel: the packed K-quant weight is uploaded to the device
// once and cached, and each single-column matvec streams it through a
// dequantize-and-dot gemv kernel. It reproduces the arithmetic of the scalar
// f32-activation reference <sonicboom/quant/quantized_matmul.h> matvec_f32
// (f32 accumulation order differs on the device, so agreement is to fp32
// tolerance, not bit-exact).
//
// The declarations here use only Layer 2 types (QuantizedTensor, span) — no
// CUDA, c10 or native-torch types appear in the public header. The CUDA
// runtime and kernels live entirely in core/src/quant/cuda_quant_matmul.cu.
// These entry points are only linked when the build is configured with
// SONICBOOM_USE_CUDA=ON.

namespace sonicboom::quant::cuda {

// True when a usable CUDA device is present (cached after the first call).
bool available() noexcept;

// True when the q8_1 mmvq decode path is active (the default).
// SONICBOOM_GEMV_F32=1 disables it and restores the fp32 gemv for differential
// testing against the CPU reference. Graph rewrites that introduce q8_1-only
// nodes (the T1 quantize-once pass) gate on this so the fp32 fallback path
// stays reachable.
bool q8_1_enabled() noexcept;

// MMQ int8-tensor-core batched prefill matmul selection (C.2). SONICBOOM_MMQ=0
// disables it and restores the blocked cuBLAS fp16 GEMM at n >= mmq_min_n();
// SONICBOOM_MMQ_MIN_N sets the batch threshold (default 2, so the per-position
// numerics stay batch-independent — see mmq_min_n). The prefill fusion pass
// (C.4) gates on the same conditions the matmul uses, so the fused
// q8_1-mmq-consumed path and the unfused matmul never disagree about which
// kernel a position runs.
bool mmq_enabled() noexcept;
int mmq_min_n();

// The MMQ work decomposition (Stage 1): the tiled kernel, or the fixed split-K
// ("stream-k") schedule that spreads one output tile's K reduction over several
// blocks, combining the partial sums in a fixed order. Auto defers to the
// runtime's weight-shape policy.
enum class MmqDecomp { Auto = 0, Tiled = 1, StreamK = 2 };

// The decomposition for a weight shape class — M = weight rows, K = weight
// columns, type = quant format. It is deliberately a function of the weight
// shape only, never of the activation batch size n: split-K changes the
// floating-point summation order, so a choice that moved with n would make
// prefix-cache reuse diverge from a full prefill (test_prefix_cache). The plan
// emitter records the answer as the node's `decomp` attribute; the runtime
// resolves Auto with the same function.
//
// The policy currently returns Tiled for every shape: the fixed split-K
// schedule has been measured slower on the target GPU at pp512 and at n =
// 64..256 (see cuda_quant_matmul.cu), so selecting it would regress the
// canonical metrics. It remains selectable with
// SONICBOOM_MMQ_DECOMP=stream_k for A/B and is covered by the unit tests.
MmqDecomp mmq_class_decomp(uint64_t rows, uint64_t cols,
                           QuantType type) noexcept;

// y = W @ x for a single-column activation, computed on the device. W's packed
// bytes are uploaded on first use and cached keyed by (data pointer, byte
// size), so repeated calls with the same weight reuse the device copy. x and y
// are host spans; the function uploads x, launches the gemv kernel, and copies
// y back.
//
// Returns false (writing nothing to y) when CUDA is unavailable, W is invalid
// or not 2-D, x/y are too short, or a device allocation/copy/kernel launch
// fails. Never reads past W.data() or x/y.
bool matvec_f32(const QuantizedTensor& W, std::span<const float> x,
                std::span<float> y);

// Y = W @ X (batched, the prefill shape). X is a host row-major [cols, n]
// matrix (n innermost, X[k*n + c] = element (k, c)); Y is a host row-major
// [rows, n] matrix (Y[r*n + c]). n == 1 falls back to matvec_f32. Uploads X,
// runs one gemv per column (each gathered from the strided activation), and
// copies Y back. Same contracts and failure semantics as matvec_f32.
bool matmul_f32(const QuantizedTensor& W, std::span<const float> X,
                std::span<float> Y, uint64_t n);


// Single-row gather on the device (host in/out): out[k] = dequantized
// W[k][token] for k in [0, W.dims()[0]). The CUDA twin of
// <sonicboom/quant/dequant.h> dequantize_row_f32 — same block layouts and
// reconstruction formulas, so the result is bit-identical to the CPU reference.
// Only the token's packed row is uploaded (the full tensor may be ~1.5 GiB and
// is not kept resident). `out` must hold >= dims()[0] floats. Returns false
// (writing nothing) when CUDA is unavailable, W is invalid or not 2-D, `token`
// is out of range, or `out` is too short.
bool gather_f32(const QuantizedTensor& W, int64_t token, std::span<float> out);

// Device-resident activation entry points. The host-span functions above copy
// the activation in and out on every call, which is the decode/prefill shape
// (one call per graph node). A device-resident executor (MOSS-Image's DiT)
// keeps its activations on the device across a whole forward, so it calls the
// same kernels directly. `X` is [n, cols] position-major, `Y` is [n, rows]
// position-major (the layout of the CPU matmul). Weight bytes are uploaded on
// first use and cached by (data pointer, byte size), exactly as the host-span
// path does — `upload_weight` exposes that step so a loader can pay the upload
// once at load time. These are internal to the core: no CUDA type appears in
// the signatures.
bool matvec_f32_dev(const QuantizedTensor& W, const float* x, float* y);
bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n);

// Loader hook for the same cache: record that `dptr` already holds W's packed
// bytes, so later calls skip the upload. The caller owns the allocation, which
// must stay valid until unregister_weight_device() or clear_cache(), and must
// carry the kernel's 256-byte zeroed read-ahead tail (allocate byte_size() +
// 256 and memset the tail). This lets a loader overlap the copy with disk I/O
// instead of paying a blocking upload per tensor. Returns false when W is
// invalid, dptr is null, or the key is already registered to another pointer.
bool register_weight_device(const QuantizedTensor& W, const void* dptr);
// Remove and free a registered or cached device copy for W. No-op when absent.
void unregister_weight_device(const QuantizedTensor& W);

// Drop all cached device weight copies (call when a model is reloaded at a new
// address so stale device buffers are not reused).
void clear_cache();

} // namespace sonicboom::quant::cuda
