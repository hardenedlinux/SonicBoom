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

// Reference quantized matrix-vector product (CPU).
//
// The weight operand is a QuantizedTensor (Q3_K/Q4_K/Q5_K); the activation is
// float32. This is the "quantized weight @ f32 activation" shape of a
// transformer matmul — the weights stay in their packed bytes and are
// dequantized on the fly per block, never expanded to a full f32 matrix.
//
// This is the scalar correctness reference for the CPU path. CUDA and (later)
// MLIR lowering reproduce this arithmetic; see core/src/quant/quantized_matmul.cpp.

namespace sonicboom::quant {

// Which matmul backend the quantized matvec/matmul dispatches to. The model
// execution layer picks one backend per forward pass; the Cuda backend is
// available only when the build is configured with SONICBOOM_USE_CUDA=ON and a
// device is present (it falls back to the f32 CPU reference otherwise, so its
// numerical semantics stay f32-dequant regardless of device).
enum class MatmulBackend : uint8_t {
  CpuQ8K,  // llama.cpp-equivalent q8_K activation (bit-exact oracle reference)
  CpuF32,  // f32 activation, scalar reference
  Cuda,    // dequantize-on-the-fly gemv on device (f32-activation semantics)
};

// Whether the Cuda backend is actually backed by a present CUDA device. False
// when the build has no CUDA support (SONICBOOM_USE_CUDA off) or no device is
// present, in which case matvec/matmul with MatmulBackend::Cuda silently fall
// back to the f32 CPU reference. Callers use this to distinguish "ran on the
// device" from "fell back to CPU" (e.g. for honest timing). No CUDA types are
// exposed; this is a pure runtime query.
bool cuda_available();

// Backend-dispatching matvec: y = W @ x through the requested backend. This is
// the single seam the model execution layer calls for quantized matmuls; it
// keeps the per-backend reference kernels (matvec_f32 / matvec_f32_q8_K above,
// and cuda::matvec_f32) as the only implementations. Cuda routes to
// cuda::matvec_f32 and falls back to the f32 CPU reference when CUDA is
// unavailable. Same shape/size contracts and failure semantics as the
// reference kernels.
bool matvec(const QuantizedTensor& W, std::span<const float> x,
            std::span<float> y, MatmulBackend backend);

// Backend-dispatching batched matmul: Y = W @ X ([cols, n] activation). Cuda
// only handles the n == 1 (single-column) case and falls back to the f32 CPU
// reference for n > 1. Same contracts as matmul_f32 / matmul_f32_q8_K.
bool matmul(const QuantizedTensor& W, std::span<const float> X,
            std::span<float> Y, uint64_t n, MatmulBackend backend);

// y = W @ x.
//
// W is a 2-D quantized matrix whose GGUF-order dims are [cols, rows]
// (dims[0] is the fastest-moving / contiguous dimension, i.e. the inner loop
// over `cols`). `x` has length `cols`, `y` has length `rows`.
//
// Returns false (writing nothing) when W is invalid, is not 2-D, or `x`/`y`
// are too short for the implied shape. Never reads past W.data() or x/y.
bool matvec_f32(const QuantizedTensor& W, std::span<const float> x,
                std::span<float> y);

// Y = W @ X (batched, the prefill shape). W is [cols, rows] as above; X is a
// row-major [cols, n] matrix (X[k*n + c] = element (k, c)), Y is a row-major
// [rows, n] matrix (Y[r*n + c]). `n >= 1` is the batch column count.
//
// Returns false (writing nothing) when W is invalid, is not 2-D, `n == 0`, or
// X/Y are too short. Never reads past W.data() or X/Y.
bool matmul_f32(const QuantizedTensor& W, std::span<const float> X,
                std::span<float> Y, uint64_t n);

// q8_K-activation variants. These quantize the f32 activation to q8_K (int8 +
// per-256-block scale) before the dot, exactly as llama.cpp's quantized matmul
// does (ggml_vec_dot_q*_K_q8_K / quantize_row_q8_K). llama.cpp's output differs
// from a full-f32-activation matmul by roughly 0.1-2% on transformer activations,
// so use these variants when SonicBoom's result must be directly comparable to a
// llama.cpp oracle; matmul_f32/matvec_f32 above remain the more accurate
// f32-activation reference. Same shape/size contracts and failure semantics.
bool matvec_f32_q8_K(const QuantizedTensor& W, std::span<const float> x,
                     std::span<float> y);

bool matmul_f32_q8_K(const QuantizedTensor& W, std::span<const float> X,
                     std::span<float> Y, uint64_t n);

} // namespace sonicboom::quant
