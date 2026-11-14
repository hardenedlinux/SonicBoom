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

// CUDA gated DeltaNet scan: the device twin of <sonicboom/nn/gdn.h>.
//
// Two launches serve any n: gdn_prep computes the convolution, the split, the
// per-key-head normalization and the gates for every row (no row depends on
// another row's result there), and gdn_recur runs the memory update, whose
// rows are sequential within one value index. The second launch reads the
// first's outputs, so a scan over n rows is the n single-row updates in order,
// exactly as on the CPU — row r of an n-row scan is bit-identical to the r-th
// of n one-row scans on this backend.
//
// The reductions have one fixed order: a lane sums its dim/32 products in
// ascending r, then the warp reduces with __shfl_xor_sync at 16, 8, 4, 2, 1.
// No fast-math is used. The kernel arithmetic therefore does not depend on n,
// the launch shape or the residency of any weight.
//
// The state is split into input and output pointers so a tentative scan
// (Tentative) leaves the committed buffers untouched: a commit passes the
// committed buffer as both, a tentative scan passes committed and scratch.
// The entry point launches on sonicboom's current CUDA stream and does not
// synchronize with the host.
//
// The declarations use only Layer 2 types plus raw float pointers that name
// device memory — no CUDA, c10 or native-torch types appear here. The kernels
// live in core/src/nn/cuda_gdn.cu and are only linked when the build is
// configured with SONICBOOM_USE_CUDA=ON.

#include <sonicboom/nn/gdn.h>

namespace sonicboom::nn::cuda {

// Runs rows 0..n-1 of one block's recurrence on the device. qkv is
// [n*C] (C = 2*heads_k*dim + heads_v*dim), beta_logit and alpha_logit are
// [n*heads_v], out is [n*heads_v*dim], mem_in/mem_out are
// [heads_v*dim*dim] with mem[(h*dim + j)*dim + i]. hist_in and hist_out are
// [(kernel-1)*C] with hist[j*C+c] (j=0 oldest); either pair may be the same
// buffer. All pointers name device memory. `weights` are host spans, uploaded
// on first use and cached by (pointer, byte size); a weight buffer's contents
// must therefore not change, and its address must not be reused for another
// weight, while the process runs (model weights are mapped once and stay put).
//
// Returns false (launching nothing) when the geometry or a span's size does
// not match, n is negative, a pointer is null, or a device allocation or
// launch fails.
bool gdn_scan_dev(const GdnDims& dims, const GdnWeights& weights,
                  const float* qkv, const float* beta_logit,
                  const float* alpha_logit, int n, const float* hist_in,
                  float* hist_out, const float* mem_in, float* mem_out,
                  float* out);

} // namespace sonicboom::nn::cuda
