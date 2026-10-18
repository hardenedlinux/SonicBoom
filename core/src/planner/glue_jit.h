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

// The runtime half of the glue code generator (compiler/GlueCodegen.h): turn a
// `fused_region` node's region name into a loaded CUDA kernel and launch it.
//
// The first launch of a region (per process) compiles its PTX through MLIR,
// then caches the text under ~/.cache/sonicboom/kernels/<region>-sm<arch>.ptx
// so later processes skip the compilation. The module is loaded with the CUDA
// driver (cuModuleLoadData, which JIT-translates the PTX for the device) and
// launched on the planner's current stream, so the launch is capturable into a
// CUDA Graph exactly like the hand-written kernels it replaces.
//
// NOT part of the public Layer 2 surface: lives under core/src/.

#include <string_view>

#include <sonicboom/compiler/AttentionCodegen.h>

namespace sonicboom::planner::glue {

// True when the compiler has a code generator for `region` (the value of a
// node's `fused_region` attribute).
bool known_region(std::string_view region);

// Launch the region's generated kernel. `args` is the kernel's parameter array,
// in the generated kernel's order; `blocks`/`threads`/`shared_bytes` are the
// launch geometry. The first call compiles and loads the kernel (a multi-call
// path with a disk cache); a failure to compile or load returns false, and the
// caller is expected to fall back to the hand-written kernel for that node.
bool launch(std::string_view region, void* const* args, int blocks, int threads,
            int shared_bytes);

// Launch the compiler-generated attention kernel for `shape` (Stage 3). The
// kernel entry takes the attention call's parameters in the order
//   q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
//   n_heads_kv, n, window, scale, out
// (same as the hand-written kernels). The grid is one block per (query tile,
// query head); the block geometry comes from the shape class. The first launch
// compiles and caches the PTX exactly like a glue region; false means the
// shape has no generator or the compile/load failed, and the caller falls back
// to the hand-written kernel.
bool launch_attention(const compiler::AttentionShape &shape, void* const* args,
                      int grid_x, int grid_y);

// Launch the flash-decoding pair for a decode shape. `args` has 13 entries:
//   0 q, 1 k_cache, 2 v_cache, 3 start, 4 n_slots, 5 n_heads_q,
//   6 n_heads_kv, 7 window, 8 scale, 9 partial, 10 chunk,
//   11 out, 12 n_chunks
// The partial kernel runs with grid (n_chunks, n_heads_q) and one warp per
// block; the combine kernel with grid (n_heads_q) and head_dim threads. The
// partial buffer is [n_chunks][n_heads_q][head_dim + 2] f32 and must be
// device-resident before capture (warmed by the first launch). False means
// the shape has no generator or the compile/load failed.
bool launch_attention_decode(const compiler::AttentionShape &shape,
                             void* const* args, int n_chunks, int n_heads_q);

} // namespace sonicboom::planner::glue
