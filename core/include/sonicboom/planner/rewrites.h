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

// Graph-level rewrite passes. These mutate a planner Graph *before* planning to
// realize a performance transform as a dataflow rewrite rather than hand-wiring
// it into a model emitter. Each pass is backend-gated by its caller (a pass is
// applied only when the executing backend has registered the fused/extra
// kernels it introduces), and the unfused operator path always remains reachable
// by simply not running the pass.

#include <sonicboom/planner/graph.h>

namespace sonicboom::planner {

// T1 quantize-once reuse. Extracts the activation quantization from every
// QuantizedMatmul into an explicit Quantize node (f32 -> q8_1), rewrites the
// matmul to QuantizedMatmulQ8 (consumes the q8_1), then CSE-merges identical
// Quantize nodes — quantizations of the same input tensor (h feeding q/k/v, f
// feeding up/gate) are computed once and shared. The Quantize output tensor keeps
// its f32 shape/size (a strict superset of the q8_1 bytes), so downstream
// memory planning is unchanged; only the CUDA Quantize/QuantizedMatmulQ8 kernels
// interpret the buffer as q8_1. Purely a rewrite: it never removes the unfused
// QuantizedMatmul path (that op and its matvec_f32_dev kernel remain).
void quantize_once_pass(Graph& g);

// T3 rms_norm + q8_1 quantize fusion. After quantize_once_pass, an rms_norm
// whose output feeds only a Quantize node (attn_norm -> h -> q8_h; ffn_norm ->
// f -> q8_f; output_norm -> hn -> q8_hn) is replaced by a single RmsNormQuantize
// node that normalizes and quantizes in one kernel, so the f32 normalized
// activation is never materialized. The RmsNormQuantize node keeps the RmsNorm
// node's (layer, slot) attributes (so the backend still resolves the norm
// weight) and the Quantize node's output tensor (the q8_1 buffer). Must run
// after quantize_once_pass.
void fuse_rms_norm_quantize_pass(Graph& g);

} // namespace sonicboom::planner
