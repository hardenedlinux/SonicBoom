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

// T5 gelu×up -> q8_1 fusion. After quantize_once_pass, the gated FFN down-
// projection consumes Quantize(Mul(GeluFp16(gate), up)). When the gelu and the
// mul product are each single-use (only the quantize reads the product, only the
// mul reads the gelu), the three nodes are replaced by one GeluMulQuantize node
// that computes quantize_q8_1(gelu_fp16(gate) * up) in one kernel — the f32 gelu
// and fh intermediates are never materialized. The GeluMulQuantize node keeps the
// GeluFp16 node's (layer) attribute, takes the gelu input (gate) and the mul's
// up operand as its two inputs, and reuses the Quantize node's q8_1 output
// tensor. Must run after quantize_once_pass (and after fuse_rms_norm_quantize_pass).
void fuse_gelu_mul_quantize_pass(Graph& g);

// T5, SwiGLU variant. The same fusion as fuse_gelu_mul_quantize_pass with Silu
// in place of GeluFp16: where the gated FFN activates with Silu (Qwen2, Llama,
// Mistral, …), the triple Quantize(Mul(Silu(gate), up)) becomes one
// SiluMulQuantize node. Mirrors the gelu pass exactly — same single-use
// conditions, same output tensor (the Quantize's q8_1 buffer), same attribute
// set (the node keeps the Silu node's attributes) — so it composes with the
// same backend kernel plumbing. Must run after quantize_once_pass, and is the
// mutually exclusive alternative to fuse_gelu_mul_quantize_pass (a graph
// activates with one or the other, never both).
void fuse_silu_mul_quantize_pass(Graph& g);

// C.4 prefill (mmq) rms_norm fusion. The prefill graph has no Quantize nodes —
// each batched QuantizedMatmul quantizes its activation internally (block_q8_1_mmq,
// 128-elem). This pass folds an RmsNorm(batch, dim) whose output is consumed
// *only* by batched QuantizedMatmul nodes that all share one `ds` (activation
// scale layout) into a single RmsNormQuantize(batch, dim, ds) node, and rewrites
// each consumer to QuantizedMatmulQ8 (which reads the pre-quantized mmq blocks
// with no internal quantize). The norm is then computed+quantized once instead of
// once per consumer, and the f32 normalized activation is never materialized. The
// RmsNormQuantize output reuses the RmsNorm's f32 output tensor (a strict superset
// of the mmq blocks when the fusion's size gate passes). Fused only when the f32
// output buffer also covers the mmq activation's J_max tail guard
// (mmq_q8_blocks(dim, n) * 144 bytes <= size_bytes). Gate on q8_1_enabled &&
// mmq_enabled at the call site, and on n >= mmq_min_n via the caller's nseg gate.
void fuse_rms_norm_mmq_pass(Graph& g);

// Projection-bias fusion (decode). A projection bias is emitted as an AddBias
// node over the projection's QuantizedMatmulQ8 output. When that output has no
// other reader and the matmul is single-column (no `batch`), the bias is folded
// into the matmul node as a `bias_slot` attribute carrying the bias weight slot:
// the matmul takes over the AddBias's output tensor, the AddBias node is dropped,
// and the buffer the matmul used to write is dead. The executing backend adds the
// bias in the gemv epilogue (one launch instead of two) where its kernel has the
// epilogue for the weight's quant type, and falls back to a separate add_bias
// kernel otherwise — the values are identical either way, since the bias is added
// once to the row's fully reduced sum. Run after quantize_once_pass, which is
// what turns the projection into a QuantizedMatmulQ8.
void fuse_bias_into_gemv_pass(Graph& g);

// Residual-add fusion (decode). The residual add around a projection is emitted
// as `Add(proj, residual)`. When `proj` is a single-column QuantizedMatmulQ8
// output with no other reader, the residual becomes the matmul's second input
// (named by the `add_input` attribute) and the matmul takes over the Add's output
// tensor, so the Add node disappears. The executing backend adds the residual in
// the gemv epilogue (one launch instead of two) where its kernel has the epilogue
// for the weight's quant type, and falls back to a separate elementwise add
// otherwise — the values are identical either way, since the residual is added
// once to the row's fully reduced sum, in the same fp32 addition the separate
// kernel performs. Run after fuse_bias_into_gemv_pass (so a projection with both
// a bias and a residual folds in that order, which is the order the unfused
// kernels apply them in).
void fuse_residual_add_into_gemv_pass(Graph& g);

// C.4 prefill (mmq) gelu×up fusion. The gated FFN down-projection
// QuantizedMatmul(fh) internally quantizes fh = gelu_fp16(gate) * up. When the
// gelu and the mul product are each single-use, the GeluFp16 + Mul +
// QuantizedMatmul triple is replaced by one GeluMulQuantize(batch, dim=ff, ds)
// node (computes quantize_mmq(gelu_fp16(gate) * up) in one kernel) plus a
// QuantizedMatmulQ8 consuming it — the f32 gelu and fh intermediates are never
// materialized. The GeluMulQuantize reuses the Mul's f32 output tensor (a
// superset of the mmq blocks), and inherits `dim`/`ds` from the down-projection
// matmul's attributes. Run after fuse_rms_norm_mmq_pass.
void fuse_gelu_mul_mmq_pass(Graph& g);

} // namespace sonicboom::planner
