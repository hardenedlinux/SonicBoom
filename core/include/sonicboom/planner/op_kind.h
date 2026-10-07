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

// The planner's operator vocabulary. The concrete operators mirror the frozen
// S-Expr v0.1 supported operator set (used for per-node capability and cost
// analysis); `Softmax` is a native-torch-routed operator (no MLIR lowering);
// `WholeGraph` is a task-granularity marker for the single whole-graph compute
// task the v0 compiler emits, not a node-level operator. The transformer
// operators (QuantizedMatmul .. LayerCombine) are the Phase 6 node vocabulary
// emitted by the Gemma 4 plan emitter and dispatched by the SonicBackend.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sonicboom::planner {

enum class OpKind : uint8_t {
  // Frozen S-Expr v0.1 CNN operators (existing).
  Conv,
  Relu,
  Add,
  MaxPool,
  ReduceMean,
  Reshape,
  Gemm,
  Softmax,
  WholeGraph,
  // Transformer operators (Phase 6 — the Gemma 4 decode path). Each is a
  // node-level operator the SonicBackend dispatches to a named nn::* / quant::*
  // kernel; the mapping below is the shared vocabulary, not the routing.
  QuantizedMatmul,  // K-quant matvec (quant::matvec)
  RmsNorm,          // nn::rms_norm
  RmsNormHeads,     // nn::rms_norm_heads (per-head, in place)
  GeluFp16,         // nn::gelu_fp16
  Rope,             // nn::rope_neox_heads (multi-head NEOX RoPE)
  Attention,        // nn::decode_attention (incremental KV-cache SDPA, appends)
  AttentionShared,  // nn::decode_attention over a donor layer's cache (no append)
  GqaBroadcast,     // nn::gqa_broadcast (n_k == 1 donor-V spread)
  Embedding,        // nn::embedding_f32
  MatvecF32,        // nn::matvec_f32 (dense f32 gate/proj matvec)
  MatvecBf16,       // nn::matvec_bf16 (dense bf16 per-layer projection)
  Mul,              // nn::mul
  Scale,            // nn::scale
  CastFp16,         // nn::cast_fp16
  Softcap,          // final-logit softcapping sc * tanh(x / sc)
  Argmax,           // nn::argmax
  LayerCombine,     // nn::layer_combine
};

// Map a canonical operator name (e.g. "conv", "rms_norm") to its OpKind;
// nullopt if the name is not a recognized operator.
std::optional<OpKind> op_kind_from_name(std::string_view name) noexcept;

const char* op_kind_name(OpKind k) noexcept;

} // namespace sonicboom::planner
