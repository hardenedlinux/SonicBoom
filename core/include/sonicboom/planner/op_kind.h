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
// operators (QuantizedMatmul .. Slice) are the Phase 6 node vocabulary
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
  Quantize,         // activation -> q8_1 (explicit quantize-once node, CUDA)
  QuantizedMatmulQ8, // K-quant matvec consuming a pre-quantized q8_1 activation
  RmsNormQuantize,  // nn::rms_norm + q8_1 quantize fused (CUDA, T3)
  GeluMulQuantize,  // gelu_fp16(gate)*up + q8_1 quantize fused (CUDA, T5)
  SiluMulQuantize,  // silu(gate)*up + q8_1 quantize fused (CUDA, T5, SwiGLU)
  RmsNorm,          // nn::rms_norm
  RmsNormHeads,     // nn::rms_norm_heads (per-head, in place)
  GeluFp16,         // nn::gelu_fp16
  Silu,             // nn::silu (SwiGLU gate, Qwen2)
  Rope,             // nn::rope_neox_heads (multi-head NEOX RoPE)
  Attention,        // nn::decode_attention (incremental KV-cache SDPA); appends
                    // iff kc/vc inputs are present, attends the kv_source_layer cache
  GqaBroadcast,     // nn::gqa_broadcast (n_k == 1 donor-V spread)
  Embedding,        // nn::embedding_f32
  MatvecF32,        // nn::matvec_f32 (dense f32 gate/proj matvec)
  MatvecBf16,       // nn::matvec_bf16 (dense bf16 projection; attrs offset/rows)
  AddBias,          // nn::add_bias (optional projection bias; `slot` names the
                    // [width] vector). Generic: any linear layer whose
                    // architecture carries a bias emits it right after the
                    // matmul (Qwen2's Q/K/V and output projections do).
  Mul,              // nn::mul
  Scale,            // nn::scale
  CastFp16,         // nn::cast_fp16
  Softcap,          // final-logit softcapping sc * tanh(x / sc)
  Argmax,           // nn::argmax
  Slice,            // sub-span copy (attrs: offset, length, optional stride)

  // Transformer operators — chunked prefill (Phase 7 / Section C.3). The
  // per-token OpKinds above (Embedding, QuantizedMatmul, RmsNorm, RmsNormHeads,
  // Rope, MatvecF32, MatvecBf16, Slice) carry an optional `batch`
  // attribute (n == the prompt length); when present the SonicBackend loops the
  // same per-token kernel over the [n, dim] position-major batch (dim
  // innermost), otherwise it runs the single-token decode path. The batch size
  // n (and `dim` where needed) is baked into the node attributes by the prefill
  // graph emitter. Position-major (C.3) removed the [dim,n] column-major layout
  // and its Transpose nodes: every batched matmul/gather/norm writes [n, dim]
  // directly, so the attention path (which always consumed [n, dim]) no longer
  // needs a data-movement transpose on either side of it.
  FlashAttention,        // chunked flash attention over prior cache + current batch
                         // (kv_source_layer selects the cache: own or donor)
  KvAppend,              // append n KV rows to the cache (deferred to after all layers)

  // A fused glue region the compiler lowered through MLIR to a device kernel
  // (Stage 5, FuseGlueRegionsPass). The region's shape is named by the node's
  // `fused_region` string attribute; the node keeps the op it replaced as its
  // name, so the region and the kernel that runs it are both identified there.
  FusedRegion,
};

// Map a canonical operator name (e.g. "conv", "rms_norm") to its OpKind;
// nullopt if the name is not a recognized operator.
std::optional<OpKind> op_kind_from_name(std::string_view name) noexcept;

const char* op_kind_name(OpKind k) noexcept;

} // namespace sonicboom::planner
