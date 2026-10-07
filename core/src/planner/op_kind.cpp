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

#include <sonicboom/planner/op_kind.h>

namespace sonicboom::planner {

std::optional<OpKind> op_kind_from_name(std::string_view name) noexcept {
  if (name == "conv")
    return OpKind::Conv;
  if (name == "relu")
    return OpKind::Relu;
  if (name == "add")
    return OpKind::Add;
  if (name == "max_pool")
    return OpKind::MaxPool;
  if (name == "reduce_mean")
    return OpKind::ReduceMean;
  if (name == "reshape")
    return OpKind::Reshape;
  if (name == "gemm")
    return OpKind::Gemm;
  if (name == "softmax")
    return OpKind::Softmax;
  if (name == "quantized_matmul")
    return OpKind::QuantizedMatmul;
  if (name == "rms_norm")
    return OpKind::RmsNorm;
  if (name == "rms_norm_heads")
    return OpKind::RmsNormHeads;
  if (name == "gelu_fp16")
    return OpKind::GeluFp16;
  if (name == "rope")
    return OpKind::Rope;
  if (name == "attention")
    return OpKind::Attention;
  if (name == "attention_shared")
    return OpKind::AttentionShared;
  if (name == "gqa_broadcast")
    return OpKind::GqaBroadcast;
  if (name == "embedding")
    return OpKind::Embedding;
  if (name == "matvec_f32")
    return OpKind::MatvecF32;
  if (name == "matvec_bf16")
    return OpKind::MatvecBf16;
  if (name == "mul")
    return OpKind::Mul;
  if (name == "scale")
    return OpKind::Scale;
  if (name == "cast_fp16")
    return OpKind::CastFp16;
  if (name == "softcap")
    return OpKind::Softcap;
  if (name == "argmax")
    return OpKind::Argmax;
  if (name == "layer_combine")
    return OpKind::LayerCombine;
  if (name == "embedding_batched")
    return OpKind::EmbeddingBatched;
  if (name == "quantized_matmul_batched")
    return OpKind::QuantizedMatmulBatched;
  if (name == "transpose")
    return OpKind::Transpose;
  if (name == "rms_norm_cols")
    return OpKind::RmsNormCols;
  if (name == "rms_norm_heads_batched")
    return OpKind::RmsNormHeadsBatched;
  if (name == "rope_batched")
    return OpKind::RopeBatched;
  if (name == "flash_attention")
    return OpKind::FlashAttention;
  if (name == "flash_attention_shared")
    return OpKind::FlashAttentionShared;
  if (name == "matvec_f32_batched")
    return OpKind::MatvecF32Batched;
  if (name == "matvec_bf16_batched")
    return OpKind::MatvecBf16Batched;
  if (name == "layer_combine_batched")
    return OpKind::LayerCombineBatched;
  return std::nullopt;
}

const char* op_kind_name(OpKind k) noexcept {
  switch (k) {
    case OpKind::Conv: return "conv";
    case OpKind::Relu: return "relu";
    case OpKind::Add: return "add";
    case OpKind::MaxPool: return "max_pool";
    case OpKind::ReduceMean: return "reduce_mean";
    case OpKind::Reshape: return "reshape";
    case OpKind::Gemm: return "gemm";
    case OpKind::Softmax: return "softmax";
    case OpKind::WholeGraph: return "whole_graph";
    case OpKind::QuantizedMatmul: return "quantized_matmul";
    case OpKind::RmsNorm: return "rms_norm";
    case OpKind::RmsNormHeads: return "rms_norm_heads";
    case OpKind::GeluFp16: return "gelu_fp16";
    case OpKind::Rope: return "rope";
    case OpKind::Attention: return "attention";
    case OpKind::AttentionShared: return "attention_shared";
    case OpKind::GqaBroadcast: return "gqa_broadcast";
    case OpKind::Embedding: return "embedding";
    case OpKind::MatvecF32: return "matvec_f32";
    case OpKind::MatvecBf16: return "matvec_bf16";
    case OpKind::Mul: return "mul";
    case OpKind::Scale: return "scale";
    case OpKind::CastFp16: return "cast_fp16";
    case OpKind::Softcap: return "softcap";
    case OpKind::Argmax: return "argmax";
    case OpKind::LayerCombine: return "layer_combine";
    case OpKind::EmbeddingBatched: return "embedding_batched";
    case OpKind::QuantizedMatmulBatched: return "quantized_matmul_batched";
    case OpKind::Transpose: return "transpose";
    case OpKind::RmsNormCols: return "rms_norm_cols";
    case OpKind::RmsNormHeadsBatched: return "rms_norm_heads_batched";
    case OpKind::RopeBatched: return "rope_batched";
    case OpKind::FlashAttention: return "flash_attention";
    case OpKind::FlashAttentionShared: return "flash_attention_shared";
    case OpKind::MatvecF32Batched: return "matvec_f32_batched";
    case OpKind::MatvecBf16Batched: return "matvec_bf16_batched";
    case OpKind::LayerCombineBatched: return "layer_combine_batched";
  }
  return "?";
}

} // namespace sonicboom::planner
