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

// Backend routing seam. `BackendTag` names the execution backend that a compute
// task is dispatched to; `route_op` is the single, replaceable decision point
// that maps an operator to its default backend.

#include <sonicboom/planner/op_kind.h>

#include <cstdint>

namespace sonicboom::planner {

// The backend a compute task is dispatched to. Distinct from DeviceKind:
// device kind alone cannot route a task; Sonic is the Phase 6 transformer
// backend (also CPU in M2, later GPU in M3).
enum class BackendTag : uint8_t {
  Mlir,        // a whole region as one sx::Executable (CpuBackend), compiled
               // from a frozen 7-op S-Expr slice, run by the CPU Boom executor
  NativeTorch, // legacy tag kept for plan/IR compatibility and its
               // "native_torch" spelling. The per-node CPU kernel backend that
               // served it was retired with the co-execution path; nothing
               // constructs a backend for this tag.
  Sonic,       // nn::*/quant::* kernels (SonicBackend), the transformer path
};
const char* backend_tag_name(BackendTag t) noexcept;

// Default backend for an operator. This is the *entire* backend-selection seam:
// replacing the routing strategy means replacing this one function, not
// threading conditionals through the planner and executor. A graph node may
// override this default via GraphNodeDesc::backend (used by the Gemma 4 plan
// emitter to pin every node — including the shared `Add` — to Sonic); when no
// override is present, this function is the routing rule.
inline BackendTag route_op(OpKind op) noexcept {
  switch (op) {
    case OpKind::Softmax:
      return BackendTag::NativeTorch;
    // Phase 6 transformer operators. These are not part of the frozen S-Expr
    // v0.1 vocabulary, so this branch is only reachable through a model-aware
    // plan emitter (e.g. the Gemma 4 decode graph); it exists so the routing
    // seam stays total over OpKind.
    case OpKind::QuantizedMatmul:
    case OpKind::Quantize:
    case OpKind::QuantizedMatmulQ8:
    case OpKind::RmsNormQuantize:
    case OpKind::GeluMulQuantize:
    case OpKind::SiluMulQuantize:
    case OpKind::RmsNorm:
    case OpKind::RmsNormHeads:
    case OpKind::GeluFp16:
    case OpKind::Silu:
    case OpKind::Rope:
    case OpKind::Attention:
    case OpKind::GqaBroadcast:
    case OpKind::Embedding:
    case OpKind::MatvecF32:
    case OpKind::MatvecBf16:
    case OpKind::AddBias:
    case OpKind::Mul:
    case OpKind::Scale:
    case OpKind::CastFp16:
    case OpKind::Softcap:
    case OpKind::Argmax:
    case OpKind::Slice:
    // A compiler-generated glue region has no MLIR-S-Expr lowering; it is a
    // device kernel like the quant::* ops around it.
    case OpKind::FusedRegion:
    // Training updates run as hand-written kernels over the parameter's own
    // device buffers; they have no MLIR-S-Expr lowering either.
    case OpKind::AdamW:
    // The training backward ops are hand-written kernels over device buffers;
    // there is no S-Expr lowering for them either.
    case OpKind::MatmulBwd:
    case OpKind::SoftmaxBwd:
    case OpKind::CrossEntropyBwd:
    case OpKind::RmsNormBwd:
    case OpKind::RopeBwd:
    case OpKind::SiluMulBwd:
    case OpKind::EmbeddingBwd:
    // Training forward vocabulary: dense matmul/bmm, layout and reduction
    // primitives, and the fused SwiGLU/CE forms. Like the backward ops they
    // run as hand-written kernels over device buffers; the trained graph is
    // executed by the device training graph executor, not a plan emitter.
    case OpKind::Matmul:
    case OpKind::Bmm:
    case OpKind::Permute:
    case OpKind::IndexSelect:
    case OpKind::ReduceSum:
    case OpKind::SiluMul:
    case OpKind::CrossEntropy:
    // Global-norm gradient clipping is a hand-written training kernel pair
    // (one accumulation pass, one scale per gradient) over device buffers.
    case OpKind::ClipGradNorm:
      return BackendTag::Sonic;
    // VS1 prefetch markers: the runtime's prefetch stream is backend state, so
    // they dispatch like the other device-side bookkeeping ops (KvAppend).
    case OpKind::AsyncWeightPrefetch:
    case OpKind::SyncPrefetch:
      return BackendTag::Sonic;
    default:
      return BackendTag::Mlir;
  }
}

} // namespace sonicboom::planner
