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

// Co-execution partitioning seam. `BackendTag` names the execution backend that
// a compute task is dispatched to; `route_op` is the single, replaceable
// decision point that maps an operator to its default backend. `slice_document`
// extracts a contiguous S-Expr sub-graph into a standalone 1-in/1-out Document
// so the MLIR backend can compile a native-torch-bounded region independently.

#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace sonicboom::planner {

// The backend a compute task is dispatched to. Distinct from DeviceKind: the
// Mlir and NativeTorch backends both run on CPU in v0, so device kind alone
// cannot route a task; Sonic is the Phase 6 transformer backend (also CPU in
// M2, later GPU in M3).
enum class BackendTag : uint8_t {
  Mlir,        // a whole region as one sx::Executable (CpuBackend), compiled
               // from a frozen 7-op S-Expr slice, run by the CPU Boom executor
  NativeTorch, // one node per task (CpuKernelBackend), run by the CPU Boom
               // executor. The tag name and its "native_torch" IR spelling are
               // kept for plan/IR compatibility; no native-torch dispatcher
               // stands behind it any more.
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

// Extract the nodes that produce `keep_outputs` (and only those nodes) into a
// standalone Document. Every SSA value a kept node references but does not
// produce becomes a graph input; `keep_outputs` becomes the graph output list.
// Parameters referenced by the slice are carried over verbatim. This is a pure
// sx::Document transform — it never touches the frozen IR format.
std::expected<sx::Document, PlannerError> slice_document(
    const sx::Document& doc, const std::vector<std::string>& keep_outputs);

// Extract an explicit node set (a backend region) into a standalone Document.
// `kept_node_outputs` names one output per kept node (v0 nodes are
// single-output), selecting exactly those nodes regardless of transitive
// dataflow; `region_outputs` (a subset of kept-node outputs) becomes the graph
// output list. Any value a kept node references but no kept node produces
// becomes a graph input; parameters referenced by the region are carried over
// verbatim. Used by the region-partitioned mixed path, where consecutive
// same-backend nodes are compiled as one unit.
std::expected<sx::Document, PlannerError> slice_region(
    const sx::Document& doc,
    const std::vector<std::string>& kept_node_outputs,
    const std::vector<std::string>& region_outputs);

} // namespace sonicboom::planner
