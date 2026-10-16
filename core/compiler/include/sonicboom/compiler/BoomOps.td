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

//===----------------------------------------------------------------------===//
// Boom IR operations (design/mlir-migration.md §6.1).
//
// One op per planner::OpKind, with the mnemonic equal to op_kind_name(OpKind),
// so `op_name -> OpKind` is a lookup rather than a table that can drift.
//
// Operands and results are variadic and typed by the dialect's own types: ODS
// then generates the value-type check (a value of any other type is rejected)
// for free, and arity — which genuinely varies, e.g. attention appends to the
// cache only when its kc/vc inputs are present (op_kind.h:79-81) — is a check
// the graph verifier makes with the shape/dtype/quant checks, not a static
// signature that would have to be corrected every time a kernel changes.
//
// All ops share one attribute record:
//
//   node_name, node_id, result_ids, node_attrs   graph-node identity — what is
//                                                needed to rebuild the
//                                                GraphNodeDesc exactly
//   backend                                      optional backend override
//   slot, layer, kv_source_layer, ds, batch,     the Blade attribute
//   dim, offset, rows, length, stride, scale     vocabulary, typed
//
// The last group is exactly the set of attribute names the plan emitters and
// the C.4 rewrite passes carry (gemma4_plan.cpp, planner/rewrites.cpp,
// planner/sonic_cuda_backend.cpp); `node_attrs` carries the same information
// losslessly and in order. The graph verifier requires the two to agree and
// checks which of them each op may carry — that per-op legality is what the
// shared record cannot express and what a verifier can.
//===----------------------------------------------------------------------===//

#ifndef SONICBOOM_COMPILER_BOOMOPS_TD
#define SONICBOOM_COMPILER_BOOMOPS_TD

include "BoomTypes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The graph container
//===----------------------------------------------------------------------===//

// The whole-graph container: one per planner::Graph. It holds the tensor table
// (every TensorDesc, verbatim) and the graph's input/output tensor ids.
//
// The body is a single block whose arguments are the graph's *free variables* —
// every tensor the graph references but no node produces: its feedable inputs,
// its model-bound weights and constants, its externals — in ascending
// tensor-id order. A node reading any of them reads a block argument, and the
// argument's type carries the shape/dtype the tensor table also records.
//
// Free variables, not just graph inputs, because a weight has to be an operand
// too: boom.quantized_matmul reads a K-quant weight, and with the graph
// IsolatedFromAbove there is nothing outside to name it. Ascending id order
// (rather than "inputs first, then the rest") keeps the block-argument order a
// property of the tensor table alone, so it is checkable without a second rule.
//
// A node's results carry their tensor ids in `result_ids`, in result order.
// Together those give the verifier a total mapping from SSA values to tensor
// ids, and the round trip never needs a positional guess.
//
// IsolatedFromAbove: the graph is a closed unit; nothing outside it is
// referenceable. SingleBlock + NoTerminator: the body is a flat, ordered node
// list, and the graph's outputs are named by id in `outputs` rather than by a
// terminator.
def Boom_GraphOp : Op<Boom_Dialect, "graph",
                      [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "a whole planner graph and its tensor table";

  let arguments = (ins
    StrAttr:$graph_name,
    ArrayAttr:$tensors,
    DenseI64ArrayAttr:$inputs,
    DenseI64ArrayAttr:$outputs);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Node ops — one per planner::OpKind
//===----------------------------------------------------------------------===//

class Boom_Op<string mnemonic, string opSummary, list<Trait> traits = []>
    : Op<Boom_Dialect, mnemonic, traits> {
  let summary = opSummary;

  let arguments = (ins
    Variadic<Boom_ValueType>:$inputs,

    // --- graph-node identity (exact GraphNodeDesc reconstruction) ----------
    StrAttr:$node_name,
    I64Attr:$node_id,
    DenseI64ArrayAttr:$result_ids,
    ArrayAttr:$node_attrs,

    // --- optional backend override (absent = route_op(op)) ----------------
    OptionalAttr<StrAttr>:$backend,

    // --- the Blade attribute vocabulary, typed ---------------------------
    OptionalAttr<I64Attr>:$slot,
    OptionalAttr<I64Attr>:$layer,
    OptionalAttr<I64Attr>:$kv_source_layer,
    OptionalAttr<I64Attr>:$ds,
    OptionalAttr<I64Attr>:$batch,
    OptionalAttr<I64Attr>:$dim,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$rows,
    OptionalAttr<I64Attr>:$length,
    OptionalAttr<I64Attr>:$stride,
    OptionalAttr<F64Attr>:$scale);

  let results = (outs Variadic<Boom_ValueType>:$results);
}

//--- frozen S-Expr v0.1 CNN operators ---------------------------------------
//
// These lower to linalg through the existing lowering.cpp path (tier 2 of
// §6.1), except boom.softmax, which routes to native-torch, and
// boom.whole_graph, which is a task-granularity marker rather than a node-level
// operator.

def Boom_ConvOp : Boom_Op<"conv",
    "frozen S-Expr v0.1 convolution">;
def Boom_ReluOp : Boom_Op<"relu",
    "frozen S-Expr v0.1 ReLU">;
def Boom_AddOp : Boom_Op<"add",
    "frozen S-Expr v0.1 elementwise add">;
def Boom_MaxPoolOp : Boom_Op<"max_pool",
    "frozen S-Expr v0.1 max pooling">;
def Boom_ReduceMeanOp : Boom_Op<"reduce_mean",
    "frozen S-Expr v0.1 mean reduction">;
def Boom_ReshapeOp : Boom_Op<"reshape",
    "frozen S-Expr v0.1 reshape">;
def Boom_GemmOp : Boom_Op<"gemm",
    "frozen S-Expr v0.1 GEMM">;
def Boom_SoftmaxOp : Boom_Op<"softmax",
    "softmax, routed to native-torch (no MLIR lowering)">;
def Boom_WholeGraphOp : Boom_Op<"whole_graph",
    "task-granularity marker for the single whole-graph compute task",
    [Pure]>;

//--- transformer operators --------------------------------------------------

def Boom_QuantizedMatmulOp : Boom_Op<"quantized_matmul",
    "K-quant matvec/matmul over a model-bound weight (quant::matvec)">;
def Boom_QuantizeOp : Boom_Op<"quantize",
    "f32 activation -> q8_1, the explicit quantize-once node (CUDA)">;
def Boom_QuantizedMatmulQ8Op : Boom_Op<"quantized_matmul_q8",
    "K-quant matvec consuming a pre-quantized q8_1 activation">;
def Boom_RmsNormQuantizeOp : Boom_Op<"rms_norm_quantize",
    "rms_norm + q8_1 quantize, fused">;
def Boom_GeluMulQuantizeOp : Boom_Op<"gelu_mul_quantize",
    "gelu_fp16(gate) * up + q8_1 quantize, fused">;
def Boom_SiluMulQuantizeOp : Boom_Op<"silu_mul_quantize",
    "silu(gate) * up + q8_1 quantize, fused (SwiGLU)">;
def Boom_RmsNormOp : Boom_Op<"rms_norm",
    "RMS normalization">;
def Boom_RmsNormHeadsOp : Boom_Op<"rms_norm_heads",
    "per-head RMS normalization, in place">;
def Boom_GeluFp16Op : Boom_Op<"gelu_fp16",
    "fp16 GELU activation">;
def Boom_SiluOp : Boom_Op<"silu",
    "SiLU activation (SwiGLU gate)">;
def Boom_RopeOp : Boom_Op<"rope",
    "multi-head NEOX RoPE">;
def Boom_AttentionOp : Boom_Op<"attention",
    "incremental KV-cache SDPA; appends iff kc/vc are present",
    [Boom_KvReadWrite]>;
def Boom_GqaBroadcastOp : Boom_Op<"gqa_broadcast",
    "n_k == 1 donor-V spread across query heads">;
def Boom_EmbeddingOp : Boom_Op<"embedding",
    "token embedding lookup">;
def Boom_MatvecF32Op : Boom_Op<"matvec_f32",
    "dense f32 matvec">;
def Boom_MatvecBf16Op : Boom_Op<"matvec_bf16",
    "dense bf16 projection (attrs offset/rows)">;
def Boom_AddBiasOp : Boom_Op<"add_bias",
    "add an optional projection bias named by `slot`">;
def Boom_MulOp : Boom_Op<"mul",
    "elementwise multiply">;
def Boom_ScaleOp : Boom_Op<"scale",
    "multiply by the `scale` attribute">;
def Boom_CastFp16Op : Boom_Op<"cast_fp16",
    "cast to fp16">;
def Boom_SoftcapOp : Boom_Op<"softcap",
    "final-logit softcapping, sc * tanh(x / sc)">;
def Boom_ArgmaxOp : Boom_Op<"argmax",
    "greedy argmax over the logits">;
def Boom_SliceOp : Boom_Op<"slice",
    "sub-span copy (attrs offset, length, optional stride)">;
def Boom_FlashAttentionOp : Boom_Op<"flash_attention",
    "chunked flash attention over the prior cache plus the current batch",
    [Boom_KvRead]>;
def Boom_KvAppendOp : Boom_Op<"kv_append",
    "append n KV rows to the layer's cache, deferred past all layers",
    [Boom_KvWrite]>;

#endif // SONICBOOM_COMPILER_BOOMOPS_TD
