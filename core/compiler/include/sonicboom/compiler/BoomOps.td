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
// Operands admit one non-value type: the `!boom.kv<layer>` cache handle. The
// three cache-touching ops carry it as their last operand (BoomVerify.cpp
// requires it there and only there); every other operand is a Boom value.
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
    DenseI64ArrayAttr:$outputs,

    // The segment length the tensor table's dynamic (leading-`?`) entries were
    // sized at: each such entry states its byte size at this extent plus the
    // fixed part of its affine size. 0 when the table has no dynamic entry.
    // The lowering binds any extent up to this one by evaluating the affine
    // size; it never exceeds it.
    I64Attr:$dynamic_extent);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The generation program (§10)
//===----------------------------------------------------------------------===//
//
// A `boom.program` is one whole greedy generation: a prefill call, then an
// `scf.while` whose body runs one `boom.step` (the decode graph), samples the
// next token, emits it to the host-visible ring, and tests the stop condition.
// It is not IsolatedFromAbove: the step takes the loop-carried position, token
// and cache handles as SSA operands instead of block arguments of an isolated
// graph.
//
// The program is a container, like boom.graph: its body's block arguments are
// the values the caller supplies — the prompt, the decoding start state and the
// cache handles. It has no tensor table; the step brings its own.

def Boom_ProgramOp : Op<Boom_Dialect, "program",
                        [SingleBlock, NoTerminator]> {
  let summary = "one greedy generation: prefill, then the decode loop";

  let arguments = (ins
    StrAttr:$graph_name,
    // The prefill graph this program calls, by the `graph_name` of a
    // module-level boom.graph. Empty when the caller runs its own prefill.
    StrAttr:$prefill_graph);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

// The prefill call: runs the named prefill graph over `tokens` at `start_pos`,
// filling the cache handles' layers. No results — the prefill's product is the
// cache state, and the first decoded position/token are the program's inputs.
def Boom_PrefillOp : Op<Boom_Dialect, "prefill", [Boom_KvReadWrite]> {
  let summary = "call the named prefill graph over the prompt";

  let arguments = (ins Variadic<Boom_AnyType>:$inputs, StrAttr:$graph_name);

  let hasVerifier = 1;
}

// One decode step. The body is the decode graph's node list, carried with its
// tensor table; its block arguments (in ascending tensor-id order, then the
// cache handles) are the step's operands. The result is the tensor the table's
// `output_ids` names.
def Boom_StepOp : Op<Boom_Dialect, "step", [SingleBlock, NoTerminator]> {
  let summary = "one decode step over the loop-carried position/token/cache";

  let arguments = (ins
    Variadic<Boom_AnyType>:$inputs,
    ArrayAttr:$tensors,
    DenseI64ArrayAttr:$input_ids,
    DenseI64ArrayAttr:$output_ids,
    I64Attr:$dynamic_extent);

  let results = (outs Variadic<Boom_ValueType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

// Greedy argmax when no sampling attributes are set; temperature/top-k/top-p
// are the sampling form, implemented later.
def Boom_SampleOp : Op<Boom_Dialect, "sample", [Pure]> {
  let summary = "choose the next token from the logits";

  let arguments = (ins
    Boom_AnyType:$logits,
    OptionalAttr<F64Attr>:$temperature,
    OptionalAttr<I64Attr>:$top_k,
    OptionalAttr<F64Attr>:$top_p);

  let results = (outs Boom_AnyType:$token);

  let hasVerifier = 1;
}

// The loop's terminating effect chain: EOS set + max tokens over the emitted
// count (`count` is the loop-carried token counter).
def Boom_StopCheckOp : Op<Boom_Dialect, "stop_check", [Pure]> {
  let summary = "EOS set and max-token stop test";

  let arguments = (ins
    AnyType:$token,
    I64:$count,
    DenseI64ArrayAttr:$eos_tokens,
    I64Attr:$max_tokens);

  let results = (outs I1:$stop);
}

// Append the token to the host-visible device ring buffer. A speculative
// iteration emits the whole accepted run (the accept node's tokens); the plain
// loop emits the one sampled token. Its write is an effect the executor must
// not reorder across the loop.
def Boom_EmitOp : Op<Boom_Dialect, "emit"> {
  let summary = "append the token(s) to the emission ring";

  let arguments = (ins Variadic<AnyType>:$tokens);
}

// Advance the loop-carried position by `delta`, or by the `by` operand when one
// is present (a speculative iteration advances by the accepted count + 1).
// Host-side in the executor (positions are staged per step), so no device
// kernel is implied.
def Boom_PosIncOp : Op<Boom_Dialect, "pos_inc", [Pure]> {
  let summary = "advance the staged position";

  let arguments = (ins Boom_AnyType:$pos, Optional<I64>:$by, I64Attr:$delta);

  let results = (outs Boom_AnyType:$next);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The speculation pipeline (B6)
//===----------------------------------------------------------------------===//
//
// A speculative iteration replaces the loop body's step/sample pair with
// draft -> verify -> accept: the drafter proposes k tokens from the accepted
// history, the verify step feeds [last token, drafts...] through the decode
// graph dealt over rows, and accept keeps the longest draft prefix that matches
// the verifier's per-row argmax plus the verifier's own next token. The plain
// step stays in the body: when the drafter proposes nothing (k == 0) the
// executor runs that captured step instead, which is the same token the verify
// would have produced for row 0.

// The proposal stage: up to `max_drafts` tokens following the accepted history.
// `strategy` names the proposal source ("ngram": the prompt-lookup drafter).
// `count` is the number of tokens proposed; zero means take the plain step.
// Drafting reads the session history, a host-side effect the executor must not
// reorder across the loop.
def Boom_DraftOp : Op<Boom_Dialect, "draft"> {
  let summary = "propose up to max_drafts tokens from the accepted history";

  let arguments = (ins StrAttr:$strategy, I64Attr:$max_drafts);

  let results = (outs Variadic<Boom_ValueType>:$drafts, I64:$count);

  let hasVerifier = 1;
}

// The verify stage: the decode graph dealt over rows. `inputs` are the last
// token, the draft tokens and the base position (the cache handles come last);
// `graph_name` names the decode graph and `max_rows` the largest row count it
// is built for. Row i's argmax is the plain greedy token at start_pos + i + 1.
// It appends k+1 rows to the cache; accept decides how many are live.
def Boom_VerifyOp : Op<Boom_Dialect, "verify", [Boom_KvReadWrite]> {
  let summary = "row-invariant verify step over the draft feed";

  let arguments = (ins Variadic<Boom_AnyType>:$inputs,
                       StrAttr:$graph_name,
                       I64Attr:$max_rows);

  let results = (outs Variadic<Boom_ValueType>:$rows);

  let hasVerifier = 1;
}

// The accept stage: the longest prefix with draft[i] == verifier row i's
// argmax, plus the verifier's next token. `inputs` is the draft array, the
// plain step's token and the verifier's row array (in that order — the
// verifier checks the arity, the executor's desc reads the structure).
// `tokens` is the accepted run (the accepted drafts, then the verifier token),
// `next` is the run's last token (the loop's next feed) and `accepted` is the
// number kept. It also sets the session's valid KV length to
// start_pos + accepted + 1: the verify rows past the run are dead and are
// overwritten by later steps.
def Boom_AcceptOp : Op<Boom_Dialect, "accept"> {
  let summary = "longest matching draft prefix plus the verifier's token";

  let arguments = (ins Variadic<Boom_AnyType>:$inputs);

  let results = (outs Variadic<Boom_ValueType>:$tokens,
                      Boom_ValueType:$next,
                      I64:$accepted);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Node ops — one per planner::OpKind
//===----------------------------------------------------------------------===//

class Boom_Op<string mnemonic, string opSummary, list<Trait> traits = []>
    : Op<Boom_Dialect, mnemonic, traits> {
  let summary = opSummary;

  let arguments = (ins
    Variadic<Boom_AnyType>:$inputs,

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
// A maximal glue region lowered by the compiler to one device kernel. It takes
// over the op it replaces (same node id, attributes, operands, results and
// backend); the `fused_region` node attribute names the region, and that name
// selects the generated kernel at dispatch.
def Boom_FusedRegionOp : Boom_Op<"fused_region",
    "compiler-generated glue region (e.g. rms_norm_quantize_q8_1)">;

//--- training operators -----------------------------------------------------

// AdamW: one fused parameter update written as a pure function. Operands are
// [param, grad, m1, m2, step], where `step` is a one-element int32 tensor
// (1-based); results are [param', m1', m2'] with the inputs' shapes. The
// hyperparameters (lr, beta1, beta2, eps, weight_decay) ride on node_attrs.
// Results are new tensors, not aliases: a traced training step reads them back
// as the next state, which keeps the same node on every step.
def Boom_AdamWOp : Boom_Op<"adamw",
    "fused AdamW parameter update (decoupled weight decay)">;

// clip_grad_norm: global-norm gradient clipping written as a pure function.
// Operands are N gradients; results are the N clipped gradients, each with its
// operand's shape, and `max_norm` rides on node_attrs. The global L2 norm is
// accumulated in double in one pass over every element of every operand. When
// it does not exceed max_norm each result is a copy of its operand (the values
// are unchanged); otherwise each is scaled by max_norm / (norm + 1e-6).
def Boom_ClipGradNormOp : Boom_Op<"clip_grad_norm",
    "global-norm gradient clipping over N gradients">;

// matmul_bwd: the VJP of Y = A @ B (A [M,K], B [K,N], dY [M,N]) with respect
// to both operands. Operands [A, B, dY]; results [dA [M,K], dB [K,N]]. The
// row-major layouts are the forward matmul's: A[m*K + k], B[k*N + n],
// dY[m*N + n], dA[m*K + k], dB[k*N + n].
def Boom_MatmulBwdOp : Boom_Op<"matmul_bwd",
    "backward of matmul: dA = dY @ B^T, dB = A^T @ dY">;

// softmax_bwd: the VJP of a last-axis softmax. Operands [Y, dY] where Y is the
// forward output (softmax of the logits) and dY the upstream gradient; result
// [dX], same [rows, cols] shape. dX = Y * (dY - sum_col(dY * Y)).
def Boom_SoftmaxBwdOp : Boom_Op<"softmax_bwd",
    "backward of last-axis softmax">;

// cross_entropy_bwd: fused mean cross-entropy over rows and its gradient.
// Operands [logits [R,C] f32, targets [R] int32]; results [loss (rank-0 f32,
// the mean over rows), dlogits [R,C]]. The gradient written is the one of that
// mean loss: (softmax(logits) - onehot(target)) / R.
def Boom_CrossEntropyBwdOp : Boom_Op<"cross_entropy_bwd",
    "fused mean cross-entropy loss and its gradient">;

// rmsnorm_bwd: the VJP of y = x * rsqrt(mean_d(x^2) + eps) * w. Operands
// [x [R,D], w [D], dY [R,D]]; results [dX [R,D], dW [D]]. `eps` is the
// forward epsilon and rides on node_attrs.
def Boom_RmsNormBwdOp : Boom_Op<"rmsnorm_bwd",
    "backward of RMSNorm: dX and the per-column dW">;

// rope_bwd: the inverse GPT-NeoX rotation of the upstream gradient. Operand
// [dY [R, H*D]] (head-major); result [dX]. The rotation is orthogonal and its
// angle depends only on the row's position, so no forward activation is read.
// Attributes on node_attrs: head_dim, theta_base, freq_scale, pos0.
def Boom_RopeBwdOp : Boom_Op<"rope_bwd",
    "backward of GPT-NeoX RoPE (inverse rotation)">;

// silu_mul_bwd: the VJP of the SwiGLU gate z = silu(g) * u. Operands
// [g, u, dZ]; results [dG, dU], all elementwise with the same shape.
def Boom_SiluMulBwdOp : Boom_Op<"silu_mul_bwd",
    "backward of the SwiGLU gated activation">;

// embedding_bwd: the VJP of the row gather y[r,:] = W[tokens[r],:]. Operands
// [tokens [R] int32, dY [R,D]]; result [dW [V,D]]. dW is written in full, so
// it needs no zero-fill; repeated tokens accumulate into one row.
def Boom_EmbeddingBwdOp : Boom_Op<"embedding_bwd",
    "backward of embedding lookup (scatter-add into the weight gradient)">;

// The forward operators of an explicitly differentiated training graph. They
// are dense-matmul and elementwise primitives with no model-bound weight, so
// every operand is a plain Boom value and all their attributes ride on
// node_attrs.
def Boom_MatmulOp : Boom_Op<"matmul",
    "dense matmul A [M,K] @ B [K,N]">;
def Boom_BmmOp : Boom_Op<"bmm",
    "batched matmul A [B,M,K] @ B [B,K,N]">;
def Boom_PermuteOp : Boom_Op<"permute",
    "reorder axes; `dims` on node_attrs">;
def Boom_IndexSelectOp : Boom_Op<"index_select",
    "gather rows along `axis`; operands [table, indices]">;
def Boom_ReduceSumOp : Boom_Op<"reduce_sum",
    "sum over `axes` (`keepdims` on node_attrs)">;
def Boom_SiluMulOp : Boom_Op<"silu_mul",
    "silu(gate) * up (SwiGLU)">;
def Boom_CrossEntropyOp : Boom_Op<"cross_entropy",
    "mean cross-entropy loss; operands [logits, targets]">;

#endif // SONICBOOM_COMPILER_BOOMOPS_TD
