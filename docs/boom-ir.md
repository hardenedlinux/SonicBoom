# Boom IR

Boom IR is SonicBoom's compiler IR. It is an MLIR dialect in the `sonicboom::compiler::boom`
namespace; the TD definitions are `core/compiler/include/sonicboom/compiler/`
(`BoomDialect.td`, `BoomTypes.td`, `BoomOps.td`), the verifier is
`core/compiler/lib/BoomVerify.cpp`, and the pass pipeline is
`core/compiler/lib/BoomPasses.cpp`.

Every execution plan is emitted as Boom IR, optimized by the passes for its
phase, and lowered back to the planner's runtime graph. The planner's boxed
executor still runs `planner::Graph`; Boom IR is the compiler's representation,
not a second runtime.

```
plan site (model emitter)
      ↓ graphToBoom()                 conversion, one boom.graph per module
   Boom IR
      ↓ runBoomPasses(phase)          fusion / fold / scheduling passes
   Boom IR
      ↓ boomToGraph()                 conversion back
planner::Graph → executor → kernels
```

## Module and graph structure

A module holds one `boom.graph`, which is the unit of verification and the
anchor the pass pipeline runs on (the op is `IsolatedFromAbove`).

```mlir
module {
  boom.graph @decode(tensors = [#boom.tensor_desc<...>, ...],
                     inputs = array<i64: 0, 1>,
                     outputs = array<i64: 900>,
                     dynamic_extent = 0) {
  ^bb0(%arg0: !boom.tensor<[1], "int64">, %arg1: !boom.tensor<[1], "int64">):
    %0 = boom.embedding(%arg0) {node_name = "embedding", node_id = 0 : i64,
                                result_ids = array<i64: 2>, node_attrs = [...]}
         : (!boom.tensor<[1], "int64">) -> !boom.tensor<[1, 2560], "float32">
    ...
  }
}
```

- **Tensor table.** `tensors` carries every `planner::TensorDesc` verbatim
  (`#boom.tensor_desc<id, shape, dtype, byte_size, dynamic_base_bytes, is_input,
  is_output, is_constant, is_external, name>`). It is authoritative for the
  fields a type cannot carry: byte size (whose dynamic entries are sized at
  `dynamic_extent`) and the four flags. A value's type must agree with its entry.
- **Body.** One block, an ordered node list, no terminator. The block arguments
  are the graph's *free variables* — every tensor the graph references but no
  node produces (feedable inputs, model-bound weights and constants, externals)
  — in ascending tensor-id order. Weights are block arguments too, so a node can
  read its K-quant weight as an operand.
- **Results.** A node's results carry their tensor ids in `result_ids`, in
  result order. Node results plus block arguments give a total SSA → tensor-id
  mapping, and `inputs`/`outputs` name the feedable and returned ids. The graph's
  outputs are ids, not a terminator.

The conversion round-trips exactly: the same tensors, node ids, and attributes
in the same order. Attribute *order* is part of the model identity
(`model_fingerprint` hashes it), so `node_attrs` is an ordered array, never a
name-keyed dictionary.

## Types

| type | meaning |
|---|---|
| `!boom.tensor<[d0, d1, ...], "dtype">` | ranked static tensor; dtype is one of the ten frozen `sx::DType` names |
| `!boom.q8_1<[...]>` | activation in the q8_1 activation layout the MMQ path consumes |
| `!boom.q3_k<...>` `!boom.q4_k<...>` `!boom.q5_k<...>` `!boom.q6_k<...>` | K-quant weights (256-element super-blocks) |
| `!boom.q5_0<...>` `!boom.q8_0<...>` | legacy 32-element quant weights |
| `!boom.kv<layer>` | layer's KV cache handle (not a value) |

Quantized types exist because a quantized format is a property of the value's
layout, while `sx::DType` is a frozen ten-name set. Shapes are the *logical*
element shape; block geometry follows from the format, and the block-contiguous
(fastest-moving) dimension must be a multiple of the block size — the verifier
enforces it. `!boom.kv<L>` names the backend-owned per-layer cache; it has no
shape, dtype or tensor-table entry.

Attribute definitions: `#boom.attr<name, value>` is one `sx::Attribute`, and
`#boom.tensor_desc<...>` is one tensor-table entry.

## Operations

### Containers and generation control

| op | purpose |
|---|---|
| `boom.graph` | a whole planner graph and its tensor table |
| `boom.program` | one greedy generation: `boom.prefill`, then an `scf.while` body of step/sample/emit/stop |
| `boom.prefill` | call the named prefill graph over the prompt |
| `boom.step` | one decode step (carries its own tensor table); operands are the loop-carried values and cache handles |
| `boom.sample` | choose the next token (greedy argmax when no sampling attributes are set) |
| `boom.stop_check` | EOS-set + max-token stop test |
| `boom.emit` | append token(s) to the host-visible ring |
| `boom.pos_inc` | advance the staged position |
| `boom.draft` | speculation: propose up to `max_drafts` tokens |
| `boom.verify` | speculation: decode graph dealt over rows, one argmax per row |
| `boom.accept` | speculation: longest matching draft prefix + verifier token |

Speculation replaces the loop body's step/sample pair with
`draft -> verify -> accept`; the plain step stays for the `k == 0` case.

### Node operations

There is one node op per `planner::OpKind`, with the mnemonic equal to
`op_kind_name(OpKind)` — the correspondence is mechanical in both directions,
not a table that can drift. All node ops share one attribute record:

| attribute | meaning |
|---|---|
| `node_name`, `node_id`, `result_ids`, `node_attrs` | graph-node identity; enough to rebuild the `GraphNodeDesc` exactly |
| `backend` | optional backend override (absent = the planner's route) |
| `slot`, `layer`, `kv_source_layer`, `ds`, `batch`, `dim`, `offset`, `rows`, `length`, `stride`, `scale` | the typed view of the attribute vocabulary the emitters and passes use |

The verifier requires `node_attrs` and the typed view to agree, and enforces
which of them each op may carry. Operands and results are variadic and typed;
arity is checked with the shape/dtype/quant checks rather than fixed per op.

- **Frozen S-Expr v0.1 CNN operators** (tier 2, lowered to linalg):
  `boom.conv`, `boom.relu`, `boom.add`, `boom.max_pool`, `boom.reduce_mean`,
  `boom.reshape`, `boom.gemm`; `boom.softmax` routes to native-torch;
  `boom.whole_graph` is a task-granularity marker.
- **Transformer operators:** `boom.quantized_matmul`, `boom.quantize`,
  `boom.quantized_matmul_q8`, `boom.matmul_q8_tc`, `boom.rms_norm_quantize`,
  `boom.gelu_mul_quantize`, `boom.silu_mul_quantize`, `boom.rms_norm`,
  `boom.rms_norm_heads`, `boom.gelu_fp16`, `boom.silu`, `boom.rope`,
  `boom.attention`, `boom.gqa_broadcast`, `boom.embedding`, `boom.matvec_f32`,
  `boom.matvec_bf16`, `boom.add_bias`, `boom.mul`, `boom.scale`,
  `boom.cast_fp16`, `boom.softcap`, `boom.argmax`, `boom.slice`,
  `boom.flash_attention`, `boom.kv_append`.
- **Scheduling:** `boom.async_weight_prefetch`, `boom.sync_prefetch` are
  layer-boundary markers for the runtime's prefetch stream, and
  `boom.fused_region` is a compiler-generated glue region: it takes over the op
  it replaces (same id, attributes, operands, results, backend) and its
  `fused_region` node attribute selects the generated kernel at dispatch.
- **Training:** `boom.adamw`, `boom.clip_grad_norm`, `boom.matmul_bwd`,
  `boom.softmax_bwd`, `boom.cross_entropy_bwd`, `boom.rmsnorm_bwd`,
  `boom.rope_bwd`, `boom.silu_mul_bwd`, `boom.embedding_bwd`, plus the forward
  primitives `boom.matmul`, `boom.bmm`, `boom.permute`, `boom.index_select`,
  `boom.reduce_sum`, `boom.silu_mul`, `boom.cross_entropy`.

### Effects

The KV cache is a memory-effect resource, not a side channel: `boom.attention`
reads the prefix and appends the current row, `boom.flash_attention` reads only
(the batched append is deferred to `boom.kv_append`, which writes only), so
MLIR can order them. The three cache-touching ops carry their
`!boom.kv<layer>` handle as their last operand.

## Pass pipeline

`runBoomPasses(module, phase, skip_kv_append_fusion)` runs one phase's passes
over the module's graph, verifying after every pass. A pass order is
significant: later passes consume the nodes earlier ones produce. A pass
failure fails the plan — the IR is never lowered as if the phase had run.

| phase | passes, in order |
|---|---|
| **Decode** | `QuantizeOncePass`, `RmsNormQuantizePass`, `GeluMulQuantizePass`, `SiluMulQuantizePass`, `FuseGeluMulPass`, `BiasIntoGemvPass`, `ResidualAddIntoGemvPass`, `FuseResidualEpiloguePass`, `FuseSiblingGemvPass`, `FuseGlueRegionsPass`, `FuseHeadsNormRopeCastPass`, `FuseKvAppendIntoPrepPass`, `BufferReusePass`, `AsyncPrefetchPass` |
| **Verify** | the decode list with `FuseKvAppendIntoPrepPass` skipped: a verify step appends one row per query position, which a single-row prep region cannot express |
| **Prefill** | `RmsNormMmqPass`, `FuseAddRmsNormMmqPass`, `FuseRmsNormAddPass`, `FoldScaleIntoGluePass`, `GeluMulMmqPass`, `SiluMulMmqPass`, `FuseSliceScalePass`, `FuseGeluMulPass`, `FuseSiluMulPass` (standard numerics profile only), `ScaleIntoDensePass`, `FuseHeadsNormRopeCastPass`, `PrefillDispatchPass`, `DenseLayoutPropagationPass`, `BufferReusePass`, `AsyncPrefetchPass` |
| **Training** | `BufferReusePass` only (semantic-neutral for a traced forward+backward graph) |

What each pass does:

| pass | effect |
|---|---|
| `QuantizeOncePass` | insert one `quantize` before the first `quantized_matmul` consumer of an activation; later matmuls share it |
| `RmsNormQuantizePass` | `rms_norm` with one reader (`quantize`) becomes `rms_norm_quantize` |
| `GeluMulQuantizePass` / `SiluMulQuantizePass` | gated `gelu_fp16`/`silu` × up followed by `quantize` becomes one fused op |
| `FuseGeluMulPass` | the dense-projection form of the same gated multiply |
| `FuseSiluMulPass` | the SwiGLU counterpart, prefill only and only under the standard numerics profile (a down projection with no mmq activation layout, e.g. Q6_K, leaves the chain for it) |
| `BiasIntoGemvPass` | `add_bias` over a single-reader gemv becomes the projection's `bias_slot` epilogue |
| `ResidualAddIntoGemvPass` | `add(proj, residual)` becomes a two-input projection epilogue |
| `FuseResidualEpiloguePass` | the same treatment where the residual add is produced by a norm (sandwich norm / PLE post-norm) |
| `FuseSiblingGemvPass` | group projections that share an activation |
| `FuseGlueRegionsPass` | fold a maximal elementwise chain into one `boom.fused_region` (one generated kernel) |
| `FuseHeadsNormRopeCastPass` | attention prep chains (norm → rope [→ cast]) in decode and prefill |
| `FuseKvAppendIntoPrepPass` | fold the K/V append into the attention prep region |
| `RmsNormMmqPass` | `rms_norm` + batched q8_1 quantize for the MMQ prefill path |
| `FuseAddRmsNormMmqPass` | add → norm → quantize as one batched region |
| `FuseRmsNormAddPass` | norm + add chains the MMQ fold did not take |
| `FoldScaleIntoGluePass` / `ScaleIntoDensePass` | a scalar `scale` after a single-reader chain becomes an `epilogue_scale` attribute (node position unchanged) |
| `GeluMulMmqPass` / `SiluMulMmqPass` | gated activation + batched quantize for MMQ |
| `FuseSliceScalePass` | strided slice + scale as one elementwise region |
| `PrefillDispatchPass` | rewrite a batched q8 matmul to the generated int8 tensor-core kernel, per `SONICBOOM_MMQ_TC` (`off` default, `auto`, `on`) and the `SONICBOOM_MMQ_TC_MAX_MB` copy budget |
| `DenseLayoutPropagationPass` | mark the projections with their dense layout (attribute only) |
| `BufferReusePass` | mark single-use elementwise results the executor may write in place |
| `AsyncPrefetchPass` | insert prefetch markers when model weights exceed `SONICBOOM_VRAM_THRESHOLD_MB` (`SONICBOOM_ASYNC_PREFETCH=on`) |

### Ablation

`SONICBOOM_IR_PASS_DISABLE=PassName[,PassName...]` skips those passes for the
run. Names are the pass class names above. A phase that does not contain a
named pass ignores it; an unrecognized name is reported on stderr with the
valid names and disables nothing. A disabled pass can also remove a later
pass's fusion (the passes run in sequence), so a plan delta measures that pass
and the folds it feeds together.

## Adding a pass

1. Write the class in `core/compiler/lib/BoomPasses.cpp` as an MLIR pass over
   `boom::GraphOp`. Read the graph through the pass-side view (`graphView`),
   which owns the tensor-id ↔ SSA-value relation and the tensor table; edit both
   only through it.
2. Create replacement nodes with `buildNode`, which writes the identity
   attributes, the authoritative attribute record, and the typed view the
   verifier checks. Do not construct an op by hand.
3. Add the pass to the phase's list in `addPasses`, in the position its
   preconditions require, and add a `{"YourPassPass", &DisabledPasses::flag}`
   row to `kPassNames` so ablation can name it.
4. Add a `TEST` to the conversion/verifier suite, and a plan golden under
   `tests/data/plans/` if the pass changes the node set.
5. Ensure `boomToGraph` still round-trips: a pass may only produce
   `GraphNodeDesc`s the planner can rebuild (same tensor table invariants, same
   attribute kinds).

## Verification rules

The verifier (`BoomVerify.cpp`) rejects, with the offending node named:

- a value whose type disagrees with its `#boom.tensor_desc` entry (shape and
  dtype; a `!boom.q8_1` value is stored as a float32 buffer, so its table entry
  is float32 with the matching shape);
- a `node_attrs` record that disagrees with the typed attribute view, or an
  attribute the op may not carry;
- a `result_ids` list that does not match the op's arity or the tensor table;
- a KV handle anywhere but the last operand of the three cache-touching ops, or
  whose layer disagrees with the op's `layer`/`kv_source_layer`;
- an unknown op kind, an unsupported dtype, or a quantized shape whose
  block-contiguous dimension is not a multiple of the block size.
