# MLIR migration plan

Status: step 0 — inventory and design. **No code has been changed by this
document.** Every factual claim about current code carries a `file:line`
citation. Statements that are inference rather than observation are marked
**(inference)**.

Scope of this document: replace the transitional `planner::OpKind` graph IR with
an MLIR-based compiler IR — the `boom` dialect, **Boom IR** — keeping the runtime
unchanged in what it executes (binary blobs, hand-written kernels), with
**Blade** (the user-facing S-expression interchange format) lowering into it. The
sections below first inventory what exists (1–5), then propose the target (6),
the staged migration (7), the constraints that bound it (8), and the design
decisions (9).

---

## 1. Inventory: the transitional planner IR

### 1.1 The two representations

There are two hierarchies, and today they are layered conceptually but not in
practice:

- `sx::Document` — the frozen Blade v0.1 IR, defined in
  `core/include/sonicboom/sx/ir.h:175-179`. It is the *specified* text format
  (`design/s-expr-v0-spec.md` — the file name stays) and the input to the MLIR
  lowering path.
- `planner::Graph` — the planner-facing graph IR, defined in
  `core/include/sonicboom/planner/graph.h:73-82`. Its header comment
  (`graph.h:19-23`) states it "is a *view* over the frozen Blade v0.1 IR
  (`sx::Document`)".

That comment is only true of one of the two front ends. There are two ways a
`planner::Graph` is produced:

1. **`adapt_graph`** (`graph.h:95`, implemented at
   `core/src/planner/graph_adapter.cpp:141-253`) converts an `sx::Document` into
   a `planner::Graph`, mapping op names through
   `op_kind_from_name(node.op_name)` (`graph_adapter.cpp:202`). This is the
   ResNet-18 / Blade path.
2. **Direct construction** by the model plan emitters in
   `core/src/model/gemma4_plan.cpp` (1732 lines) — a `Builder` that appends
   `GraphNodeDesc`s straight into the graph. Gemma 4 and Qwen2 never pass
   through `sx::Document`. **(inference, well-supported):** the header comment's
   "view over `sx::Document`" is therefore a description of intent, not of the
   LLM path's data flow; `GraphNodeDesc` is a standalone struct
   (`graph.h:57-71`) with no variant of `sx::Node` inside it.

### 1.2 Graph data structures

- `TensorDesc` (`graph.h:41-53`): `id`, `shape`, `dtype`, `size_bytes`, and the
  flags `is_graph_input`, `is_graph_output`, `is_graph_constant`,
  `is_graph_external`, plus `name`.
- `GraphNodeDesc` (`graph.h:57-71`): `id`, `op` (an `OpKind`), `inputs`,
  `outputs` (tensor ids), `attributes` (a `std::vector<sx::Attribute>` — the
  Blade attribute type, reused verbatim, `sx/ir.h:101-111`), `name`, and an
  optional `backend` override.
- The `backend` override is the load-bearing part of the current design: the
  Gemma 4 emitter pins **every** node to `BackendTag::Sonic`
  (`gemma4_plan.cpp:229` and `:521`), including the shared `Add`, so routing is
  never left to the default rule. `graph.h:65-69` documents this.
- `model_fingerprint` (`graph.h:86`, implemented `graph_adapter.cpp:97-139`)
  hashes the graph's structure and baked attributes into a stable id.

### 1.3 The OpKind vocabulary

`OpKind` is `enum class OpKind : uint8_t` spanning
`core/include/sonicboom/planner/op_kind.h:34-90`, 34 values. Two disjoint sets
live in one enum:

**Legacy Blade v0.1 set** (`op_kind.h:36-44`), the seven MLIR-lowerable ops plus
two markers:

| OpKind | line | notes |
|---|---|---|
| `Conv` | :36 | MLIR lowering exists |
| `Relu` | :37 | MLIR lowering exists |
| `Add` | :38 | MLIR lowering exists |
| `MaxPool` | :39 | MLIR lowering exists |
| `ReduceMean` | :40 | MLIR lowering exists |
| `Reshape` | :41 | MLIR lowering exists |
| `Gemm` | :42 | MLIR lowering exists |
| `Softmax` | :43 | **native-torch-routed**, no MLIR lowering |
| `WholeGraph` | :44 | task-granularity marker, not an operator |

**Transformer set** (25 values), all routed to Sonic:

| OpKind | line | role |
|---|---|---|
| `QuantizedMatmul` | :48 | K-quant matvec, per-column (n small) |
| `Quantize` | :49 | fp32 → q8_1 activation quantize |
| `QuantizedMatmulQ8` | :50 | K-quant matmul over a pre-quantized q8_1 activation |
| `RmsNormQuantize` | :51 | rms_norm + q8_1 fused (T3) |
| `GeluMulQuantize` | :52 | gelu(gate)*up → q8_1 fused (T5) |
| `SiluMulQuantize` | :53 | silu(gate)*up → q8_1 fused (T5, SwiGLU) |
| `RmsNorm` | :54 | |
| `RmsNormHeads` | :55 | per-head rms norm (qk_norm) |
| `GeluFp16` | :56 | |
| `Silu` | :57 | |
| `Rope` | :58 | |
| `Attention` | :59-60 | decode attention |
| `GqaBroadcast` | :61 | KV head expansion |
| `Embedding` | :62 | |
| `MatvecF32` | :63 | dense fp32 matvec |
| `MatvecBf16` | :64 | dense bf16 matvec |
| `AddBias` | :65-68 | |
| `Mul` | :69 | |
| `Scale` | :70 | |
| `CastFp16` | :71 | |
| `Softcap` | :72 | final-logit softcapping (Gemma) |
| `Argmax` | :73 | sampling |
| `Slice` | :74 | |
| `FlashAttention` | :87 | chunked prefill attention |
| `KvAppend` | :89 | deferred KV cache append |

`op_kind_from_name` (`op_kind.h:94`) and `op_kind_name` (`op_kind.h:96`) are the
string bridge; only the legacy names round-trip through Blade text
(`graph_adapter.cpp:202`), so a Blade document cannot today name any
transformer op. **(inference):** the transformer half of `OpKind` is an
in-process C++ enum with no textual form at all.

### 1.4 Attributes

Node attributes are `sx::Attribute` values (`sx/ir.h:101-111`): a name plus an
`AttrValueKind` (`sx/ir.h:89-98`) and a value list. The transformer graph uses
only a handful of integer attributes, and they are **emitter-baked, not
validated**:

- `ds` — the mmq activation-layout selector. The emitter computes it from the
  weight quant type via the `ds_a` lambda (`gemma4_plan.cpp:564-568`:
  `Q4_K`/`Q5_K` → `1`, `Q6_K` → `-1`, else `0`), with the comment at
  `gemma4_plan.cpp:562-563` explaining that it is baked here "because only the
  emitter knows the weight type".
- `batch` — marks a batched (prefill) form; the bias-fusion pass refuses to fire
  on such nodes (`rewrites.cpp:563-564`).
- `kv_source_layer` and related KV routing attributes are set by the emitter
  (see §5.4).

There is no schema for these: nothing checks that a `ds` value is legal for the
op it is attached to, and `plan_validator.cpp` (291 lines) validates only ids,
existence and uniqueness (`plan_validator.cpp:36-38` plus the id checks), never
attributes or shapes.

### 1.5 The plan emitters (Gemma 4 / Qwen 2)

`core/src/model/gemma4_plan.cpp` is one emitter serving both models. The model
difference is a profile struct filled by a single switch:

- `profile_for` (`gemma4_plan.cpp:121-143`) with `switch (cfg.arch)`
  (`:123-142`). Gemma 4 flags at `:125-135` (`embd_scale`, `qk_norm`, `v_norm`,
  `post_attn_norm`, `post_ffw_norm`, `per_layer_input`, `layer_scale`, `kv_shared`,
  `softcap`, `tied_output`, `attn_scale = 1.0f`). Qwen 2 flags at `:138-141`
  (`proj_bias`, `silu_ffn`, `tied_output = false`,
  `attn_scale = 1/sqrt(head_dim_global)`).

Graph construction is split by regime: the decode graph is built at
`gemma4_plan.cpp:186-477`, the prefill graph from `:479` onward. The config that
drives both is `Gemma4Config` (`core/include/sonicboom/model/config.h:54-99`),
with `ModelArch{Gemma4, Qwen2}` at `config.h:39` and per-layer
`LayerConfig`/`AttentionKind` at `config.h:31,44-52`.

**Finding (inference with strong support):** model-awareness is not confined to
the emitter's control flow. The `switch (cfg.arch)` is the *only* place the two
models are distinguished structurally; every other model difference flows
through the boolean flags in `profile_for`, and every node is emitted with
`BackendTag::Sonic` already pinned. So the "compiler" today is a
model-specific plan emitter that hard-codes its own scheduling decisions, not a
target-independent IR with a lowering pipeline.

---

## 2. Planner passes

All nine passes lived in `core/src/planner/rewrites.cpp` (700 lines), declared
in `core/include/sonicboom/planner/rewrites.h` (136 lines); both files have since
been deleted (§7c status). They are
**not** run by any generic pass manager: they are called only from the model
emitter, at `gemma4_plan.cpp:1032-1038` (decode, six passes) and
`:1386-1390` (prefill, three mmq passes), plus a duplicate call site for the
logit-dump graph at `:1290-1296`. **(inference):** this contradicts the header's
claim (`rewrites.h:19-24`) that "each pass is backend-gated by its caller (a pass
is applied only when the executing backend has registered the fused/extra
kernels it introduces)" — the actual gate is an environment flag
(`quant::cuda::q8_1_enabled()` at `gemma4_plan.cpp:1032`, `mmq_enabled()` and
`mmq_min_n()` at `:1386-1387`), not a capability query on the backend.

Notation: "**kernel**" names the Sonic kernel the rewritten op requires; "**gate**"
is what must be true for the pass to be legal today.

### T1 — `quantize_once_pass` (`rewrites.cpp:79`)

- **Match:** a `QuantizedMatmul` node whose single input is produced by a
  `Quantize` node (`rewrites.cpp:97` guard).
- **Precondition:** the quantized activation fits the mmq buffer
  (`mmq_q8_fits`, `rewrites.cpp:71-75`, using `kMmqBlockBytes = 144` and
  `kMmqTailGuardBlocks = 128`, `:62-63`).
- **Rewrite:** to `QuantizedMatmulQ8` reading the q8_1 tensor directly
  (`rewrites.cpp:122-136`).
- **Kernel:** the q8_1 mmq matmul (`QuantizedMatmulQ8`).
- **General?** Generic over the matmul, but it is also the **only CSE-like
  structure in the tree**: it keeps a `std::unordered_map<uint32_t,uint32_t>
  q8_of` table (`rewrites.cpp:87-91`, lookup `:104-105`, insert `:129`) keyed on
  the quantize input's tensor id, so two matmuls sharing one activation quantize
  it once. **(inference):** this is one ad-hoc CSE, not a shared analysis.

### T3 — `fuse_rms_norm_quantize_pass` (`rewrites.cpp:143`)

- **Match:** `RmsNorm` → `Quantize`, with the `RmsNorm` having a single consumer
  (`rewrites.cpp:172-173`).
- **Rewrite:** one `RmsNormQuantize` node (`op_kind.h:51`).
- **Kernel:** the fused rms_norm+q8_1 kernel.

### T5 (GELU) — `fuse_gelu_mul_quantize_pass` (`rewrites.cpp:315`, body at
`:224`)

- **Match:** `GeluFp16`(gate) → `Mul` with the up-projection → `Quantize`.
- **Rewrite:** one `GeluMulQuantize` node (`rewrites.cpp:296-301`).
- **Kernel:** the fused gelu-mul-quantize kernel.

### T5 (SwiGLU) — `fuse_silu_mul_quantize_pass` (`rewrites.cpp:319`)

Identical body (`fuse_act_mul_quantize_pass`, `:224`), with `Silu` as the
activation. This is the Qwen 2 path.

### C.4 — `fuse_rms_norm_mmq_pass` (`rewrites.cpp:323`)

- **Match:** `RmsNorm` whose consumers are all batched `QuantizedMatmul`s.
- **Precondition (the delicate one):** every consumer must share the same `ds`
  (`rewrites.cpp:362-377`), and `ds < 0` fails the pass outright — the comment
  at `:371` records that `ds < 0` "marks a weight type with no mmq activation
  layout (Q6_K…)". A size gate at `:384` bounds the fused buffer.
- **Rewrite:** fold the norm into the matmul's activation producer
  (`RmsNorm` → mmq-consumable activation).
- **Gate:** `q8_1_enabled() && mmq_enabled() && nseg >= mmq_min_n()` at
  `gemma4_plan.cpp:1386-1387`.
- **Shape/quant-specific:** yes — it is explicitly gated on the activation
  layout of the weight it feeds.

### C.5 (GELU) — `fuse_gelu_mul_mmq_pass` (`rewrites.cpp:531`, body `:421`)

- **Match:** `GeluFp16`(gate) → `Mul` → batched `QuantizedMatmul`.
- **Rewrite:** fold into the consumer's fused activation (body
  `rewrites.cpp:491-527`).

### C.5 (SwiGLU) — `fuse_silu_mul_mmq_pass` (`rewrites.cpp:535`)

Same body, `Silu` activation (`rewrites.cpp:416` comment).

### Bias — `fuse_bias_into_gemv_pass` (`rewrites.cpp:539`)

- **Match:** an `AddBias` following a projection.
- **Rewrite:** fold the bias into the gemv.
- **Gate:** refuses the batched (prefill) form (`rewrites.cpp:563-564`) and any
  mmq-path node, which has no epilogue (`rewrites.cpp:570-571`).

### Residual — `fuse_residual_add_into_gemv_pass` (`rewrites.cpp:614`)

- **Match:** a residual `Add` adjacent to a gemv; tries both operand orders
  (`rewrites.cpp:642`).
- **Rewrite:** fold the add into the gemv epilogue.

**Finding.** Of the nine passes, two are pure graph hygiene that any correct
compiler would need (T1's dedup, the residual/bias epilogue folds), and seven
are hand-coded fusions that encode, in their match predicates, facts a real
compiler would derive from analysis: which activation layout a quantized weight
demands (`ds`), whether a node is batched, and whether the consuming kernel has
an epilogue. None of these facts are re-derivable after the rewrite; they are
guessed by the emitter and re-checked by each pass's own ad-hoc predicate.

---

## 3. Existing analyses

| Concern | How it is done today | Shared? |
|---|---|---|
| Common subexpression elimination | One ad-hoc tensor-id → tensor-id table inside `quantize_once_pass` (`rewrites.cpp:87-91`) | No |
| Dead code elimination | Each pass computes its own `std::unordered_set<uint32_t> dead` and erases tensors: `rewrites.cpp:184-216`, `:275-311`, `:492-527`, `:582-610`, `:662-694` — five copies of the same three lines | No |
| Liveness / buffer reuse | `TensorLifetime` is computed (`execution_plan.h:41-47`) but the memory planner allocates one aligned buffer per tensor and never reuses (`memory_planner.cpp:153-155` comment: "one aligned buffer per tensor … no buffer reuse is possible — the honest static result is N buffers"; loop `:162-195`) | Computed, unused |
| Runtime buffer reuse | Device buffers reused through a size-keyed pool with `last_use` release (`device_pool.cpp:66-96`, `kAlignment = 256` at `:33-38`; release driven from `runtime_executor.cpp:150-169`) | Yes, at runtime only |
| Shape handling | Static shapes only, carried on `TensorDesc::shape` (`graph.h:41-53`); **no** inference, no compatibility check — `plan_validator.cpp` validates ids/uniqueness only, and grepping `shape` across `plan_validator.cpp`, `memory_planner.cpp`, `planner.cpp` finds no shape logic (only a comment at `planner.cpp:138`) | No |
| Quant-type handling | Carried as the emitter-baked integer attribute `ds` (`gemma4_plan.cpp:564-568`), never validated; `sx::DType` has no quantized members (`sx/ir.h:39-50` lists only Float32…Bool) | No |
| KV handling | **Not in the graph at all.** The KV cache is backend-resident (`core/src/planner/sonic_cuda_backend.cpp:215-254`), keyed by layer, a ring indexed `j % n_slots` (`core/include/sonicboom/planner/kv_cache.h:38`, `KvCacheDev` `:40-47`). The graph sees only a deferred `KvAppend` marker (`op_kind.h:89`) | Invisible to the planner |

**Finding.** The planner has no analysis layer. Every property a rewrite needs
is either re-derived by that rewrite's private predicate or baked by the
emitter. Liveness is computed and then discarded. Shapes are never checked.
Quant types and the KV cache have no representation in the IR at all — the
cache is a runtime side-channel the compiler cannot reason about, which is
precisely the class of thing a compiler IR exists to make visible.

---

## 4. The Blade route

### 4.1 The op set and planner

The Blade path's operator vocabulary is exactly the seven lowerable ops plus
two markers (`op_kind.h:36-44`). The lowering dispatch table is
`lowerers()` at `core/src/sx/lowering.cpp:637-647`, registering `relu`, `add`,
`reshape`, `conv`, `max_pool`, `reduce_mean`, `gemm`; `lower_node`
(`lowering.cpp:649-666`) does string dispatch. `lower_to_mlir`
(`lowering.cpp:830-861`) verifies the module then prints it.

The design intent is explicit in `design/s-expr-to-mlir-lowering.md:41`: "No
custom SonicBoom dialect" is introduced — the lowering targets builtin + func +
arith + tensor + linalg. `lowering.cpp:26` repeats this. **(inference):** that
statement scopes the *Blade route's* lowering, which stays custom-dialect-free;
Boom IR (§6) is the compiler IR one layer below it, not a second text form.

### 4.2 How it reaches MLIR and runs

The execution half is in `core/src/sx/exec.cpp`. `run_cpu_pipeline`
(`exec.cpp:116-169`) runs convert-elementwise-to-linalg → one-shot-bufferize →
… → reconcile-unrealized-casts. `Executable::compile` (`exec.cpp:217-347`)
registers bufferization external models (`:267-277`), creates 10 dialect
contexts (`:285-294`), lowers (`:306`), runs the pipeline (`:310`), builds an
`ExecutionEngine` (`:320`), and uses a **bare-pointer ABI** (`:151-153`), with
`memrefCopy` supplied from `runner_utils.cpp` (see the pipeline listing in
`design/cpu-execution-mvp.md:43-73` and the `memrefCopy` note at `:79-88`).
`exec.cpp:298` comments on the deliberate absence of the gpu/nvvm dialects.

### 4.3 Relationship to the production IR

**(inference).** The Blade route and the production LLM path are two disjoint
worlds today:

- The Blade route lowers 7 ops to linalg and JITs them (`BackendTag::Mlir`,
  `partition.h:40-44`).
- The LLM path never enters `sx::Document`; its 25 transformer ops route to
  `BackendTag::Sonic` (`partition.h:57-84`) and are executed by hand-written
  kernels behind the backend dispatch switch.
- The only bridge is `route_op` (`partition.h:53-88`), the single routing seam:
  `Softmax` → NativeTorch (`:55-56`), all transformer kinds → Sonic (`:61-84`),
  default → Mlir (`:85-86`). **(inference):** that default-to-Mlir arm is
  unreachable for the transformer set today, because the emitter pins every
  node's backend at construction.

**(inference):** the Blade route is the *existing proof* that SonicBoom can
lower and JIT, and the LLM path is the *existing proof* that hand-written
kernels are where the performance is. The migration's central question is how to
get one IR that does both.

---

## 5. Execution: plan → launches → CUDA Graph

### 5.1 From graph to plan

`plan_execution` (`pipeline.h:33`) runs four stages (`pipeline.cpp:28-48`):
StaticPlanner → MemoryPlanner → TransferScheduler → PlanValidator.

`StaticPlanner` (`core/include/sonicboom/planner/planner.h:38-52`) walks the
graph in topological order and groups it into **regions**
(`planner.cpp:257-274`), then emits one `Compute` task per region
(`planner.cpp:282-366`). The region rule is the important structural fact:

- Only *consecutive `BackendTag::Mlir`* nodes merge
  (`planner.cpp:257-274`, condition
  `regions.back().backend == BackendTag::Mlir && b == BackendTag::Mlir`).
- **Every other backend is one region per node.**

The per-node backend is `node.backend.value_or(route_op(node.op))`
(`planner.cpp:106-107`). Because the emitter pins every node to Sonic, the LLM
plan is **one Compute task per node** — which is why `ComputeTaskDesc` carries a
`std::vector<GraphNodeId> graph_nodes` (`task.h:50-55`) whose only job, in
practice, is to be `front()`ed by the backend (`sonic_cuda_backend.cpp:259-262`).
`compute.op` is copied from the region's first node (`planner.cpp:296`), and
`compute.graph_nodes` is filled at `:298-299`; tasks are placed via `place()`.
`plan_id` is an FNV-1a hash (`planner.cpp:377-381`).

`ExecutionPlan` (`execution_plan.h:65-90`) carries `kSchemaVersion = 1` (`:66`),
`plan_id`, `model_fingerprint`, `resource_fingerprint`/`resource_version`,
`devices`, `memory_spaces`, `tensors`, `tasks`, `lifetimes`, `allocations`,
`execution_order`, `estimated_cost`, `memory_summary`. **It has no `Graph`
field.** Node attributes are therefore dropped at the plan boundary; the plan
records tasks, not the graph that produced them.

### 5.2 Where launch parameters are decided

Launch parameters are decided **twice, in two different places**, neither of
which is the plan:

- Kernel grid/block/shared-memory are computed inside each `.cu` wrapper (e.g.
  the grid-clamping sites at `core/src/nn/cuda_elementwise.cu:2305` and `:2513`
  using `sm_count()` at `:1720-1727`).
- The argument set — which buffers a given op reads and writes, and the
  attribute values it passes — is decided inside the backend dispatch `switch`
  in `core/src/planner/sonic_cuda_backend.cpp:344-780` (case lines: Scale 345,
  RmsNorm 349, RmsNormHeads 363, Rope 375, CastFp16 396, Attention 404,
  GqaBroadcast 436, QuantizedMatmul 441, Quantize 450, QuantizedMatmulQ8 460,
  RmsNormQuantize 516, GeluMulQuantize 538, SiluMulQuantize 558, MatvecF32 579,
  MatvecBf16 593, GeluFp16 634, Silu 638, Mul 642, Add 646, AddBias 650,
  Slice 658, Embedding 678, Argmax 713, FlashAttention 733, KvAppend 756, with a
  hard error default at `:775-779`).

The CPU equivalent is `core/src/planner/sonic_backend.cpp:205-498`.

### 5.3 The CUDA Graph

`GraphStep` (`core/src/nn/cuda_graph.h:33-101`) wraps the capture cycle:
begin/ThreadLocal capture (`:42-50`), end/instantiate (`:55-71`), replay (`:74`),
sync (`:77`), destroy (`:83-96`). The emitter drives it: decode runtime cache
keyed on `(model, n_ctx)` (`gemma4_plan.cpp:993-994`, `:1024-1028`), a warm-up
before capture (`:1060-1078`), and a replay loop (`:1115-1133`). Prefill capture
is gated on `n > chunk && start_pos == 0` (`gemma4_plan.cpp:1431`). `pos` and the
token id are persistent device slots, which is what lets the capture survive
depth changes; the decode-step program is `CudaStepProgram` at
`sonic_cuda_backend.cpp:106-128`, with `record_step` at `:200-213`.

### 5.4 What a plan diff would need, and what exists

`runtime_executor.cpp` (334 lines) walks `execution_order` (`:94`), resolves the
backend per task (`:108-115`), calls `backend->execute` (`:139`), and releases
per `last_use` (`:150-169`); `set_task_backend` is at `:37-39`. `device_pool.cpp`
supplies the size-keyed buffers (`:66-96`) and the copies (`:125-136`) and
memsets (`:138-147`), with downloads suppressed under capture
(`ClearScope`/`CaptureScope`, `:109-123`) and wholesale teardown at `:157-165`.

To diff two plans **exactly** — kernel sequence, launch config, buffers — four
things are required that have no carrier today:

| Needed for an exact diff | Carrier today | Gap |
|---|---|---|
| A serialized plan record | `ExecutionPlan` (`execution_plan.h:65-90`) — no serializer exists across `core/src/planner/` | No text/binary form of a plan is emitted anywhere |
| Per-node attributes | `GraphNodeDesc::attributes` (`graph.h:57-71`) | Not copied into `ExecutionPlan`, which has no `Graph` field |
| Launch config (grid, block, smem) | Computed in `.cu` wrappers (`cuda_elementwise.cu:2305,2513`) and the backend switch (`sonic_cuda_backend.cpp:344-780`) | Never recorded on any plan or task |
| Buffer / stream identity | Size-keyed pool (`device_pool.cpp:66-96`); stream is a thread-local `sb_current_stream()`; KV cache is backend-resident (`sonic_cuda_backend.cpp:215-254`) | No stable per-tensor buffer id in the plan; the KV side-channel is invisible |

**(inference).** A plan diff test (step (b) in §7) is thus not a test we can
write against today's structures — it requires the serialized execution-plan
record named in the migration scope. This is the single largest piece of missing
plumbing in the migration, and it is worth building *first*, because it is also
the gate for every subsequent step.

### 5.5 The `_step` record

`sonic_cuda_backend.cpp` already records a per-step program (`CudaStepProgram`,
`:106-128`; `record_step`, `:200-213`). **(inference):** this is the closest
thing to a plan record that exists, but it is a runtime capture of what the
backend did, not a description of the plan, and it is not serialized.

---

## 6. Proposed `boom` dialect (Boom IR)

### 6.1 Ops

The dialect should have **one op per `OpKind`** initially — this is the point of
step (a): a mechanical, round-trippable correspondence, not a redesign. Each op
carries its operands and results as tensors of the dialect's types, and its
attributes mirror the `GraphNodeDesc` attributes it replaces.

Ops fall into three tiers, by what they lower to:

1. **High-level, lowered to a call into an existing hand-written kernel.**
   Every one of the 25 transformer ops (§1.3): `boom.quantized_matmul`,
   `boom.quantize`, `boom.quantized_matmul_q8`, `boom.rms_norm_quantize`,
   `boom.gelu_mul_quantize`, `boom.silu_mul_quantize`, `boom.rms_norm`,
   `boom.rms_norm_heads`, `boom.gelu_fp16`, `boom.silu`, `boom.rope`,
   `boom.attention`, `boom.gqa_broadcast`, `boom.embedding`, `boom.matvec_f32`,
   `boom.matvec_bf16`, `boom.add_bias`, `boom.mul`, `boom.scale`,
   `boom.cast_fp16`, `boom.softcap`, `boom.argmax`, `boom.slice`,
   `boom.flash_attention`, `boom.kv_append`. The
   distinction here is that these do **not** lower to linalg — they lower to a
   runtime call. **(inference):** keeping them high level is what preserves the
   current performance; lowering a K-quant gemv to linalg would discard the
   hand-written kernel that is the whole point.
2. **High-level, lowered to linalg.** The seven legacy ops (`boom.conv`,
   `boom.relu`, `boom.add`, `boom.max_pool`, `boom.reduce_mean`, `boom.reshape`,
   `boom.gemm`), plus `boom.softmax` (which routes to native-torch,
   `partition.h:55-56`). These reuse `lowerers()` (`lowering.cpp:637-647`) and
   the `exec.cpp` pipeline unchanged.
3. **Marker.** `boom.whole_graph` (`op_kind.h:44`).

### 6.2 Types

- `!boom.tensor<shape x dtype>` — a ranked tensor type mirroring
  `TensorDesc::shape` and `sx::DType` (`sx/ir.h:39-50`).
- **Quantized types as first-class MLIR types.** This is the key change from
  `sx::DType`, which has no quantized members (`sx/ir.h:39-50`). The dialect
  should define `!boom.q3_k`, `!boom.q4_k`, `!boom.q5_k`, `!boom.q6_k`,
  `!boom.q5_0`, `!boom.q8_0` for weights (mirroring `QuantType` at
  `core/include/sonicboom/quant/quantized_tensor.h:46-53`, with block sizes from
  `QuantTypeInfo` at `:61-67`) and `!boom.q8_1` for the activation layout the mmq
  path consumes. The set is `QuantType`'s, in its order, and that table is
  authoritative: `test_boom_dialect` checks each of its entries against the
  dialect's type of the same format name, so the two cannot drift apart again. **(inference):** making the quant type a *type* rather than the
  emitter-baked `ds` integer (`gemma4_plan.cpp:564-568`) is what lets the
  `ds`-consistency precondition of `fuse_rms_norm_mmq_pass`
  (`rewrites.cpp:362-377`) become a real verifier check rather than a private
  predicate — this is the single highest-value part of the type design.
- **The format's source is not in this document.** These types can be *spelled*
  today, but no value in a converted graph carries one, because a weight is not a
  graph tensor — it is a `(layer, slot)` attribute plus a `model.weights` lookup
  (`design/weight-quant-format.md` §2). The Blade-side format field, its blast
  radius over every reader of `size_bytes`, and the removal of the `ds` predicate
  this section's **(inference)** points at are `design/weight-quant-format.md`.
- A KV type for the cache-side effects (§6.3).

### 6.3 Effects: KV cache read/write

The KV cache is invisible to the planner today (§3). In the dialect, KV must be
a first-class effect rather than a side-channel:

- `boom.kv_read` — read the prior cache for positions `< pos`; consumed by
  `boom.attention` and `boom.flash_attention`.
- `boom.kv_write` (replacing the deferred `boom.kv_append`, `op_kind.h:89`) —
  append the current rows.
- The ring semantics (`j % n_slots`, `kv_cache.h:38`) and the
  slot-count/per-layer keying (`KvCacheDev`, `kv_cache.h:40-47`;
  `KvCacheSpec`, `:54-75`) should be attributes of the cache value, not baked
  into the backend's `ensure_caches` (`sonic_cuda_backend.cpp:215-254`).
- **Effect modelling (inference):** model reads and writes as op effects so MLIR's
  effect machinery can order them; decode attention reads the whole prefix, so
  the write from step *i* must be ordered before the read at step *i+1*. Today
  this ordering is implicit in "the loop runs the ops in order".

### 6.4 Blade lowers to Boom IR

Blade and Boom IR are **two layers**, with a lowering between them (§9). Blade
is the user-facing S-expression interchange format — what Guile emits, the ONNX
importer produces, `model-export!` writes, and the file extension `.blade`
names; Boom IR is the compiler IR the passes and the planner work on. They may
share an S-expression reader, but they are separate grammars with separate
versioning: Blade is stable and versioned (`design/s-expr-v0-spec.md:76-114` for
the frozen v0.1 grammar; **Blade v0.2** for the in-progress increment list in
`design/s-expr-v0.2-draft.md`, which keeps that file name), while Boom IR is
internal and may change as the migration proceeds.

**(inference):** the lowering between them is the natural place to keep the
textual identity for three reasons: (i) the format already exists and is frozen
at v0.1; (ii) the Guile binding already emits canonical text
(`bindings/guile/scheme/sonicboom/nn.scm:223`), so the round-trip test has a
real producer and consumer on day one; (iii) it keeps the user-facing identity
in SonicBoom's own hands rather than an MLIR-generic form. The transformer ops
need names in Blade — the v0.1 grammar cannot name them today (§1.3), so
extending the op name set is part of step (a), not a separate design.

---

## 7. Migration steps and gates

Each step is one commit series, with the gates in §7.6 run before the step is
declared done. Steps are ordered so that each is independently revertible.

### (a) Dialect + bidirectional conversion

Define the `boom` dialect (Boom IR, §6) and a **bidirectional** conversion:
`OpKind` ↔ dialect. The round-trip test is the gate: for the Gemma 4 decode
graph and the Qwen 2 decode graph, `OpKind → dialect → OpKind` must reproduce a
graph that is `op`-wise and attribute-wise identical (this is checkable without
running anything).

- **Gate:** round-trip identity on both models' decode and prefill graphs. No
  numerics needed yet; this is a structural test.

### (b) Execution-plan diff test

Build the serialized execution-plan record (§5.4) and a test that diffs two
plans exactly on (kernel sequence, launch config, buffers). Run it against the
current `plan_execution` output for Gemma 4 and Qwen 2 as the baseline, so that
every later step is measured against a frozen plan.

- **Gate:** the plan produced from the current pipeline and the plan recorded
  fresh from the same code differ by nothing. This pins the baseline for (c)
  and (d).

### (c) Port the passes, one at a time

Port each of the nine passes (§2) as an MLIR rewrite pattern on the dialect,
**one commit per pass**, with the pattern's legality conditions derived from the
dialect's types and effects rather than copied from the predicates. The first
passes to port are the two hygiene rewrites (T1 dedup, residual/bias folds),
because they are the ones whose conditions are derivable; the `ds`-dependent
passes come after §6.2's quant types exist.

- **Gate per pass:** the plan diff from (b) is unchanged, *plus* all of §7.6.

**Status.** All nine passes are ported (§2's list), so every phase's pipeline now
runs entirely through Boom IR. The Graph rewrites and the `SONICBOOM_IR_PASSES`
switch that chose between the two paths have since been deleted, so Boom IR is
not merely the default implementation — it is the only one. The passes and the
entry that runs them live in the compiler library
(`core/compiler/lib/BoomPasses.cpp`, `<sonicboom/compiler/BoomPasses.h>`); what
the planner still names is the phase to run
(`planner::IrPassPhase`, `<sonicboom/planner/ir_passes.h>`). The `rewrites.cpp:NNN` references earlier in this
document are to that deleted file and are kept as the port's provenance; the
semantics they describe now live in `core/compiler/lib/BoomPasses.cpp`.

### (d) Delete the `OpKind` IR

Once every pass is a dialect pattern and the emitter builds the dialect directly,
delete `planner::OpKind`, `GraphNodeDesc`, and `adapt_graph`'s `op_kind_from_name`
bridge, and reduce the Blade route to a lowering into Boom IR (§6.4).

- **Gate:** all of §7.6, and the (b) diff is unchanged.

### (e) GPU codegen for glue ops

With the IR in place, generate the glue (norms, quantize, elementwise, residual
adds, RoPE) via the MLIR GPU path (linalg → gpu → nvvm → cubin), JIT-compiled
and cached by shape signature. This is where the migration starts paying for
itself; the hand-written kernels remain the numerical and performance reference.

- **Gate:** §7.6, plus the generated glue must match the hand-written reference
  on tolerance (per the v1 rules in `CLAUDE.md`), with any divergence reported
  (first divergent step, top-1/top-2 margin).

### 7.6 The gates (run before each step is declared done)

| Gate | Definition |
|---|---|
| Gemma golden | the 64-step `[spine-cuda]` token line is bit-identical |
| Qwen anchor | `test_qwen2_generate` passes: the greedy stream matches the llama.cpp golden, divergences allowed only inside the test's near-tie margin |
| Prefix cache | `test_prefix_cache` 12/12 |
| Perf | `tools/bench.sh --ab` with its control group: every metric's delta inside the control-derived gate (the control's median bias + 2σ of its paired deltas) |
| Plan time | the `[cuda-timing] plan X ms` observable (`gemma4_plan.cpp:1177-1181`) does not regress |

**(inference):** the plan-time cap matters specifically because (c) moves work
from emitter-time predicates into pattern matching; if the patterns cost more
than the predicates they replace, the migration has made the compiler slower
even if the runtime is unchanged.

### 7.7 Ordering rationale

(b) before (c) is deliberate: without a serialized plan and a diff test, a
ported pass that changes the plan would be invisible, and the migration would be
unfalsifiable. **(inference):** step (a) also has standalone value — the dialect
alone gives the compiler a place to *report* what it did, which is what the
plan-diff record in (b) then reads.

---

## 8. Constraints

1. **MLIR lives only in a compiler module.** The runtime executes artifacts
   (kernels, blobs), not the IR. No MLIR type may appear in
   `core/include/sonicboom/` — the Layer 2 boundary rule in `CLAUDE.md` already
   forbids this for native-torch types, and the same reasoning applies. The
   dialect headers belong in a compiler-only include path.
2. **Pinned LLVM major, external same-major install allowed.** Today the tree
   pins MLIR/LLVM 23 and *warns* (not errors) on other majors
   (`CMakeLists.txt:128-133`); the two supported pointing methods are
   `-DMLIR_INSTALL_PREFIX` and `-DMLIR_DIR`/`-DLLVM_DIR` (`CMakeLists.txt:69-109`).
   The migration keeps this: one tested major, external installs of the same
   major allowed, no in-tree LLVM patch.
3. **Own analyses on MLIR's dataflow framework.** Quant propagation, KV effects,
   shape/range, and fusion legality should be built as MLIR analyses (on the
   dataflow framework) rather than as more private predicates in rewrite
   functions (§3, §2). **(inference):** this is the constraint that prevents the
   migration from being a re-spelling of the current ad-hoc design.
4. **No `if (arch == ...)` outside the loader and the plan emitter.** Today the
   only runtime architecture conditionals are capability/launch-size probes:
   `sm_count()` and its grid clamping (`cuda_elementwise.cu:1720-1727`, `:2305`,
   `:2513`), plus compile-time `#ifdef SONICBOOM_USE_CUDA` guards
   (`device_pool.cpp:19,40,53,67,87,99,110,126,139,150,158`;
   `gemma4_plan.cpp:30,53,…`; `model/exec.cpp:36,79,429`;
   `quantized_matmul.cpp:26,543,558,574`). **(inference):** the rule is
   forward-looking — there is no such branch in the planner today — and its
   purpose is to keep target selection in the plan emitter, where it can be
   reasoned about, rather than scattered through passes.
5. **The model-specific `ds` and batching predicates must become derivable.**
   This is not stated in the migration scope but follows from §6.2: the passes
   that depend on `ds` (`rewrites.cpp:362-377`) cannot move to a target-independent
   IR while `ds` is an opaque emitter-baked integer.

---

## 9. Design decisions

- **Transformer ops stay high-level Boom IR ops.** All 25 transformer ops
  (§1.3) lower to calls into the hand-written kernels; only the glue — norms,
  quantize, elementwise, residual adds, RoPE — additionally acquires a
  linalg/GPU-codegen lowering, in step (e), with the hand-written kernel kept as
  the reference. The K-quant gemv, attention and flash-attention kernels are
  where the performance lives (`design/benchmark.md`; `CLAUDE.md` names the
  hand-written kernels as the reference), so lowering them to linalg would
  discard it.
- **The execution-plan record and its diff test come before any pass is
  ported.** They are built in step (b), on the four-vector in §5.4 — kernel
  sequence, launch config, buffer identity, stream/effect ordering. Without them
  step (c) is unfalsifiable.
- **Launch parameters stay in the backend as a transitional choice.** Through
  step (d) they remain kernel-specific tuning inside the backend and the `.cu`
  wrappers, tracked by `design/tuning-parameters.md` and recorded *on the plan*
  as an observable; **from step (e) they become a tuning-config attribute on
  `boom` ops**, chosen by the autotuner / runtime JIT and recorded in the plan.
  Moving them into the IR before the autotuner exists would only re-derive
  `design/tuning-parameters.md` for no runtime benefit.
- **The KV cache is modelled as dialect effects; its physical storage stays
  backend-owned.** `boom.kv_read` / `boom.kv_write` effects (§6.3) give the
  compiler the dependency to analyse and order, while the cache itself remains a
  backend-resident ring keyed by layer (`kv_cache.h:38-47`) — the effect model
  describes the dependency, it does not have to own the storage. This is what
  makes the `kv_shared` / `kv_source_layer` routing (set by the emitter today) a
  checkable property instead of an emitter convention.
- **Blade and Boom IR are two layers, with a lowering between them.** Blade is
  the user-facing S-expression interchange format — what Guile emits, the ONNX
  importer produces, `model-export!` writes; extension `.blade` — and Boom IR is
  the compiler IR. They may share an S-expression reader, but they are separate
  grammars with separate versioning: Blade is stable and versioned, Boom IR is
  internal and may change.
- **`BackendTag` becomes an attribute of an op's lowering, not a graph-node
  field.** `route_op` (`partition.h:53-88`) survives as the *default* routing
  rule, and the emitter stops pinning every node (`gemma4_plan.cpp:229,521`).
  Once the dialect's ops lower to a known target, pinning every node is
  redundant.
- **A predicate a pattern cannot reproduce exactly is re-expressed as an
  analysis, never copied into the pattern.** A predicate that cannot be
  re-expressed is evidence that the property it tests is not derivable from the
  IR, which is itself the finding; the pass stays unported (constraint §8.3).
- **`WholeGraph` survives step (a) and is deleted in step (d).** It is task
  granularity, and step (a)'s round-trip gate needs it present for fidelity;
  after (b) the plan record expresses task grouping directly.
