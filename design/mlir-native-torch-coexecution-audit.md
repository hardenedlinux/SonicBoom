# MLIR + Native Torch 协同执行现状审计

> Audit only — this document records the current state of the MLIR→JIT→C ABI
> chain and the native-torch dispatcher chain, the gap between them, and the
> minimal co-execution closed loop. No code is modified here. Implementation
> boundaries are decided *after* this audit (see §9).

Date: 2026-10-05. Branch: `native-torch/refactor`. Baseline `214bde6`, CUDA
checkpoint `f667257`.

> **Status update (2026-10-06):** the gap this audit describes has since been
> closed. The co-execution loop landed as `6a8030f`, branch/merge + cost
> benchmark as `f973e20`, and region-partitioned mixed co-execution as `0a12303`.
> See §11 "Implemented since the audit".

---

## 0. Executive summary

SonicBoom currently has **two independent, non-communicating execution paths**:

- **Path A (committed):** S-Expr v0.1 → sx IR → MLIR lowering (7 ops) → MLIR pass
  pipeline → LLVM JIT (`sx::Executable`) → C ABI. Whole-graph, single
  `float*`-in / `float*`-out, CPU only. ResNet-18 verified.
- **Path B (partially committed):** Layer 2 `nt::` types → Layer 1 adapter →
  native-torch dispatcher (`redispatchBoxed`, tensor-driven). Direct
  `OperatorHandle::call(args)`, no graph. CPU kernels migrated; CUDA committed
  in `f667257`. `test_native_ops.cpp` proves 6 aten op categories execute.

**There is no bridge between them.** Specifically, none of the following exist:

1. An op-routing decision: the planner classifies every node into one of the
   7 `OpKind`s and emits a single whole-graph JIT task — there is no predicate
   "this node goes to native-torch, that node goes to MLIR".
2. A tensor conversion `sx::Bytes ↔ nt::Tensor`.
3. An op-name + attribute mapping `sx "conv"/"add"/… → aten "aten::conv2d"/
   "aten::add.Tensor"/…`.

The **integration seam already exists** — `planner::Backend` (`core/include/
sonicboom/planner/backend.h`) — and only `CpuBackend` implements it. The
minimal co-execution loop is a **`NativeTorchBackend`** peer to `CpuBackend`,
driven by a planner that partitions the graph into MLIR-lowerable subgraphs vs.
native-torch nodes, with `sx::Bytes` handoff at the boundaries.

The honest first milestone (no MLIR lifting required) is a **2-node CPU graph
`[softmax → relu]`**: `softmax` (not in the 7-op MLIR set) dispatches to
`aten::_softmax` via the new backend; `relu` compiles to a single-node MLIR JIT
sub-graph. This proves one graph that *compiles + executes + correctly calls
native-torch*, with a real `Bytes ↔ Tensor` boundary.

---

## 1. Path A inventory — MLIR → JIT → C ABI (committed)

| Stage | Location | State |
|---|---|---|
| S-Expr v0.1 lexer/parser/validator | `core/src/sx/` (`lexer.cpp`, `parser.cpp`, `sexpr.cpp`, `runner_utils.cpp`) | committed |
| sx IR (pure C++, no MLIR/ATen) | `core/include/sonicboom/sx/ir.h` — `Document/Graph/Node/TensorType/Shape/Attribute/Parameter` | committed |
| ONNX → S-Expr importer (offline Python) | `tools/onnx2sx/` + `design/onnx-to-s-expr-importer.md` | committed |
| S-Expr → MLIR lowering | `core/src/sx/lowering.cpp` | committed |
| JIT facade (MLIR-free header) | `core/include/sonicboom/sx/exec.h` — `sx::Executable::compile/run` | committed |
| C ABI (Layer 3) | `capi/include/sonicboom/capi.h`, `capi/src/capi.cpp` — `sb_parse/sb_compile/sb_bind_input/sb_execute/sb_retrieve_output` | committed |
| Static planner + runtime executor | `core/src/planner/` + `core/include/sonicboom/planner/` | committed |
| Build wiring | `core/CMakeLists.txt` (`SONICBOOM_ENABLE_MLIR` block: MLIR dialects + ExecutionEngine + bufferize→LLVM passes) | committed |

### 1.1 Lowering operator set (frozen v0.1)

`core/src/sx/lowering.cpp` `lowerers()` and `core/src/planner/op_kind.cpp`
agree on exactly **7 operators**, all lowered to `linalg`/`arith`/`tensor`:

| sx name | MLIR lowering | notes |
|---|---|---|
| `relu` | `arith.maximumf(x, 0)` | |
| `add` | `arith.addf` | |
| `reshape` | `tensor.reshape` | static output shape |
| `conv` | `linalg.conv_2d_nchw_fchw` + `tensor.pad` + bias add | group==1, dilations==1 only |
| `max_pool` | `linalg.pooling_nchw_max` | ceil_mode==0, dilations==1 |
| `reduce_mean` | `linalg.generic` sum + divide | axes from `:values` param |
| `gemm` | `linalg.matmul` + optional transpose/bias | rank-2 A/B only |

Any other op name is rejected by the lowering (`"unsupported operator"`) and by
the planner's `op_kind_from_name` (`UnsupportedOperator`). This is the single
source of truth for "what MLIR can lower".

### 1.2 Execution granularity constraint

`sx::Executable` compiles a whole `sx::Document` into one `func.func` and exposes
**exactly one `float32` input and one `float32` output** (`core/include/
sonicboom/sx/exec.h`, `planner/backend.h`, `capi.h`). The planner emits a single
`TaskKind::Compute` task covering the entire graph; `CpuBackend::execute`
asserts `inputs.size() == 1`. `RuntimeExecutor::execute` (`runtime_executor.cpp`)
walks `execution_order` but only `Compute` is implemented — `Transfer`,
`Allocate`, `Release`, `Synchronize` are stubbed as errors.

This whole-graph 1-in-1-out constraint is the **single most important blocker**
for fine-grained co-execution (see §7.2).

### 1.3 Build + test coverage

- Build: MLIR/LLVM pinned `llvmorg-23.1.2` (`third_party/llvm-project`,
  unmodified), built once into a gitignored prefix. `libsonicboom.so` links the
  MLIR dialects/ExecutionEngine PRIVATE.
- Tests: `tests/core/` holds 28 test programs (incl. `test_mlir`,
  `test_s_expr_lowering`, `test_s_expr_exec`, `test_s_expr_resnet18*`,
  `test_onnx_import_resnet18_exec`, the planner suite, `test_capi`).
- Verification: ResNet-18 max abs err `3.34e-6`, argmax `107 = 107`
  (`design/onnx-to-s-expr-importer.md`, `cpu-execution-mvp.md`).

**Conclusion:** Path A is committed, builds, and is verified end-to-end on CPU.
It is *not* to be redesigned — the co-execution work builds *beside* it.

---

## 2. Path B inventory — native-torch dispatcher (partial)

| Stage | Location | State |
|---|---|---|
| Layer 2 `nt::` types | `core/include/sonicboom/` (`tensor.h`, `operator_handle.h`, `value.h`, `device.h`, `backend.h`) | committed (CUDA fields in `f667257`) |
| Layer 1 adapter (only ATen/c10 users) | `core/layer1/adapter/*.cpp` | committed |
| Tensor-driven dispatch | `core/layer1/adapter/operator_handle.cpp` (`getDispatchKeySetBoxed` → `redispatchBoxed`) | committed |
| CPU kernel migration | `third_party/native-torch/aten/src/ATen/**`, `c10/**` | **uncommitted** (Stage E continuation) |
| CUDA wiring + 75 stubs | `f667257` (`ExcludedKernelsStubs.cpp`, `CUDAGuardImpl`, `--whole-archive`) | committed |
| Dispatch test (G-3) | `tests/core/test_native_dispatch.cpp` | **uncommitted** |
| Op-coverage test (G-4) | `tests/core/test_native_ops.cpp` | **uncommitted** |
| GPU closed loop | `tests/core/test_cuda.cpp` | committed |

The aten side of the op-name mapping is **already proven executable** by
`test_native_ops.cpp` (G-4): `aten::mm`, `aten::_softmax`, `aten::max`,
`aten::index_select`, `aten::conv2d`, `aten::to.dtype` all dispatch through the
real dispatcher, plus a clean `cudnn_convolution` errpath. So "native-torch can
execute these aten ops" is established; what is missing is *routing sx nodes to
them* and *feeding them sx tensors*.

---

## 3. Uncommitted working-tree audit

The working tree carries **three interleaved, uncommitted streams** (none is
part of `f667257`):

1. **Stage E CPU kernel migration** — the bulk of `third_party/native-torch/
   aten/src/ATen/**` and `c10/**` are untracked (`??`) or modified (`M`). This
   is the ongoing "migrate the minimum native implementation" work; it does not
   belong to the co-execution milestone and must be committed/landed separately.
2. **Two native-torch tests** — `tests/core/test_native_dispatch.cpp` (G-3) and
   `tests/core/test_native_ops.cpp` (G-4) are untracked; the
   `tests/core/CMakeLists.txt` `foreach` line that wires them into the build is
   present in the working tree but **uncommitted** (deliberately excluded from
   the CUDA checkpoint).
3. **Design docs** — `design/soniccross-per-op-header-requirement.md` and
   `design/stage-e-completeness-audit.md` are untracked; `design/native-torch-
   migration.md` and `.gitmodules` are modified.

No `core/`, `capi/`, or `bindings/` file is uncommitted — the SonicBoom-owned
Layer 2/3 code is fully committed. The co-execution milestone must **not** be
swept into any of these three streams; it lands on its own.

---

## 4. The gap — five missing pieces

Between Path A and Path B, five concrete pieces are missing. These are the
entire substance of the co-execution work.

### G-1 — op routing (planner capability predicate)

Today the planner classifies every node via `op_kind_from_name` (7 names) and
emits one whole-graph JIT task. There is no predicate that answers, for a given
sx node, "does native-torch have a registered operator for this?" (the aten
dispatcher's `OperatorEntry` lookup) vs. "can MLIR lower this?" (the `lowerers()`
7-op set). The routing table must decide, per node, which backend executes it.

### G-2 — `sx::Bytes ↔ nt::Tensor` conversion

`sx::Bytes` (host `std::vector<std::byte>`) has no conversion to/from
`nt::Tensor` (owns `at::Tensor`). The bridge must materialize a host buffer +
shape + dtype as an `nt::Tensor` and copy the result back. v0 copy (not
zero-copy) — see §6 lifetime.

### G-3 — op-name mapping `sx → aten`

| sx | aten (native-torch) | status (G-4 test) |
|---|---|---|
| `conv` | `aten::conv2d` (composite) | proven |
| `relu` | `aten::relu` | proven via dispatcher |
| `add` | `aten::add.Tensor` | proven (CPU + CUDA) |
| `max_pool` | `aten::max_pool2d` | not yet wired |
| `reduce_mean` | `aten::mean.dim` | not yet wired |
| `reshape` | `aten::reshape` | not yet wired |
| `gemm` | `aten::mm` / `aten::addmm` | `aten::mm` proven |
| *(softmax, for the loop)* | `aten::_softmax` | proven |

This table is the seed of the routing decision — but it does not yet exist as a
runtime mapping, and the **attribute conversion** (sx `pads/strides/dilations/
group/kernel_shape/axes/transA/transB/alpha/beta` → aten schema arguments) is
only partially present (the `SymInt[]` decay in `core/layer1/adapter/schema.cpp`
handles `conv2d`'s stride/padding/dilation only).

### G-4 — a `NativeTorchBackend`

No `planner::Backend` implementation exists besides `CpuBackend`. A
`NativeTorchBackend` must wrap the Layer 1 dispatch (or call `nt::OperatorHandle`
directly), converting `sx::Bytes` inputs → `nt::Tensor`, building an
`nt::ArgumentList` from the node's attributes, dispatching, and converting the
result back.

### G-5 — runtime executor multi-backend dispatch

`RuntimeExecutor` holds a single `Backend&` and passes every `Compute` task to
it. With two backends, the executor needs a backend registry (keyed by
`DeviceKind` or an explicit per-task backend tag) and must implement the
`Bytes` handoff between consecutive tasks of different backends.

---

## 5. Integration point

The seam is `planner::Backend` (`core/include/sonicboom/planner/backend.h`):

```text
sx::Document
   │  adapt_graph() → planner::Graph (per-node OpKind already present)
   ▼
planner (partition: MLIR-lowerable subgraphs  vs  native-torch nodes)
   │
   ├── Compute task (MLIR)  ──► CpuBackend      ──► sx::Executable (unchanged)
   └── Compute task (nt)    ──► NativeTorchBackend ──► nt::OperatorHandle
   ▲
RuntimeExecutor walks execution_order, hands sx::Bytes between backends
```

Two partitioning strategies are possible; **A is the recommended first step**
(it keeps `sx::Executable` and its 1-in-1-out contract untouched):

- **A — planner-level partitioning (multi-backend).** The planner emits one
  `Compute` task per native-torch node and one per contiguous MLIR-lowerable
  region. The minimal loop uses regions that are themselves 1-in-1-out, so no
  MLIR lifting is needed. Tensor conversion lives at the backend/executor
  boundary.
- **B — lowering-level fusion (MLIR calls native-torch).** The lowering emits a
  `func.call` to a native-torch shim for unsupported ops. Far more invasive: the
  MLIR JIT needs an external ABI, native-torch symbol resolution, and it breaks
  the memref↔at::Tensor boundary inside a compiled module. Rejected for v0.

---

## 6. Boundaries

| Concern | Decision for the loop |
|---|---|
| **Tensor** | `sx::Bytes ↔ nt::Tensor` via **copy**, not zero-copy. The bridge allocates an `nt::Tensor` (`empty` or `from_blob`), copies host bytes in, dispatches, copies the result out to a fresh `sx::Bytes`. Zero-copy (`from_blob` over the `Bytes` buffer) is deferred — it entangles the `Bytes` and `Tensor` lifetimes. |
| **Device** | CPU first. `sx::Bytes` is host memory; the minimal loop runs both nodes on CPU. CUDA co-execution (an op on `Device::cuda(0)` with `Bytes` host handoff) reuses the committed `to_device` machinery but requires the bridge to route through host↔device copies — a follow-up, not the first loop. |
| **Error** | native-torch throws c10 exceptions (`TORCH_CHECK_NOT_IMPLEMENTED`, etc.); sx uses `std::expected`/error strings. `NativeTorchBackend::execute` must catch `std::exception` and map to `RuntimeError` (the errpath is already proven catchable in `test_cuda.cpp` §4). No c10 exception may cross the `Backend::execute` boundary. |
| **Lifetime** | `nt::Tensor` owns `TensorImpl` (refcounted); `sx::Bytes` owns a `std::vector<std::byte>`; `sx::Executable` owns the JIT engine and baked weights. Each backend produces a self-contained `sx::Bytes` result; the executor moves it to the next task. No cross-backend pointer sharing. |
| **Layer discipline** | `NativeTorchBackend` lives where it can use `nt::` types (Layer 2) — it must **not** reach into `core/layer1/` or `third_party/native-torch/` directly. It calls the Layer 2 `nt::OperatorHandle`/`nt::Tensor`/`nt::ArgumentList` API, keeping the Layer 1 boundary intact. |

---

## 7. Minimal co-execution closed loop (first milestone)

### 7.1 The loop

A 2-node, 1-input/1-output CPU graph:

```text
x ──► softmax (aten::_softmax, dim=1) ──► relu (MLIR JIT) ──► y
        └─ NativeTorchBackend ─┘          └─ CpuBackend ─┘
                  └──── sx::Bytes handoff ────┘
```

- `softmax` is **not** in the 7-op MLIR set → routed to `NativeTorchBackend`
  (`aten::_softmax`, `dim` attribute from the sx node).
- `relu` **is** in the 7-op set → compiled as a 1-node sub-Document by
  `CpuBackend` (`sx::Executable`), which trivially satisfies 1-in-1-out.
- The two are connected by an `sx::Bytes` handoff through `RuntimeExecutor`.

This proves the three claims the user requires: **one graph compiles + executes
+ correctly calls native-torch**, with a genuine tensor boundary — not a lone
MLIR JIT run nor a lone native-torch dispatch.

### 7.2 What must be built (new code)

1. `nt::Tensor ↔ sx::Bytes` bridge (copy-based) — a small helper in Layer 2 or a
   new planner header; host `float32` only for the loop.
2. `NativeTorchBackend` (`core/src/planner/native_torch_backend.cpp`) —
   implements `planner::Backend`; calls `nt::find_operator` + `nt::OperatorHandle::call`.
3. An op-name + attribute mapping table for the loop's ops (`softmax` dim,
   later the 7 op set). Seed: sx→aten table in §4 G-3.
4. Planner partitioning — a capability predicate (`can_mlir(node)` = the 7-op
   set; else `can_native_torch(node)` = dispatcher lookup) and a partition step
   emitting per-node/per-region `Compute` tasks.
5. `RuntimeExecutor` — a backend registry and `Bytes` handoff between tasks.

### 7.3 Explicitly *not* in the first loop

- No MLIR lifting (the `sx::Executable` 1-in-1-out contract is untouched; only
  1-in-1-out regions are partitioned).
- No CUDA in the loop (CPU only); no device transfer scheduling.
- No fusion, no cost model, no dynamic programming (see §8).
- No op beyond `softmax` + `relu` — the attribute-conversion table grows later.

---

## 8. Scheduling expansion (deferred, step 4)

After the loop is correct end-to-end, in order:

1. **Semantic correctness / end-to-end** — partition + multi-backend dispatch +
   correct `Bytes` handoff for arbitrary mixtures of the 7 MLIR ops and their
   aten twins (the full sx→aten table in §4 G-3).
2. **Fusion / segmentation** — contiguous native-torch-compatible regions into a
   single task where the cost model favors it; re-enable the stubbed
   `Transfer`/`Synchronize` task kinds for device handoff.
3. **Dynamic programming / cost model** — a DP pass over the partition space
   (already has `cost_model.cpp` + `transfer_scheduler.cpp` scaffolding) to pick
   the cheapest valid segmentation; this is the last step and is gated on the
   correctness milestone.

---

## 9. Blockers & open decisions (need user input before implementing)

1. **Which partitioning strategy** — A (planner-level multi-backend) is
   recommended; B (MLIR-embeds-native-torch) is rejected for v0. Confirm A.
2. **Where the op-name/attribute mapping table lives** — planner-owned (Layer 2,
   no ATen types) vs. a Layer 1 adapter lookup. Recommended: planner-owned,
   calling the Layer 2 `nt::` API only.
3. **The uncommitted Stage E CPU migration** must be landed (or at least made
   independent of) the co-execution milestone — the two interleave in
   `third_party/native-torch/`. Confirm sequencing.
4. **CUDA in the first loop or not** — recommended CPU-only first, CUDA as a
   second loop reusing the committed `to_device` path.
5. **Zero-copy vs copy** for the `Bytes ↔ Tensor` bridge — recommended copy.

---

## 10. Done / not-done / blocked at a glance

| Item | State |
|---|---|
| MLIR → JIT → C ABI chain (7 ops, ResNet-18 verified) | ✅ done, committed |
| native-torch dispatcher (CPU kernels + CUDA loop) | ✅ committed (`f667257`) |
| op routing / `Bytes↔Tensor` / op-name mapping / `NativeTorchBackend` / multi-backend executor (G-1…G-5) | ✅ done (`6a8030f`) |
| First end-to-end co-execution loop (`[softmax → relu]`, CPU) | ✅ done (`6a8030f`) |
| Branch/merge DAG correctness + cost benchmark | ✅ done (`f973e20`) |
| Region-partitioned mixed co-execution (merge consecutive MLIR nodes) | ✅ done (`0a12303`) |
| Fusion / DP / cost model | ⏸ deferred (step 4) |

---

## 11. Implemented since the audit

The gap (§4) and the first milestone (§7) have been closed on
`native-torch/refactor`, using partitioning strategy **A** (planner-level
multi-backend, §5) and the copy-based `Bytes ↔ Tensor` bridge (§6):

- **G-1 op routing** — `planner/partition.h` `route_op()`: `Softmax` →
  `NativeTorch`, everything else → `Mlir`. `StaticPlanner::plan` branches on
  "any node routes to NativeTorch" (`has_native`): homogeneous MLIR graphs keep
  the single whole-graph fast path; otherwise the mixed path runs.
- **G-2 `sx::Bytes ↔ nt::Tensor`** — `NativeTorchBackend` materializes host
  float32 buffers as `nt::Tensor` and copies results back.
- **G-3 op-name mapping** — `NativeTorchBackend` dispatches the loop's ops to
  `aten::_softmax` / `aten::relu` / `aten::add.Tensor`.
- **G-4 `NativeTorchBackend`** — `core/src/planner/native_torch_backend.cpp`,
  a peer to `CpuBackend`.
- **G-5 multi-backend executor** — `RuntimeExecutor` takes a
  `map<BackendTag, Backend*>` plus per-task `set_task_backend()` overrides, and
  hands `sx::Bytes` between tasks by `TensorId`.

The mixed path evolved in two steps:

1. `6a8030f` — one compute task per node (the §7.1 `[softmax → relu]` loop);
   `slice_document()` extracts each MLIR node's sub-document.
2. `f973e20` + `0a12303` — branch/merge DAG + cost benchmark, then **region
   partitioning**: topologically-consecutive same-backend nodes merge into one
   region (NativeTorch stays single-node; a run of MLIR nodes compiles as one
   unit), so region-internal tensors never cross a task boundary.
   `slice_region()` extracts an explicit node set into a standalone Document.
   `test_coexecution_merge` / `test_coexecution_region` /
   `test_coexecution_benchmark` verify region structure and numeric results.

§9 open decisions resolved: strategy A (1), planner-owned mapping (2), CPU-only
first loop (4), copy bridge (5). Stage E CPU-migration sequencing (3) is tracked
in `design/native-torch-migration.md`.
