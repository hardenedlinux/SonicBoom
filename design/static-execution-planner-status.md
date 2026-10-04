# Static Execution Planner v0 — Status

Living status document for the Static Execution Planner v0 implementation.
Updated as phases complete; describes implemented behavior, not aspirational.

## Baseline (audit snapshot)

- Branch `master`, baseline HEAD `6eea2b4` ("Update docs"). The S-Expr → MLIR
  CPU JIT execution MVP + stable C ABI were already committed (`f5f9e75`).
- The Milestone 2 ONNX importer working-tree files are preserved and not
  committed here.
- Build: `libsonicboom.so` (SHARED), C++23 / g++-13, MLIR 23.1.2 static,
  native-torch (c10 + aten_core) static.

## What already exists (reused, not rebuilt)

| Component | Location | Status |
|---|---|---|
| S-Expr v0.1 IR (`Document`/`Graph`/`Node`/`TensorType`/`DType`) | `core/include/sonicboom/sx/ir.h` | ✓ frozen, reused |
| Checked arithmetic (`numel_checked`, `tensor_byte_size`, `dtype_size`) | `core/src/sx/ir.cpp` | ✓ reused |
| Parser + validator (`parse_document`) | `core/src/sx/parser.cpp` | ✓ reused |
| S-Expr → MLIR lowering (7 ops) | `core/src/sx/lowering.cpp` | ✓ reused |
| CPU JIT (`Executable::compile`/`run`) | `core/src/sx/exec.cpp` | ✓ reused |
| External weight loader (`load_external_weights`) | `core/src/sx/exec.cpp` | ✓ reused |
| Stable C ABI (`sb_*`) | `capi/` | ✓ unchanged |

## What was newly implemented (all under `core/include/sonicboom/planner/` + `core/src/planner/`)

| Component | Files |
|---|---|
| Strong IDs (`next_id`) | `ids.h` |
| Error model (`PlannerError`/`RuntimeError`) | `errors.h`, `errors.cpp` |
| Resource model + CPU provider | `resource.h`, `resource.cpp` |
| Cost model | `cost_model.h`, `cost_model.cpp`, `cost.h` |
| Planner graph + adapter | `graph.h`, `graph_adapter.cpp` |
| Task model + DAG | `task.h`, `task.cpp`, `op_kind.h`, `op_kind.cpp` |
| ExecutionPlan + lifetimes + allocations | `execution_plan.h`, `execution_plan.cpp` |
| Deterministic static planner | `planner.h`, `planner.cpp` |
| Memory planner | `memory_planner.h`, `memory_planner.cpp` |
| Transfer scheduler | `transfer_scheduler.h`, `transfer_scheduler.cpp` |
| Plan validator | `plan_validator.h`, `plan_validator.cpp` |
| Pipeline orchestrator | `pipeline.h`, `pipeline.cpp` |
| Runtime executor + CPU backend | `runtime_executor.h/.cpp`, `backend.h`, `cpu_backend.cpp` |
| Telemetry | `telemetry.h` (header-only) |
| FNV-1a fingerprint (private) | `src/planner/fingerprint.h` |

## Key integration decisions (frozen-contract-compliant)

1. **Whole-graph compute task.** The existing compiler emits exactly one JIT
   entry per graph (`float* <graph_name>(float*)`), not per-operator entries.
   Per §13.2/§17, the planner therefore produces **one** `Compute` task covering
   the whole graph (one `JitEntryId` → one `sx::Executable`), while the graph
   adapter retains per-operator `GraphNodeId`s, capability checks, and cost
   analysis for the hierarchical graph/operator/tensor levels and future
   subgraph fusion. No per-operator JIT entry is fabricated.
2. **Single host memory space.** CPU v0 uses one device (CPU) + one memory space
   (Host); no cross-space transfers are ever required. The `TransferScheduler`
   verifies that invariant and **rejects** a plan that would need a transfer
   (`UnsupportedCapability`) — v0 implements no copy/async capability, and
   reports `supports_async = false`, `supports_concurrent_copy = false` honestly.
3. **Buffer model.** The memory planner records one aligned logical buffer per
   tensor. Because the whole-graph compiled unit keeps every input, weight,
   activation, and output materialized at once, all lifetimes overlap and no
   reuse is possible; the honest static peak is the sum of every tensor's aligned
   size. Actual JIT-internal `memref.alloc`s remain below the planner's
   abstraction (one-shot-bufferized), documented, not planned.
4. **Fingerprints** are deterministic 64-bit FNV-1a over a canonical
   serialization of resource properties / model structure (non-cryptographic,
   documented). The plan records both `model_fingerprint` and
   `resource_fingerprint` (plus `resource_version`); the validator recomputes the
   latter from the plan's own devices/memory spaces.
5. **Cost model** is a deterministic analytical estimator (configurable), not a
   real-latency predictor; measured wall-clock time lives only in telemetry.

## Test results

- 26 C++ test binaries pass: the 14 pre-existing tests (schema/value/operator/
  backend/tensor/mlir/s_expr/s_expr_lowering/s_expr_exec/s_expr_resnet18/
  s_expr_resnet18_mlir/s_expr_resnet18_exec/capi/onnx_import_resnet18_exec) plus
  the 11 new planner tests (resource, cost_model, graph, execution_plan,
  planner, memory_planner, plan_validator, transfer_scheduler, runtime_executor,
  telemetry, benchmark) and the planner acceptance test over ResNet-18.
- `test_planner_acceptance`: ResNet-18 plans as 49 nodes, peak working set
  `70,329,856` bytes, within budget, deterministically.
- The tests are run by executing each binary under `build/tests/core/` (the
  build does not register `add_test` targets, so `ctest` finds none; this is a
  pre-existing build characteristic, not a planner regression).

## Phase status

- P0 audit: **complete**.
- P1 resource + cost model: **complete**.
- P2 graph + task + plan: **complete**.
- P3 static planner: **complete**.
- P4 memory planner: **complete**.
- P5 transfer + validator: **complete**.
- P6 runtime executor + CPU backend: **complete**.
- P7 telemetry + benchmark + acceptance + docs: **complete**.

## Blockers

None identified.

## Pre-commit audit (see static-execution-planner-v0-audit.md)

- Verdict: **PASS WITH FINDINGS**; commit recommendation **READY TO COMMIT**.
- Verified fix: `PlanValidator` now validates tensor lifetimes, buffer
  allocations (unique buffer ids, resolvable memory spaces, non-zero alignment,
  resolvable/unique assigned tensors), and memory-summary keys / peak-vs-budget
  consistency — the trust boundary no longer trusts those fields blindly.
  Negative tests added to `test_plan_validator`.
- No Critical/High findings; remaining findings are Low/Informational and are
  documented in the audit report, not blocking.
