# Static Execution Planner v0

The static execution planner turns a parsed SonicBoom S-Expr v0.1 `Document`
into an immutable, independently-verifiable `ExecutionPlan` that the runtime
executor executes without re-deciding anything. Planning is **static** (happens
once, before execution) and **deterministic** (identical inputs → identical
plan).

## Mission

For v0, CPU is the only backend required to execute real workloads. The planner
therefore produces a plan for CPU-only, whole-graph execution, while the
resource model still *declares* GPU/accelerator/pinned/persistent-storage
capabilities so they can be represented honestly (a capability is usable only
when implemented and verified — v0 implements none of them).

## Responsibilities

Seven components, each with a single responsibility:

| Component | Responsibility |
|---|---|
| `StaticPlanner` | build the task DAG, assign device/memory space, verify capability, order, cost |
| `MemoryPlanner` | tensor lifetimes, buffer allocation, peak working set, budget check |
| `TransferScheduler` | verify no cross-space movement is required (reject otherwise) |
| `PlanValidator` | independently re-validate a plan's structural invariants |
| `RuntimeExecutor` | execute a validated plan; never replan/migrate/alter it |
| `Backend` (CPU) | actually run the whole-graph JIT entry |
| `Telemetry` | measured wall-clock time, kept separate from the cost model |

The pipeline entry point `plan_execution(graph, snapshot, cost_model)` composes
the first four, returning a fully validated `ExecutionPlan`.

## Granularity

Three levels, matching the frozen hybrid-granularity decision:

- **graph/subgraph** — the whole graph is one compiled unit (one JIT entry), so
  the planner emits one `Compute` task covering every `GraphNodeId`.
- **operator/task** — the graph adapter retains per-operator `GraphNodeId`s and
  the planner checks each operator/dtype against the device capability; cost is
  decomposed per operator and summed.
- **tensor/buffer** — every tensor gets a logical buffer and a lifetime; the
  whole-graph model keeps all of them live simultaneously.

## Determinism

All ids, the execution order, both fingerprints (FNV-1a), and the cost estimates
are derived deterministically. `plan_execution` over the same graph/snapshot
yields the same `plan_id` bit-for-bit (verified by `test_planner_benchmark`).

## Boundaries

- All planner types are pure SonicBoom C++ under `sonicboom::planner`; no
  MLIR/LLVM/ATen/c10 type appears in any public planner header.
- The planner consumes the frozen `sx` IR and the MLIR-free `sx::Executable`
  facade; it never mutates the frozen `sx` IR and never touches
  `third_party/native-torch` or `third_party/llvm-project`.
- The frozen S-Expr spec, the parser, the lowering, the CPU JIT, and the C ABI
  are all preserved unchanged.

## File layout

- `core/include/sonicboom/planner/*.h` — public headers.
- `core/src/planner/*.cpp` — implementations (plus private `fingerprint.h`).
- `tests/core/test_planner_*.cpp`, `test_execution_plan.cpp`,
  `test_runtime_executor.cpp` — unit/acceptance tests.
