# Static Execution Planner Algorithm v0

Deterministic, single-pass planning of a whole-graph S-Expr model for CPU.
Stages run in a fixed order; each consumes the previous stage's output and is
independently testable.

## Pipeline

```text
sx::Document
   │ adapt_graph
   ▼
Graph ──┬── StaticPlanner::plan        (task DAG, device, order, cost)
        ├── MemoryPlanner::plan_memory (lifetimes, buffers, peak, budget)
        ├── TransferScheduler::schedule(verify no cross-space movement)
        └── PlanValidator::validate     (independent re-check)
   ▼
ExecutionPlan (immutable, verified)
```

`plan_execution(graph, snapshot, cost_model)` composes these four steps.

## 1. Graph adaptation (`adapt_graph`)

Assigns deterministic ids in document order — graph inputs, then parameters
(constants, flagged `is_constant` and `is_external` for `:external` weights),
then node outputs — resolving each node's inputs against previously defined
values. Rejects: an unknown operator (`UnsupportedOperator`), a tensor whose
byte size overflows checked arithmetic (`ArithmeticOverflow`), and an unresolved
or duplicate definition (`InvalidGraph`). Attributes are preserved verbatim.
`model_fingerprint` hashes the result (FNV-1a over tensors, nodes, attributes).

## 2. Static planning (`StaticPlanner::plan`)

1. Validate the snapshot (`validate_snapshot`).
2. Locate the lowest-id CPU device (`can_execute`) and the lowest-id Host memory
   space; a missing one is `UnsupportedCapability`.
3. Capability check per node: the operator must be in the device's
   `supported_ops`, and every **non-constant** input and every output dtype must
   be in `supported_dtypes` (constants are baked, not computed, so their dtype —
   e.g. int64 reshape axes — does not constrain the device).
4. Emit one `Compute` task (`WholeGraph`, `jit_entry = 0`, `graph_nodes` = every
   node id, `inputs` = graph inputs + constants, `outputs` = graph outputs).
5. Cost: decompose into per-node `ComputeRequest`s (sum of input/output bytes,
   output dtype), sum the cost model's `estimate_compute` latencies, take the
   minimum confidence.
6. Fill devices/memory spaces/tensors, both fingerprints, `resource_version`,
   the single-entry `execution_order`, and a deterministic `plan_id`.

### 2.1 Mixed-graph path (co-execution)

When `route_op` classifies any node as `NativeTorch` (`has_native`, i.e. the
graph contains `softmax`), planning takes a second branch instead of the
whole-graph task above. Nodes are grouped into topologically-consecutive
regions by backend — a `NativeTorch` node is always its own single-node region,
and a run of consecutive `Mlir` nodes merges into one region — and one `Compute`
task is emitted per region. Region inputs (external node inputs, first-reference
order), outputs (region-produced tensors that are graph outputs or consumed
outside the region), dependencies (producers of the external inputs), and cost
(sum of per-node estimates, minimum confidence) are computed across the merged
node set; the representative `op` is the first node's op. Each MLIR region is
compiled from a single `slice_region()` sub-document. See
`design/mlir-native-torch-coexecution-audit.md`.

## 3. Memory planning (`MemoryPlanner::plan_memory`)

1. Lifetimes: every tensor is live for the single compute task — graph
   inputs/constants have no producer, node outputs have `producer = task 0`;
   non-outputs have `consumers = [task 0]`.
2. Buffers: one per tensor, size = tensor bytes rounded up to the space
   alignment (`align_up`, checked), no reuse (lifetimes overlap).
3. Peak = sum of aligned buffer sizes; compared against the space's
   `effective_budget_bytes()` — exceeding it is `InsufficientMemory` (with
   required/available bytes in the error context).
4. Memory cost = sum of per-allocation `estimate_memory` latency.

## 4. Transfer scheduling (`TransferScheduler::schedule`)

For each task, every input/output tensor's buffer memory space must equal the
task's memory space. With one host space this always holds; a mismatch is
rejected (`UnsupportedCapability`) because v0 implements no transfer
capability. Transfer cost is recorded as zero.

## 5. Validation (`PlanValidator::validate`)

Structural invariants (see the plan-format document): unique ids, resolvable
references, acyclic dependency DAG (Kahn's algorithm), a topological execution
order, and a resource fingerprint that matches the plan's own
devices/memory spaces.

## 6. Execution (`RuntimeExecutor::execute`)

1. Recompute `resource_fingerprint(live)` and compare to the plan — a mismatch
   is `ResourceChanged` (no silent replanning/migration).
2. Re-validate the plan defensively (`InvalidPlan` on failure).
3. Walk `execution_order`; the single `Compute` task is dispatched to the
   backend, which runs the whole-graph `sx::Executable` entry (one input buffer
   → one output buffer). Transfer/allocate/release/synchronize kinds are
   rejected (`TransferFailure`/`BackendFailure`) since v0 emits none.

## Determinism guarantees

Identical graph + snapshot + cost model produce identical ids, fingerprints,
task DAG, execution order, cost, and `plan_id` — verified by
`test_planner_benchmark` (1000 identical plans).
