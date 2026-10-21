# Static Execution Plan Format v0

`ExecutionPlan` is the explicit, independently-verifiable contract between the
static planner and the runtime executor. It is a plain-data C++ struct
(`core/include/sonicboom/planner/execution_plan.h`) with no MLIR/LLVM/ATen/c10
objects, so it can be inspected and validated without pulling in the execution
implementation.

## Schema

- `schema_version` (`uint32_t`) — currently `1` (`kSchemaVersion`); the
  validator rejects any other value.
- `plan_id` (`uint64_t`) — deterministic identifier (FNV-1a over the model and
  resource fingerprints, v0).
- `model_fingerprint` (`uint64_t`) — FNV-1a over the adapted graph (tensors,
  nodes, attributes), set by `model_fingerprint`.
- `resource_fingerprint` (`uint64_t`) — FNV-1a over the resource snapshot.
- `resource_version` (`uint32_t`) — the snapshot `version` the fingerprint was
  computed over, so the validator can recompute it.

## Resources

- `devices` (`vector<Device>`) — the devices the plan runs on (v0: one CPU).
- `memory_spaces` (`vector<MemorySpace>`) — the memory spaces (v0: one Host).

Both are copied from the `ResourceSnapshot`; the validator recomputes
`resource_fingerprint` from exactly these and compares to the stored value.

## Tensors

- `tensors` (`vector<TensorDesc>`) — every value in the graph (graph inputs,
  parameters/constants, and every node output), each with a stable `TensorId`,
  static shape, `sx::DType`, byte size, classification flags (`is_graph_input`,
  `is_graph_output`, `is_constant`, `is_external`), and its original name.

## Tasks

- `tasks` (`vector<TaskDesc>`) — the task DAG. v0 has exactly one task: a
  `Compute` task with payload `ComputeTaskDesc{ op = WholeGraph,
  graph_nodes = every node id, jit_entry = 0 }`. Its `inputs` are the graph
  inputs + constants, `outputs` are the graph outputs, `device` is the CPU,
  `memory_space` is the host space, and `cost` is the summed per-operator
  estimate. `dependencies` is empty.

Other task kinds (`Transfer`, `Allocate`, `Release`, `Synchronize`) exist in the
model for future multi-task planning; v0 emits none of them.

## Lifetimes

- `lifetimes` (`vector<TensorLifetime>`) — per tensor: optional `producer` (a
  `TaskId`, absent for graph inputs/constants), `consumers`, `first_required`,
  `last_required`. In the whole-graph model every tensor is live for the single
  compute task.

## Buffers

- `allocations` (`vector<BufferAllocation>`) — one aligned logical buffer per
  tensor (no reuse, since lifetimes overlap): `buffer` id, `memory_space`,
  `offset_bytes = 0`, `size_bytes` (tensor size rounded up to the space
  alignment), `alignment_bytes`, and `assigned_tensors`.

## Order and summaries

- `execution_order` (`vector<TaskId>`) — a topological permutation of the task
  ids (v0: `[0]`), respecting dependencies.
- `estimated_cost` (`PlanCostSummary`) — compute/transfer/memory/total in
  microseconds (analytical estimates, not measurements).
- `memory_summary` (`PlanMemorySummary`) — `peak_bytes` and
  `effective_budget_bytes` per memory space.

## Invariants (enforced by `PlanValidator`)

Unique device/memory-space/tensor/task ids; every task's device, memory space,
dependencies, inputs, and outputs resolve; dependency graph is acyclic; the
execution order is a topological permutation; transfer tasks (if any) have
distinct, valid endpoints and positive bytes; allocate/release tasks reference a
known buffer; and the resource fingerprint matches the recorded
devices/memory spaces. Any violation yields a `PlannerError` (`InvalidPlan`, or
`ResourceChanged` for a fingerprint mismatch), never a silently-altered plan.
