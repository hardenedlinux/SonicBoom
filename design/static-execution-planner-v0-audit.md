# Static Execution Planner v0 — Pre-Commit Audit

Independent, evidence-based audit of the completed Static Execution Planner v0
implementation. This report does not trust the prior delivery report; every
claim below was re-derived from the working tree, the source, and live test
execution.

## 1. Executive verdict

**PASS WITH FINDINGS.**

- No Critical or High severity findings.
- One Medium finding was identified and **fixed** (the plan validator did not
  validate tensor lifetimes, buffer allocations, or the memory summary — see
  §4 F1 and §11).
- Remaining findings are Low / Informational and do not block commit.
- Frozen contracts (S-Expr v0.1 spec, parser, MLIR lowering, CPU JIT, C ABI)
  are untouched; `third_party/*` is clean; the ONNX importer files are
  preserved; nothing was committed or pushed.

## 2. Working-tree scope

Baseline: branch `master`, `HEAD = 6eea2b4` ("Update docs"). The M2–M5 native-
torch audit commits (`21de1ad`…`d872db3`) are ancestors of HEAD.

Committed but modified in the working tree (2 files, unstaged):

- `core/CMakeLists.txt` — adds a `GLOB_RECURSE ... CONFIGURE_DEPENDS` for
  `src/planner/*.cpp` and adds it to the `sonicboom` SHARED library sources.
- `tests/core/CMakeLists.txt` — adds 11 planner tests to a new `foreach`, and
  adds `test_onnx_import_resnet18_exec` + `test_planner_acceptance` to the
  integration `foreach`.

Untracked (new, not committed):

- `core/include/sonicboom/planner/` — 17 public headers.
- `core/src/planner/` — 15 translation units + private `fingerprint.h`.
- `tests/core/` — 12 new test sources (11 planner + 1 acceptance).
- `design/` — 5 planner design/status docs + the ONNX importer doc
  (`onnx-to-s-expr-importer.md`, pre-existing).
- `tools/onnx2sx/` — ONNX importer (pre-existing).

Preserved (not modified by the planner work or this audit):

- `tools/onnx2sx/`, `design/onnx-to-s-expr-importer.md`,
  `tests/core/test_onnx_import_resnet18_exec.cpp`.
- `third_party/*` — `git status third_party/` is empty (clean).
- All committed S-Expr / parser / lowering / JIT / C ABI sources.

### Files changed during this audit

- `core/src/planner/plan_validator.cpp` — added lifetimes / allocations /
  memory-summary validation (§11).
- `tests/core/test_plan_validator.cpp` — added 9 negative tests (§11).
- `design/static-execution-planner-status.md` — corrected the test-binary count
  (22 → 26), recorded the audit verdict/fix, and documented that `ctest` finds
  no tests (pre-existing).

## 3. Architecture and data-flow trace

```
sx::Document (core/include/sonicboom/sx/ir.h)
  └─ adapt_graph(doc)                          graph_adapter.cpp
       └─ Graph { tensors, nodes, inputs, outputs }   graph.h
            └─ model_fingerprint(graph)        graph_adapter.cpp (FNV-1a)

ResourceSnapshot = CpuResourceProvider::snapshot()   resource.cpp
  └─ resource_fingerprint(snapshot)            resource.cpp (FNV-1a)
  └─ validate_snapshot(snapshot)               resource.cpp

plan_execution(graph, snapshot, cost_model)    pipeline.cpp
  1. StaticPlanner::plan(graph)                planner.cpp
       - validate_snapshot, find lowest-id CPU + Host
       - per-node capability (op + non-constant dtype) check
       - emit ONE Compute task (OpKind::WholeGraph, jit_entry=0,
         graph_nodes = all node ids, inputs = graph inputs + constants,
         outputs = graph outputs)
       - per-node cost decomposition → summed ComputeCost
       - plan_id = FNV-1a(model_fp, resource_fp)
  2. MemoryPlanner::plan_memory(plan)          memory_planner.cpp
       - one TensorLifetime per tensor (all live for the single task)
       - one aligned BufferAllocation per tensor (no reuse)
       - peak = sum of aligned sizes; budget check → InsufficientMemory
  3. TransferScheduler::schedule(plan)         transfer_scheduler.cpp
       - verify every task tensor's buffer space == task space; else
         UnsupportedCapability; zero transfer cost
  4. PlanValidator::validate(plan)             plan_validator.cpp
       - structural re-check (see §5)
  └─ ExecutionPlan (plain-data, immutable)     execution_plan.h

RuntimeExecutor::execute(plan, live, inputs)  runtime_executor.cpp
  1. resource_fingerprint(live) != plan.resource_fingerprint → ResourceChanged
  2. re-validate plan → InvalidPlan
  3. count graph inputs/outputs; bound buffers
  4. walk execution_order → dispatch single Compute task to Backend
       └─ CpuBackend::execute(task, inputs)   cpu_backend.cpp
            └─ sx::Executable::run(input, out)  (existing whole-graph JIT)
```

Error propagation: every stage returns `std::expected<T, PlannerError>` (or
`RuntimeError` at execution). No planner API throws for a *logical* failure; the
C ABI (`capi/`) is unchanged and maps `sx`/`planner` failures onto its opaque
status model.

## 4. Findings by severity

### Medium

**F1 — PlanValidator did not validate lifetimes, allocations, or memory summary.**
`plan_validator.cpp` checked schema version, devices, memory spaces, tensors,
tasks, the dependency DAG, the execution order, and the resource fingerprint —
but never examined `lifetimes`, `allocations`, or `memory_summary`. A manually
corrupted plan (dangling lifetime reference, duplicate buffer id, unknown
memory space, zero alignment, tensor hosted by two buffers, peak exceeding
budget) passed validation. For an "independent trust boundary" this is a real
gap. **Fixed** (§11): the validator now rejects all of the above. Severity
Medium because v0's planner always produces well-formed structures, so the gap
was latent (no live plan was affected), but the validator is explicitly a
defense-in-depth boundary.

### Low

**F2 — Unchecked `uint64_t` accumulation in cost decomposition.**
`planner.cpp` computes `in_bytes += t->size_bytes` / `out_bytes += t->size_bytes`
without overflow checks. In practice these are bounded by already-checked
per-tensor sizes and cannot overflow for any real graph, but the arithmetic is
not formally overflow-safe (audit §C). Documented, not fixed (non-reproducible;
a fix would add noise for zero practical benefit).

**F3 — Transfer scheduler silently skips a task tensor with no buffer.**
`transfer_scheduler.cpp` returns success when a task input/output tensor has no
buffer (`tensor_space.end()`). In the pipeline this branch is unreachable
(memory planning runs first and allocates every tensor), so it is not a live
defect; it is a latent leniency for a standalone invocation. Documented, not
fixed.

### Informational

- **F4 — Float attributes are fingerprinted via `std::bit_cast<double→uint64_t>`
  and then serialized as a numeric value**, which is endian-dependent. The
  `fingerprint.h` comment claims "endian-independent"; that holds for integers
  but not for the float/double `bit_cast` path. Irrelevant on the v0 x86-64
  target; noted for portability.
- **F5 — `AnalyticalCpuCostModel::estimate_transfer` does not reject
  `source == destination`** (same-space "transfer"). The validator already
  rejects same-endpoint *transfer tasks*, and v0 emits no transfer tasks, so
  this is inert.
- **F6 — No compiler warning flags are configured** (`-O3 -DNDEBUG -std=c++23
  -fPIE` only; no `-Wall/-Wextra/-Werror`). A manual `-Wall -Wextra -Wpedantic
  -Wshadow -Wconversion` pass over all 15 planner translation units produced
  zero warnings.
- **F7 — Tests are not registered via `enable_testing()`/`add_test()`**, so
  `ctest --test-dir build` reports "No tests were found". This is pre-existing
  (predates the planner) and the acceptance doc's `ctest` command is therefore
  inaccurate. Tests are run by executing the binaries under `build/tests/core/`.
- **F8 — `model_fingerprint` and `ComputeTaskDesc::jit_entry` cannot be
  independently re-checked at validation time.** The plan does not carry the
  source `Graph` (so `model_fingerprint` cannot be recomputed) and has no list
  of JIT entries (so `jit_entry` cannot be resolved). Both are carried as
  provenance metadata; `jit_entry` is not consulted by `CpuBackend` (it runs
  the single compiled `Executable` directly). Documented limitation.
- **F9 — `ComputeTaskDesc::jit_entry`, `TaskDesc::id`, and `TaskDesc::kind`
  lack default member initializers.** Every production path assigns them before
  use (verified), and `{}` value-initialization is used in tests, so there is no
  uninitialized read. Code-hygiene note only.

## 5. Correctness / safety results by area

- **IDs / errors / ownership** — 7 distinct strong-ID types (C++23 defaulted
  `<=>`), `next_id` is overflow-checked; `PlannerError`/`RuntimeError` carry
  code/message/phase/optional context/optional nested cause. No raw owning
  pointers in the planner (only `std::unique_ptr` for the compiled
  `Executable`); all buffers are `std::vector<std::byte>` (RAII). No exception
  escapes for logical failures (all `std::expected`). No manual memory
  management to leak.
- **Resource model / fingerprints** — device vs memory-space are distinct and
  validated; `validate_snapshot` rejects duplicate ids, unknown owners, zero
  alignment, and reserved-over-capacity; fingerprints are FNV-1a over canonical
  vector-ordered serializations (deterministic, no unordered-container
  iteration); CPU is the only `can_execute` device; GPU/pinned/async are
  declared `false`. Fingerprint recomputation in the validator and executor is
  correct (verified by `ResourceChanged` on live-snapshot mismatch).
- **Cost model / arithmetic** — deterministic analytical estimates; unknown
  device/op/dtype and `WholeGraph`-without-decomposition are rejected; invalid
  (non-finite/negative) costs are rejected; `PlanCostSummary::summarize`
  rejects non-finite totals. Single-op byte counts use checked `tensor_byte_size`;
  the only unchecked arithmetic is the informational F2 accumulation.
- **Graph adapter / task DAG** — deterministic id assignment (inputs, then
  parameters, then node outputs, document order); unknown ops →
  `UnsupportedOperator`; unresolved/duplicate refs → `InvalidGraph`; byte-size
  overflow → `ArithmeticOverflow`; constants are correctly distinguished from
  computed tensors for the dtype capability check (int64 axes/shapes are baked);
  whole-graph model uses the real single JIT entry (no fabricated per-op
  entries); dependencies acyclic and topological (Kahn).
- **Memory planning** — one lifetime per tensor, all live for the single task;
  one aligned buffer per tensor, no reuse (lifetimes overlap); checked
  alignment and peak accumulation; budget check with
  `InsufficientMemory` + required/available bytes.
- **Transfer scheduling** — verifies buffer-space == task-space, rejects
  cross-space with `UnsupportedCapability`; records zero transfer cost; no
  fictitious GPU/pinned/async ops.
- **Plan validation** — see §5 fix; now reconstructs and re-checks the
  structural invariants (plus the resource fingerprint) rather than trusting
  them.
- **Runtime executor / CPU backend** — re-checks the live fingerprint
  (`ResourceChanged`, no replanning), re-validates, bounds I/O counts, walks the
  validated order, and rejects Transfer/Allocate/Release/Synchronize tasks
  explicitly (no silent no-op). Backend maps `sx::ExecError` → `RuntimeError`.
  Telemetry (steady-clock) is separate from the cost model and never feeds
  planning.

## 6. Determinism

Methodology: `test_planner_benchmark` plans a 3-op graph **1000 times** and
asserts the `plan_id` is byte-identical every iteration (passes). The acceptance
test plans ResNet-18 twice and asserts identical `plan_id` (passes). Source
review confirms determinism is by construction: ids are sequential counters in
document order; fingerprints iterate vectors (never `unordered_*`); no
timestamps, pointer addresses, random seeds, or uninitialized memory enter any
deterministic field. `CpuResourceProvider::snapshot` in auto mode reads OS
physical RAM, so the *snapshot* (hence the plan) is machine-dependent; with an
explicit `host_memory_budget_bytes` the entire plan is bit-reproducible across
runs. Runtime telemetry is deliberately measured (non-deterministic) and is kept
out of plan generation.

## 7. Build / test commands and outcomes

```bash
cmake -S . -B build
cmake --build build -j
# ctest --test-dir build            # finds no tests (no add_test; pre-existing)
cd build/tests/core && for t in $(ls | grep '^test_'); do ./$t; done
```

Outcome: **26 / 26 test binaries pass** (9 M2 boundary + 3 ResNet-18 S-Expr +
`test_capi` + `test_onnx_import_resnet18_exec` + 11 planner + 1 acceptance).

Compiler: `/usr/bin/g++-13`, `-O3 -DNDEBUG -std=c++23 -fPIE`.

## 8. Numerical ResNet-18 results (observed)

- `test_planner_acceptance`: ResNet-18 plans as **49 nodes**, peak working set
  **70,329,856 bytes**, within budget, deterministic.
- `test_s_expr_resnet18_exec` **and** `test_onnx_import_resnet18_exec`:
  max abs error `3.33786e-06`, max rel error `0.000115104`, argmax `107` vs
  ONNX reference `107`.

## 9. Sanitizer and ABI coverage

- **Compiler warnings**: `-Wall -Wextra -Wpedantic -Wshadow -Wconversion
  -fsyntax-only` over all 15 planner sources → **zero warnings**.
- **ASan + UBSan**: 10 planner test binaries were rebuilt with
  `-fsanitize=address,undefined` (planner sources instrumented) and run clean
  (`detect_leaks=0`, `halt_on_error=1`): **10/10 pass, no sanitizer reports**.
  Coverage note: `sx`/native-torch/MLIR (inside the prebuilt `libsonicboom.so`)
  are **not** instrumented; the sanitizer instruments only the planner's own
  code plus the `sx::Executable` call boundary.
- **ABI / symbols**: `ldd build/core/libsonicboom.so` shows no shared MLIR/LLVM/
  aten/c10 dependency; `nm -D` shows the 12 `sb_*` C-ABI symbols exported with C
  linkage. `test_capi` (compiled as C) passes.
- **Public-header dependency scan**: no `#include` of `mlir/`, `llvm/`, `ATen/`,
  `c10/`, `torch/`, or `caffe2/` anywhere under `core/include/sonicboom/planner/`
  or `core/src/planner/`.

## 10. Architectural extensibility classification

1. Replaceable static planner through a stable abstraction — **EXTENDABLE**
   (`StaticPlanner` is concrete and `plan_execution` hardcodes the pipeline;
   the resource/cost/plan/validator contracts are already separable).
2. Sharing resource model, cost model, plan schema, validator across planners —
   **READY** (all four are independent, pure-data/interface types; only the
   pipeline composition is hardcoded).
3. Multiple execution regions / JIT entries — **EXTENDABLE** (schema already
   allows multiple tasks each with `jit_entry` + `graph_nodes`; the executor/
   `CpuBackend` currently assume one whole-graph entry).
4. Associating strategies with batch/subgraph/task group — **EXTENDABLE** (no
   strategy field today; would need a compatible schema addition).
5. Telemetry feeding a decision layer without contaminating the planner —
   **READY** (telemetry is header-only and fully separated; the deterministic
   planner never reads it).
6. Future planner producing a validated replacement without silent replanning —
   **READY** (the executor only executes a supplied plan and hard-fails on
   mismatch; it has no planning path).
7. Schema versioning / compatibility — **EXTENDABLE** (exact-match versioning
   exists; no forward-compat negotiation policy).
8. Resource model distinguishes device / memory space / executable capability —
   **READY** (device kind, capability flags, memory kind/owner are cleanly
   separated and honestly declared).
9. Global budgets + concurrent per-strategy reservations — **EXTENDABLE** (a
   single global `reserved_bytes` is representable; per-strategy concurrent
   reservations would need a schema addition).
10. Premature abstractions — **READY** (the unused `TaskKind` variants,
    `DeviceKind::GPU/Accelerator`, `MemoryKind::DeviceLocal/PinnedHost/
    PersistentStorage`, and the transfer/allocate/release payloads are
    contractually required by the frozen "represent honestly" rule, not
    speculative; `offset_bytes` is inert but present for the buffer model).

## 11. Fixes made and regression tests

**F1 fix** — `plan_validator.cpp` now additionally validates:
- each `TensorLifetime`: tensor, producer (if any), consumers, and
  first/last-required all resolve;
- each `BufferAllocation`: unique buffer ids, resolvable memory space, non-zero
  alignment, resolvable assigned tensors, and no tensor hosted by two buffers;
- `memory_summary`: keys resolve to recorded memory spaces and a recorded peak
  never exceeds its recorded effective budget.

Regression tests added to `test_plan_validator.cpp` (all pass):
valid plan with lifetimes/allocations passes; lifetime→unknown tensor rejected;
lifetime→unknown producer rejected; duplicate buffer id rejected; allocation→
unknown memory space rejected; zero-alignment allocation rejected; tensor in two
buffers rejected; memory-summary unknown space rejected; peak-over-budget
rejected.

Verification: full suite **26/26 pass** after the fix; the strengthened
validator test also passes under ASan+UBSan.

## 12. Remaining limitations and known risks

- v0 is whole-graph/single-entry; the plan schema can express more (multiple
  tasks, jit entries, transfers) but the executor/backend do not execute them
  yet — by design (§ EXTENDABLE).
- `model_fingerprint` and `jit_entry` are provenance metadata that the validator
  cannot independently re-derive (F8).
- The "no overlapping live buffers" invariant is not byte-address-checkable in
  the current schema (there is no shared arena; every buffer is an independent
  logical allocation with `offset_bytes = 0`). The validator instead enforces
  "no tensor hosted by two buffers", which is the expressible form in v0.
- `ctest` is not wired up (pre-existing; F7).

## 13. Commit recommendation

**READY TO COMMIT.**

No Critical or High findings. The single Medium finding was fixed and
regression-tested. Frozen contracts, `third_party/*`, and the ONNX importer
files are untouched. Nothing was committed or pushed during this audit.
