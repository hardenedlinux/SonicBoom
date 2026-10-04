# Static Execution Planner v0 — Acceptance

Acceptance criteria, mapped to the test that verifies each one. All tests link
`libsonicboom.so` (the published shared library) and include only public
`<sonicboom/planner/...>` / `<sonicboom/sx/...>` headers.

## How to run

```bash
cmake -S . -B build && cmake --build build -j
# Tests run by executing each binary under build/tests/core/ (the build does
# not register add_test targets, so ctest finds none — pre-existing):
for t in build/tests/core/test_*; do "$t"; done
```

## Criteria

| # | Criterion | Test |
|---|---|---|
| 1 | Resource snapshot is valid; fingerprint is deterministic and changes with content; a corrupt snapshot is rejected | `test_planner_resource` |
| 2 | Cost model estimates valid costs and rejects unknown/unsupported inputs | `test_planner_cost_model` |
| 3 | Graph adapter assigns deterministic ids, classifies inputs/constants/external tensors, preserves attributes, and rejects unknown ops / unresolved refs / duplicates / overflow | `test_planner_graph` |
| 4 | ExecutionPlan carries schema version, task payloads, and working find_* helpers | `test_execution_plan` |
| 5 | Static planner emits one whole-graph compute task, verifies capability, is deterministic, and rejects unsupported op/dtype/device and invalid snapshots | `test_planner` |
| 6 | Memory planner computes lifetimes, aligns buffers, enforces the budget, and reports peak + memory cost | `test_memory_planner` |
| 7 | Plan validator rejects duplicate ids, missing deps, cycles, missing devices, invalid transfer endpoints, order violations, and fingerprint mismatch | `test_plan_validator` |
| 8 | Transfer scheduler records zero transfers for one space and rejects cross-space requirements | `test_transfer_scheduler` |
| 9 | Runtime executor runs the whole-graph CPU entry correctly, and fails hard on resource mismatch or corrupt plan (no replanning) | `test_runtime_executor` |
| 10 | Telemetry timer/aggregate behave monotonically and accumulate correctly | `test_planner_telemetry` |
| 11 | Planning is fast and deterministic (1000 identical plans) | `test_planner_benchmark` |
| 12 | The full frozen ResNet-18 fixture plans end-to-end (49 nodes, peak within budget, deterministic) | `test_planner_acceptance` |
| 13 | Existing S-Expr/MLIR/JIT/C-ABI tests still pass (no regression) | pre-existing 14 test binaries |

## Frozen-contract constraints (no-regression guarantees)

- The frozen S-Expr v0.1 spec, parser, MLIR lowering, CPU JIT, and C ABI are
  unchanged (criterion 13).
- No `third_party/llvm-project` or `third_party/native-torch` source is
  modified.
- No per-operator JIT entry is fabricated: the single whole-graph entry is the
  only compiled unit, and the planner's compute task matches it.
- CPU is the only executable backend; GPU/pinned/async capabilities are declared
  only as `false` (nothing is fabricated).
- Planning is static and deterministic; the runtime never replans, migrates, or
  alters the plan.

## Results (this build)

- All 26 planner + core test binaries pass (see the status document).
- `test_planner_acceptance`: ResNet-18 plans as **49 nodes**, peak working set
  **70,329,856 bytes** within the effective host budget, deterministically.
- `test_planner_benchmark`: ~2 µs/plan for a 3-op chain (environment-dependent,
  informational).
