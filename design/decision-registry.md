# Decision registry

Every automatic choice SonicBoom makes for execution — a reduction mode, a
kernel tile, an algorithm switch — is a *decision*. This document defines how
those decisions are made, keyed, recorded, overridden, inspected, and costed.

The rule it implements: every automatic choice is decided once per device (or
per static shape class), recorded in the registry, keyed into the plan cache
and the autotune cache, overridable by env, and shown by `sb-explain` and the
C API.

## The decision record

A decision is `(name, scope)` plus:
| field | meaning |
|---|---|
| `value` | the chosen value, serialized (`f64`, `j=64,split=2`, `br32xbc32`, ...) |
| `source` | `default`, `env`, `cached`, `measured`, `verdict`, `skipped` |
| `env` | the variable that overrides it (empty when none) |
| `inputs` | the facts it used: device identity, SM version, shape class, measured ratio |
| `candidates` | the alternatives considered, `\|`-separated |
| `reason` | the rule that picked the value |
| `numerics` | the numerics class: bit-identical, token-identical, fp64-ref bound, informational |
| `vram_bytes` | declared device-memory cost (0 = unknown) |
| `tuning_ms` | declared first-run tuning cost (0 = none) |

`name` names the choice (`reduce.mode`, `tune/gemv`, `mmq.min-n`, ...).
`scope` is `device`, `cpu`, or a static shape class (`bpr10-r2048`,
`64/512-1023/8-15`). Shape classes bucket coarsely and never contain a batch
size or a KV position, so a reused prefix and a full prefill resolve the same
way (R1: no algorithm selected by `n` inside a class).

The first record for a `(name, scope)` pair wins. Repeating the same value is a
no-op; a different value is a conflict, counted and dropped rather than
silently replacing an earlier decision. `sb-explain` exits 2 if any conflict
was dropped.

## Persistence and precedence

One registry per device, persisted as JSONL at
`<decision root>/<device>/decisions.jsonl` (header line + one record per line,
temp file + rename). A new process loads it as *historical*: the dump marks
those records `(persisted)`, and this process may replace them when it resolves
the same pair differently.

Precedence, highest first: environment override → process memo → persisted
record → static default. `SONICBOOM_AUTOTUNE=force` skips the disk record so
the class is retimed; `=0` skips measurement and keeps the defaults.

## Cache keys

- **Hand-written tuner** (`core/src/planner/autotune.{h,cpp}`):
  `<decision root>/<gpu>-sm<arch>-<driver>-reduce-<mode>/<op>.json`. Device
  identity and reduction mode are part of the key; entries are keyed by shape
  class. Versioned JSON (`__format__`), temp file + rename.
- **JIT candidate tuner** (`core/src/planner/mlir_autotune.{h,cpp}`):
  `<cache>/sonicboom/autotune/sm_<NN>.bin`, one versioned fixed-record file per
  compute capability, flock-protected, keyed by `(op, shape class)`.
- **Spec ratio**: the `spec_verify_ratio` record in the per-SM JIT file.
- **Plan cache**: `PlannerEnv` carries `decisions::fingerprint()`, an FNV-1a
  hash of the sorted `(name, scope, value)` triples. A new decision changes the
  fingerprint and invalidates cached prefill segments. Plan-record goldens pin
  `SONICBOOM_AUTOTUNE_JIT=0` because their text is static kernel geometry, not
  a per-device measurement.

The decision root is `$SONICBOOM_TUNE_DIR`, else
`$XDG_CACHE_HOME/sonicboom/tune`, else `$HOME/.cache/sonicboom/tune`.

## Variables, constraints, objective

- **Variables**: reduction mode, spec probe, autotune level, MMQ (`enabled`,
  `j`, `decomp`, `min-n`), attention (impl, prefill, decode, chunk), dense
  (cuBLAS, Lt, GEMM, min-n), gemv width, async prefetch, CUDA graphs, training
  switches. The static T0b table is seeded by `seed_static_knobs()`.
- **Constraints**: R1 (one kernel per class, chosen by device and class only);
  each entry's numerics gate (bit-identical preferred, else fp64-ref error ≤
  current with top-1 ≥ 99%); cache keys contain every variable that changes a
  graph; no `cudaMalloc` in a capture; VRAM ≤ the device budget.
- **Objective**: minimize kernel time for the model at the operating shape
  (Gemma pp512 kernel time at or below the llama.cpp baseline, decode not
  regressed), subject to the constraints above.

### Interactions

- **VRAM budget** couples async prefetch (its staged buffers), MMQ tensor-core
  `q8_1` copies (weight conversion cache, `SONICBOOM_MMQ_TC_MAX_MB`), CUDA-graph
  capture reuse (one capture per prefill shape), and the KV cache size
  (`n_ctx`). Each new resident buffer shrinks the KV budget; a joint solver
  must price them together.
- **MMQ decomposition/tile** changes summation order, so it is class-wide and
  also pins which plan is reusable for prefix reuse.
- **Attention chunk** interacts with the window (SWA) and split-KV; changing it
  changes the split and therefore token identity, so only the identity rule is
  automatic (R1).
- **Reduction mode** changes the norm kernels' instruction mix and is part of
  the hand-written tuner's device key.
- **Spec mode** depends on the verify/plain step ratio, which depends on the
  MMQ and attention choices; the probe must run after those are resolved.

## Environment overrides

| decisions | override |
|---|---|
| numerics profile (f16 KV, TF32 dense) | `SONICBOOM_NUMERICS=strict\|standard` |
| reduction mode | `SONICBOOM_REDUCE=f64\|df32\|f32` |
| hand-written tuner picks | `SONICBOOM_AUTOTUNE=0` (defaults), `=force` (retime) |
| JIT candidate picks | `SONICBOOM_AUTOTUNE_JIT=0`, `SONICBOOM_AUTOTUNE_BUDGET_MS` |
| spec mode / probe | `SONICBOOM_SPEC=off\|on\|auto`, `SONICBOOM_SPEC_PROBE_MS` |
| the T0b algorithm switches | each keeps its documented variable (table in `decisions.cpp`) |

No algorithm-selecting variable has to be set by hand: each has a default or an
automatic choice, and the registry records which one applied.

## Inventory

| name | scope | storage |
|---|---|---|
| `reduce.mode`, `reduce.df32-acceptance`, `reduce.double-cost-ratio`, `reduce.cpu-measurement` | device / cpu | registry + verdict file |
| `tune/attention-prefill`, `tune/attention-decode`, `tune/dense-gemm`, `tune/mmq-j`, `tune/spec-length` | shape class | hand-written JSON |
| `tune/gemv`, `tune/matmul_q8_tc`, `tune/attn_prefill`, `spec_verify_ratio` | shape class | per-SM JIT file |
| `mmq.*`, `attention.*`, `dense.*`, `gemv.*`, `spec.*`, `train.*`, feature switches | device | seeded from the T0b table |
| `numerics.profile` | process | registry (strict/standard, with its env/application origin) |
| `kv.cache` | `n_ctx=<n>` | registry (storage dtype + the VRAM bytes it allocated) |

## Inspection: sb-explain and the C API

`tools/build_sb_explain.sh` builds `tools/sb-explain` against
`libsonicboom.so`. Without options it prints the device identity and cache
paths, every `SONICBOOM_*` switch (value, default, meaning), the resolved
numerics profile and reduction modes, the registry as a human table, the
caches on disk, and the static JIT defaults. `--json` prints the registry as a JSON document. It tunes
nothing.

The same two views are exported through the C ABI:

```c
sb_status sb_explain_table(char *buf, size_t cap, size_t *len_out, sb_error **);
sb_status sb_explain_json (char *buf, size_t cap, size_t *len_out, sb_error **);
```

Two-call sizing: a NULL/short buffer yields `SB_ERR_BINDING` and the byte count
in `*len_out`.

## Adding a decision

1. Give it a stable `name` and a scope that is a device or a static class —
   never a live batch size or position; resolve it once.
2. Record a full `Entry` at every path that resolves it: env override, cache
   hit, measurement, static default; declare its candidates, reason, numerics
   class and VRAM and tuning costs.
3. Persist it in a cache keyed by device identity (and the reduction mode when
   the kernel's instruction mix depends on it).
4. Add its environment override to the `sb-explain` table and to this file.
