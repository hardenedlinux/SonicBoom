# Decision registry

Every automatic choice SonicBoom makes for execution — a reduction mode, a
kernel tile, an algorithm switch — is a *decision*. This document defines how
those decisions are made, keyed, recorded, overridden and inspected.

The rule it implements:

> Every automatic choice is decided once per device (or per static shape
> class), recorded in the decision registry, keyed into the plan cache and the
> autotune cache, overridable by env, and shown by `sb-explain`.

## Identity and scope

A decision is identified by `(name, scope)`:

- `name` names the choice (`reduce.mode`, `tune/gemv`, ...).
- `scope` is `device`, `cpu`, or a static shape class (`bpr10-r2048`,
  `64/512-1023/8-15`, ...). Shape classes bucket the problem coarsely; they
  never contain a batch size or a KV position, so a reused prefix and a full
  prefill resolve the same way.

The first record for a `(name, scope)` pair wins. Repeating the same value is a
no-op (the common case: a memoized per-process lookup); repeating a different
value is a conflict, counted and dropped rather than silently replacing an
earlier decision. `sb-explain` prints the conflict count and exits 2 if any.

## Sources and precedence

| source | meaning |
|---|---|
| `env` | an environment variable forced the value |
| `cached` | a record read from the per-device cache |
| `measured` | timed in this process |
| `verdict` | an acceptance verdict recorded by `tools/sb-numerics` |
| `skipped` | the cold-tuning budget expired; the static default stands |
| `default` | the static default, with no measurement and no cached pick |

Precedence, highest first: environment override, process memo, disk record,
static default. `SONICBOOM_AUTOTUNE=force` skips the disk record so the class is
retimed; `SONICBOOM_AUTOTUNE=0` skips measurement entirely and keeps the
defaults.

## Cache keys

Two services persist decisions across processes:

- **Hand-written tuner** (`core/src/planner/autotune.{h,cpp}`):
  `<decision root>/<gpu>-sm<arch>-<driver>-reduce-<mode>/<op>.json`. The device
  identity and the reduction mode are part of the key; a record is keyed by its
  shape class. Versioned JSON (`__format__`), written by temp file + rename.
- **JIT candidate tuner** (`core/src/planner/mlir_autotune.{h,cpp}`):
  `<cache>/sonicboom/autotune/sm_<NN>.bin`, one versioned fixed-record file per
  compute capability, flock-protected. Records are keyed by `(op, shape
  class)`.

The decision root is `$SONICBOOM_TUNE_DIR`, else
`$XDG_CACHE_HOME/sonicboom/tune`, else `$HOME/.cache/sonicboom/tune`
(`SONICBOOM_TUNE_DIR` is the test seam; `XDG_CACHE_HOME` covers both services).

The **plan cache** (prefill segments, `PlannerEnv` in
`core/src/model/gemma4_plan.cpp`) carries `decisions::fingerprint()`: an FNV-1a
hash of the sorted `(name, scope, value)` triples recorded so far. A newly
recorded decision changes the fingerprint, invalidates the cached segments,
and makes the next segment build use the new set. Plan-record goldens pin
`SONICBOOM_AUTOTUNE_JIT=0` because their text is the static kernel geometry,
not a per-device measurement.

## Environment overrides

Each decision type has a switch that overrides it wholesale:

| decisions | override |
|---|---|
| reduction mode | `SONICBOOM_REDUCE=f64\|df32\|f32` |
| hand-written tuner picks | `SONICBOOM_AUTOTUNE=0` (defaults), `=force` (retime) |
| JIT candidate picks | `SONICBOOM_AUTOTUNE_JIT=0` (defaults), `SONICBOOM_AUTOTUNE_BUDGET_MS` |
| spec probe ratio | `SONICBOOM_SPEC=off\|on\|auto` (auto uses the record) |

The static algorithm switches (`SONICBOOM_MMQ`, `SONICBOOM_ATTN_PREFILL`, ...)
keep their documented meaning; `sb-explain` prints each one with its default
and whether it is set.

## Current inventory

| name | scope | stored in |
|---|---|---|
| `reduce.mode`, `reduce.df32-acceptance`, `reduce.double-cost-ratio`, `reduce.cpu-measurement` | device / cpu | registry (verdict files on disk) |
| `tune/attention-prefill`, `tune/attention-decode`, `tune/dense-gemm`, `tune/mmq-j`, `tune/spec-length` | shape class | hand-written JSON |
| `tune/gemv`, `tune/matmul_q8_tc`, `tune/attn_prefill` | shape class | per-SM JIT file |
| `spec_verify_ratio` | `decode` | per-SM JIT file |

## sb-explain

`tools/build_sb_explain.sh` builds `tools/sb-explain` against
`libsonicboom.so`. It prints, without tuning or writing anything:

1. the device identity and both cache paths;
2. every relevant `SONICBOOM_*` switch with its current value, default and
   meaning;
3. the resolved reduction modes;
4. the decision registry for the decisions the tool itself resolves;
5. the hand-written tuner choices on disk;
6. the JIT candidate records on disk;
7. the static JIT defaults.

Run it before a benchmark to confirm which decisions a run will use; pair it
with `SONICBOOM_AUTOTUNE=force` to see the picks after a retime.

## Adding a decision

1. Give it a stable `name` and a scope that is a device or a static class —
   never a live batch size or position.
2. Resolve it once (function-local static or a memo keyed by the class).
3. Record it with `planner::decisions::record(name, scope, value, source, env)`
   at every path that resolves it: env override, cache hit, measurement,
   static default.
4. Persist it in a cache keyed by device identity (and the reduction mode when
   the kernel's instruction mix depends on it).
5. Add its environment override to the `sb-explain` table and to this file.
