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

The one permitted exception is a knob **proven numerics-neutral**: every one of
its settings writes the same bytes, so a choice that moves with the batch
cannot make two calls that share rows disagree. MMQ's J tile is the declared
case — it selects the token-tile width only, the K-reduction order and the
split-K schedule are independent of it, and the engine suite's
`test_cuda_quant` asserts that every tiled J candidate writes identical output. Its entry gains a fixed
token-count bucket (`…-n<b>`, a static 2× ladder from 128 tokens, never the
live `n`), so a small-batch caller can no longer pin the tile that a
prefill-scale caller (e.g. the DiT at 4113 tokens) needs. Knobs that change the
summation order — MMQ's decomposition, stored as a class-wide record — get no
such bucket; R1 applies to them unchanged.

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
  `boom_fattn` uses a three-component shape class,
  `f<version>/<capability>/<software>/<workload>`, built from probed device
  limits, the driver/runtime/NVRTC versions, the kernel source hash and the
  bucketed workload — never the product name. Version 6 widened the record so
  that key fits; older files are reported as `version-mismatch` and not read.
  See `fattn-autotune.md`. The workload's KV span is bucketed to a power of two
  because every selectable candidate is measured bit-identical to the
  reference; a non-bit-identical candidate must not join that set.
- **Spec ratio**: the `spec_verify_ratio` record in the per-SM JIT file.
- **Plan cache**: `PlannerEnv` carries `decisions::fingerprint()`, an FNV-1a
  hash of the sorted `(name, scope, value)` triples. A new decision changes the
  fingerprint and invalidates cached prefill segments. Plan-record goldens pin
  `SONICBOOM_AUTOTUNE_JIT=0` because their text is static kernel geometry, not
  a per-device measurement.

The decision root is `$SONICBOOM_TUNE_DIR`, else
`$XDG_CACHE_HOME/sonicboom/tune`, else `$HOME/.cache/sonicboom/tune`.

## Environment overrides

| decisions | override |
|---|---|
| numerics profile (f16 KV, TF32 dense) | `SONICBOOM_NUMERICS=strict\|standard` |
| reduction mode | `SONICBOOM_REDUCE=f64\|df32\|f32` |
| hand-written tuner picks | `SONICBOOM_AUTOTUNE=0` (defaults), `=force` (retime) |
| JIT candidate picks | `SONICBOOM_AUTOTUNE_JIT=0`, `SONICBOOM_AUTOTUNE_BUDGET_MS` |
| `boom_fattn` kernel candidates | `SONICBOOM_FATTN_AUTOTUNE`, `SONICBOOM_FATTN_CANDIDATE` (see `fattn-autotune.md`) |
| spec mode / probe | `SONICBOOM_SPEC=off\|on\|auto`, `SONICBOOM_SPEC_PROBE_MS` |
| the T0b algorithm switches | each keeps its documented variable (table in `decisions.cpp`) |
| attention class (any op) | `SONICBOOM_ATTN_CLASS=<class>`; `SONICBOOM_DIT_ATTN` is kept as its alias |

Attention class selection is keyed on the **op's attributes**, never on the
model that issued the call. The op carries `causal`, `kv_cache`
(present/absent), `prefix_reuse` (possible/not), `gqa`, `head_dim` and
`window`; the registry key is those attributes plus the device:

- `causal1` and/or a KV cache present → the LLM prefill/decode classes. R1
  applies unchanged: one kernel per class, nothing chosen by `n`.
- non-causal, no KV cache, no prefix reuse → the full-attention classes,
  keyed `causal0-kvcache0-prefixreuse0-window0-d<head_dim>-g<gqa>`, split by
  the score-plane capacity boundary:
  - `full.score-gemm` at or below `kFullScoreMaxTokens = 2048` positions: the
    per-head `[tokens, tokens]` f16 score planes fit the scratch budget, so
    S and P are batched cuBLAS f16 GEMMs with a row softmax between them;
  - `full.tiled` above it: online-softmax tiling with no score planes. The
    default is the generated FA-2 warp-head kernel (one warp per 16 query
    rows of one head holding the whole head_dim, P kept in registers as the
    PV mma's A operand, K/V double-buffered with cp.async, online-softmax
    rescale per tile); the hand-written tiled kernel is an autotune
    candidate, kept only if it beats the default by the 3% switch margin on
    the fixed 2048-token probe (`tune/attention-full`, one entry per device,
    cached in `attention-full.json`). `full.scalar` is the pre-fattn
    fallback, reachable only by override.

`full_attn` (the mask kind) is part of the generated kernel's region identity
and the generator's support rules reject a causal, windowed or cached shape
for it, so a full-attention cache entry can never serve an LLM class and a
causal entry can never serve a full-attention op. `sb-explain` prints the
class each op took and the registry record names its source.

The score-capacity split is a **documented exception to R1**. R1 forbids
selecting an algorithm by `n` inside a shape class because a reused prefix
would then execute a different code path than the full prefill. Here the two
regimes are distinct shape classes separated by a capacity boundary (the score
matrix fits the scratch budget or it does not), not one class tuned by size;
this op has no prefix reuse and no KV cache, so no two calls must share a
path, and each resolution deterministically maps to one regime. A resolution
whose token count crosses the boundary therefore changes path once, at a fixed
point, reproducibly. No other class may use `n` this way.

No algorithm-selecting variable has to be set by hand: each has a default or an
automatic choice, and the registry records which one applied.

## Inventory

| name | scope | storage |
|---|---|---|
| `reduce.mode`, `reduce.df32-acceptance`, `reduce.double-cost-ratio`, `reduce.cpu-measurement` | device / cpu | registry + verdict file |
| `tune/attention-decode`, `tune/attention-full`, `tune/dense-gemm`, `tune/mmq-j`, `tune/spec-length` | shape class | hand-written JSON |
| `attention.class` | attention attribute key (`causal…-kvcache…-d…-g…`, plus `-cap2048`/`-over2048` for the full classes) | registry |
| `tune/gemv`, `tune/matmul_q8_tc`, `spec_verify_ratio` | shape class | per-SM JIT file |
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
