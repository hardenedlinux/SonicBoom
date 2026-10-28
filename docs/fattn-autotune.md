# boom_fattn: candidate registry, JIT and startup autotuning

`boom_fattn` is the hand-written prefill flash-attention kernel used when the
prefill attention implementation is `boom_fattn`
(`SONICBOOM_ATTN_PREFILL=boom_fattn`). Its layout and schedule parameters are
not fixed at build time: the kernel exists as a small set of candidates, and a
startup autotuner picks one per device and shape class, then reuses that pick
from a persistent cache.

This document describes the external contract: the candidates, the selection
flow, the cache, the environment switches and the report a run prints. The
measurements behind each candidate live in the project's experiment record, not
here.

## Candidates

Every variant that was built and measured keeps a row, including the ones
excluded from the default set. `id` is the stable name; `num` is the JIT
entry-point suffix and the numeric form accepted by the switches.

| num | id | KSTRIDE | BKP | softmax warps | schedule | default | note |
|---|---|---|---|---|---|---|---|
| 1 | `k8-v0-s2-pipe` | +8 | 16 | 2 (8 warps) | QK/PV pipeline | yes | reference: K padding, phase-1b row-half split, unpadded V, V triple buffered |
| 2 | `k8-v0-s2-seq` | +8 | 16 | 2 (8 warps) | sequential | yes | same layout without the pipeline |
| 3 | `k8-v0-s1-pipe` | +8 | 16 | 1 (4 warps) | QK/PV pipeline | yes | row half not split |
| 4 | `k0-v0-s2-pipe` | none | 16 | 2 (8 warps) | QK/PV pipeline | yes | K rows unpadded (bank conflicts on every K fragment load) |
| 5 | `k8-v8-s2-seq` | +8 | 24 | 2 (8 warps) | sequential | yes | V rows padded; consumes the whole sm_8.x shared-memory opt-in |
| 6 | `k8-v8-s2-pipe` | +8 | 24 | 2 (8 warps) | QK/PV pipeline | yes | legal on devices with a larger shared-memory budget; eliminated by the resource check on sm_8.x |
| 7 | `k8-v4-s2-seq` | +8 | 20 | 2 (8 warps) | sequential | no | forces 8-byte `cp.async.ca`; measured slower, kept as the experiment record and reachable only by forcing it |

`BR = BK = 16`, `NW = 4` for every candidate; the tile geometry itself is not a
tuning axis. Candidates 1–6 form the default set; the resource check removes
the ones the device cannot run before anything is compiled or timed.

## Selection flow

For each selection key (the device, software and workload fingerprints below)
the first call resolves once and the result is memoized for the process:

1. **Cache lookup.** A record for this key that matches a known candidate is
   used as is (`source=cache`). Nothing is re-measured on a cache hit, but the
   record is *not* trusted blindly: the candidate still passes the resource
   and register legality checks and is resolved (AOT image or NVRTC) before it
   is launched. A record whose candidate no longer runs is reported as
   `cache_status=invalidated` and the class is measured again.
2. **Probe.** The reference candidate (1) runs once to produce the output the
   correctness gate compares against and an initial cost estimate.
3. **Legality.** A candidate that needs `cp.async` below compute capability
   8.0, whose shared-memory mirror exceeds
   `cudaDevAttrMaxSharedMemoryPerBlockOptin`, whose registers per block exceed
   the device's register file, or whose K/V row lengths are not a whole number
   of `cp.async` transfers is eliminated without being compiled or run
   (`eliminated`).
4. **Compilation.** The AOT image is used when this device's cubin has one;
   otherwise the candidate is compiled with NVRTC (`jit`) — see below. A
   candidate that fails to compile is skipped (`compile-failed`) and is not
   retried in the same process.
5. **Correctness.** The candidate's output is compared with the reference:
   no non-finite element and relative L2 within `1e-2`
   (`correctness-failed` otherwise).
6. **Timing.** One untimed warm-up launch, then `reps` samples; a sample is
   `iters` back-to-back launches sized to reach about 0.5 ms, so the number is
   not dominated by launch latency at short context. The initial `iters` comes
   from the probe. The reference and every challenger are measured with the
   same shapes, warm-up, stream and sample count.
7. **Selection.** A challenger replaces the reference only when its median
   beats the reference median by at least 2 %. A challenger that is ahead but
   inside the margin gets two more samples on both sides before the decision.
   The margin is what keeps a noise-level difference from flipping the pick.
8. **Persistence.** The winner is written to the per-device cache and used for
   the rest of the process (`source=measured`).
9. **Fallback.** Any failure — the class cannot be tuned, a cache record names
   a candidate that no longer runs — selects the reference geometry and says so
   (`source=fallback`). The fallback is not an exemption from the checks: the
   reference still passes the resource, compile and register legality checks,
   so if it cannot run here nothing runs (`source=error`) and the caller's own
   path is used.

The initial cost is bounded by a process-wide budget (default 4000 ms). The
budget limits how many candidates are timed and how many repetitions each
gets; it never changes the shapes, the correctness gate or the margin.

Tuning never happens inside CUDA-graph capture (events and allocations cannot
be enqueued there): a captured call uses the reference and leaves the class
unresolved, so the first non-captured call still tunes.

## The one-time JIT

The AOT build contains kernels for the architectures it was compiled for. On a
device without an AOT image, a candidate is compiled from the same device
source the AOT kernel is built from, with the same parameters, using NVRTC.
The PTX is cached and reused:

```
$XDG_CACHE_HOME/sonicboom/kernels/fattn-<hash>-sm<NN>.ptx
                                    (default: ~/.cache/sonicboom/kernels/)
```

The hash covers the complete source text — which embeds the layout and
schedule parameters — every NVRTC option the compile is invoked with, and the
NVRTC version, and the file name also carries the device architecture. A kernel
change, an option change, a compiler update or a different device therefore
misses instead of loading a stale image; the options are built once and used
both for the compile and for the hash, so the two cannot drift apart. Files are
written through a unique temporary name and renamed, so concurrent processes
see either the old file or the complete new one.

`SONICBOOM_FATTN_JIT=off` refuses the JIT path (a candidate with no AOT image
is then unavailable and the class falls back); `force` compiles even when an
AOT image exists (the way the JIT path is exercised on a device that has one).
JIT and AOT kernels are numerically identical: the JIT path is selected per
device, not per shape, and the correctness gate runs on it like on any other
candidate.

## Selection keys: fingerprints

A selection is valid only for the device it was measured on, the software that
produced and launched the kernel, and the workload it was measured for. Those
are three independent facts, and each changes for a different reason, so the
key has three components:

```
f<fingerprint version>/<capability>/<software>/<workload>
```

Each component contains only fields that can change **candidate legality**,
**binary compatibility**, or the **performance ordering**. A field that changes
none of the three is not in the key, because every extra field splits the cache
into classes that would be measured again for nothing.

### Capability (probed from the device)

| field | role |
|---|---|
| compute capability | legality (`cp.async` needs ≥ 8.0) and the compile target |
| shared memory per block (opt-in) | legality: a candidate's shared-memory mirror must fit |
| registers per multiprocessor | legality: registers × block must fit |
| max threads per block | legality: the block size must fit |
| max threads per multiprocessor | ordering (occupancy) |
| shared memory per multiprocessor | ordering (blocks resident per SM) |
| SM count | ordering (parallelism) |

Example: `cc86-sm20-shm101376:102400-reg65536-thr1024:1536`.

These values are read with `cudaGetDeviceProperties` (CUDA Runtime). The GPU
**product name is never used** for a decision: two cards with the same name can
differ in SM count, shared memory or clocks, and a modified card can report any
name. The name appears in the report for a human reader only.

Compute capability is the instruction-capability field: the only ISA feature
this kernel requires is `cp.async` (8.0+), and the cubin/PTX target is derived
from it. If the device cannot be queried at all, the tuner does not guess a
capability: no candidate is run, the report says `source=error` with
`capability=unavailable` and `cache_status=unavailable`, and the caller's own
path is used. The same rule applies per candidate: a kernel whose register
count cannot be read is eliminated, not launched unchecked.

Deliberately **not** in the key: the maximum clock rate (a nominal figure that
changes with thermals and power limits, so it is neither a stable identity nor
the clock a measurement actually saw — keying on it would fragment the cache
and still be wrong), L2 size and memory bandwidth (the candidates differ in
shared-memory layout and schedule, not in bytes moved; per-call time on this
kernel is long-scoreboard/issue bound with DRAM at ~1.4 %, so those properties
do not reorder the candidates), PCIe topology and power limits. If a future
candidate becomes bandwidth-sensitive, that is the moment to reconsider, not
before.

### Software (probed from the runtime and the build)

| field | role |
|---|---|
| driver CUDA version (`cudaDriverGetVersion`) | the driver JIT-compiles the PTX; ordering and PTX-ISA compatibility |
| runtime CUDA version (`cudaRuntimeGetVersion`) | the toolkit the library was built with; AOT codegen identity |
| compile path (`aot` / `jit`) | which code object actually runs |
| NVRTC version (JIT only) | which compiler produced the kernel; compatibility and ordering |
| device-source hash | the kernel body; a code change is a different kernel |
| candidate registry version | launch parameters (pads, warps, schedule) |

Example: `drv13.2-rt12.0-aot-src36663b84de839790-reg1`.

### Workload (host-side, bucketized)

| field | role |
|---|---|
| head dim | legality (shared memory, tile fit) and ordering |
| GQA | shared-memory layout and work per block |
| KV dtype | the staging path |
| mask kind (`swa` / `global`) | how many KV rows a query reads |
| KV span bucket | power-of-two bucket of `slots + queries` (global) or `min(slots, window)` (sliding window) |

Example: `hd512-gqa4-f16-global-kv2048`.

The query count is not a separate field: the span bound is the coarsest
host-side measure of how much work one row does, and it is bucketed to a power
of two so that neighbouring context lengths share a class. Run rule R1 (one
kernel per shape class) is preserved in the sense that matters: the same query
row is computed by the same candidate within a class.

The span **can** differ between two chunks of one prefill. That is safe here
only because every selectable candidate is **bit-identical** to the reference:
all in-set candidates measured `rel_l2` exactly 0 against the reference on both
head dims (d512 global and d256 sliding-window), so two chunks that resolve to
different candidates still produce the same bytes as an unchunked call. The
`fattn_autotune_report` test re-checks this byte-for-byte. **A future candidate
that is not bit-identical must not be added to the tuning set while the span
bucket is part of the key** (or the bucket must be dropped).

### Two caches

- **Selection cache** — one flock-protected file per SM version,
  `$XDG_CACHE_HOME/sonicboom/autotune/sm_<NN>.bin`, versioned (currently
  **schema 6**, shared with the planner's tuner). Version 6 widened the shape
  field from 48 to 192 bytes and the record from 112 to 256 bytes so a
  fingerprint key fits; a version-5 file is reported as `version-mismatch` and
  is never read as the new layout.
- **PTX cache** — `$XDG_CACHE_HOME/sonicboom/kernels/fattn-<hash>-sm<NN>.ptx`.
  The hash covers the complete source text, **every NVRTC compile option**
  (architecture, language standard, include paths) and the **NVRTC version**;
  the file name also names the architecture. A cache hit is therefore only
  possible when the file is the artifact this build would produce for this
  device (binary compatibility), never merely a file with a similar name.

### Invalidation

| change | effect |
|---|---|
| shared cache schema bump (`kVersion`) | every record is dropped; the first store rewrites the file |
| fingerprint schema bump (`kFingerprintVersion`) | every selection key changes, so old records can never match |
| candidate added/removed/retuned (`kRegistryVersion`) | the software component changes; old records cannot match |
| kernel source change | the software component and the PTX hash change |
| driver/runtime/NVRTC change | the software component changes (retune) |
| different device capabilities/resources | the capability component changes |
| a different context-length bucket | the workload component changes |
| a record naming an unknown candidate, or one the device cannot run | `cache_status=invalidated`, the class is measured again |
| an older schema file | `cache_status=version-mismatch`, nothing is read, the next store rewrites the file |

Devices with different product names but identical capability and software
components share a key, so they reuse each other's selections. That is
deliberate — the record is a measurement of those exact capabilities — but the
reuse is only as good as the components: a property that affects this kernel
and is not in the table above would be missed. Devices with the same product
name but different resources get different keys automatically.

`SONICBOOM_FATTN_FP_SIMULATE` exists to test this mechanism without faking
hardware: it replaces a component in the key and the report, and nothing else.
Legality, compilation and timing always use the real device and the real build,
so a simulated key can never make an illegal kernel run.

## Report

Each class the tuner handles prints, to stderr, one human line and one JSON
line, on cache hits as well as after a fresh measurement:

```
[SonicBoom autotune] GPU="..." CC=8.6 SM=20 smem/block=101376 smem/mp=102400 \
    regs/mp=65536 maxthr/block=1024 driver_cuda=13.2 runtime_cuda=12.0 ... \
    source=cache selected=k8-v0-s2-pipe id=1 BR=16 BK=16 KSTRIDE=520 BKP=16 ... \
    cache=hit cache_status=hit fingerprint=v1 capability=cc86-sm20-... \
    software=drv13.2-rt12.0-aot-src...-reg1 workload=hd512-gqa4-f16-global-kv2048
[SonicBoom autotune JSON] {"schema_version":2,"event":"fattn_selection",...}
```

`source` (also `selection.status` in the JSON) is one of:

| value | meaning |
|---|---|
| `measured` | selected in this process from timings taken in this process |
| `cache` | reused from the selection cache; the timings behind it are historical, not measured now |
| `forced` | `SONICBOOM_FATTN_CANDIDATE` named a candidate that runs here |
| `default` | autotuning disabled, not cached under `SONICBOOM_FATTN_SHOW=1`, or under graph capture |
| `fallback` | a forced or cached candidate could not run; the reference geometry is used, with a reason |
| `error` | nothing could run; no candidate is reported as selected |

The JSON is produced by a serializer, always parses on its own line, and
carries: the device (name, compute capability, SM count, shared-memory,
register and thread limits, driver and runtime CUDA versions), the
`fingerprint` object (schema version, the full selection key, its capability,
software and workload components, and which components a simulate override
replaced), the selected candidate and its geometry (candidate, id, BR, BK,
KSTRIDE, BKP, softmax warps, schedule, block, compile path, shared memory,
registers, spill bytes), the basis of the selection, the tuning summary
(candidates considered, resource eliminations, compile failures, correctness
failures, candidates timed, warm-up launches, repetitions, launches per
sample, budget, margin, reference and probe medians, and per-candidate
results), the workload (head dim, GQA, query count, head counts, sliding
window, dtype, capture state), and the cache status (op, shape class, file
version, hit, status, reason). Values that do not apply or were not measured
are `null`/`unavailable`; a cache hit reports `reps = 0` and
`reference_median_ms = null` rather than a historical number dressed up as a
measurement. `device.name` is informational (`device.name_is_informational`
is always true) and is the only place the product name appears. No prompt
text, model input, path or user name appears in either line.

`cache.status` (and `cache_status=` on the human line) is one of:

| status | meaning |
|---|---|
| `hit` | a record for this key was found and its candidate is legal and runnable here |
| `miss` | no record for this key; the class was measured |
| `invalidated` | a record was found but its candidate can no longer run (unknown candidate, resource check, compile failure); the class was measured again |
| `version-mismatch` | the cache file is an older schema; nothing was read and the next store rewrites it |
| `unreadable` | the cache file could not be read; treated as a miss |
| `disabled` | `SONICBOOM_FATTN_AUTOTUNE=0` |
| `bypassed` | `force`, CUDA-graph capture, or `SHOW`: the cache was not consulted for a decision |
| `unavailable` | the selection key did not fit the record's shape field, so the class was measured but cannot be persisted |

`cache.reason` explains the status in one sentence. `source` is independent of
the cache status: a miss that was then measured reports `status=miss` with
`source=measured`; a record whose candidate fails the legality or compile check
reports `status=invalidated` (with the failing check in the reason) and the
class is measured again (`source=measured`), or falls back to the reference
(`source=fallback`) if the fresh measurement cannot run the reference either.

`SONICBOOM_FATTN_REPORT=0` silences both lines; `SONICBOOM_FATTN_REPORT_FILE=<path>`
appends the JSON line followed by the human line for each class to a file.

## Switches

| variable | values | effect |
|---|---|---|
| `SONICBOOM_FATTN_AUTOTUNE` | unset/`1`, `0`, `force` | on (default), off (reference geometry, `source=default`), retune ignoring the cache |
| `SONICBOOM_AUTOTUNE` | `0`, `force` | same, both tuners (`SONICBOOM_FATTN_AUTOTUNE` wins) |
| `SONICBOOM_FEATURE_DISABLE` | `autotune` | disables both tuners |
| `SONICBOOM_FATTN_CANDIDATE` | id or num | force one candidate; it still passes the legality and compile checks, and falls back with a reason if it cannot run |
| `SONICBOOM_FATTN_CANDIDATES` | `1,3,5` or ids | restrict the set the tuner times (the reference is always included) |
| `SONICBOOM_FATTN_AUTOTUNE_BUDGET_MS` | integer | one-time cost budget for a fresh tuning (default 4000; `<= 0` unlimited) |
| `SONICBOOM_FATTN_JIT` | `auto` (default), `off`, `force` | NVRTC fallback when the device has no AOT image, disabled, or always |
| `SONICBOOM_FATTN_REPORT` | `1` (default), `0` | print the report lines |
| `SONICBOOM_FATTN_REPORT_FILE` | path | append the report to a file |
| `SONICBOOM_FATTN_SHOW` | `1` | print the cached selection without tuning (reference when none is cached) |
| `SONICBOOM_FATTN_FP_SIMULATE` | `capability=..;software=..;workload=..` | replace one or more fingerprint components in the selection key and the report; legality, compilation and timing still use the real device and build |

## Kernel mode switches

`boom_fattn` has one kernel family per head width. The default geometry of each
is the query-resident variant validated on the RTX 3050; the earlier kernel of
each width is kept as an explicit fallback, not as an automatic choice. These
switches select the variant; the autotuner above still selects among the
layout/schedule candidates *of* the selected variant.

| variable | values | effect |
|---|---|---|
| `SONICBOOM_BOOM_FATTN_QREG` | `1` (default), `0`, `2`, `3` | d512, f16 accumulator: `1` query-resident, QK^T split over the four dim slices, BR16/BK16, 512-thread block. `0` the previous kernel (BR32/BK16, 1024 threads, phase 1 on the slice-0 warps, Q re-read per key tile) — the explicit fallback. `2` split without Q residency and `3` BR16/BK16 without the split are diagnostic controls, not alternatives to prefer. |
| `SONICBOOM_BOOM_FATTN_D256` | `1` (default), `0`, `2`, `3`, `4` | d256 (the sliding-window head): `1` the d512 mode-1 schedule at this width (BR16/BK16, 512 threads). `0` the previous kernel (BR32/BK32, 1024 threads) — the explicit fallback. `2`, `3`, `4` are controls (split only, geometry only, 8-slice split). |
| `SONICBOOM_BOOM_FATTN_ACC` | unset (f16), `f32` | `f32` keeps the earlier fp32-accumulator kernel for either width, regardless of the QREG/D256 setting; the per-width mode switches do not apply to it. |

Rules that hold for this table:

- **Mode 1 is the tuner's reference.** The default path stages K/V, then calls
  the tuner, which selects among candidates of the mode-1 schedule. Setting the
  variable to `0` bypasses the tuner for that width and launches the previous
  kernel directly (its geometry appears in the trace, not in the autotune
  report).
- **An unrecognized value means the default** (`1`), and the variables are read
  per call, so one process can benchmark both variants.
- **Mode 1 fixes the schedule; the tuner only varies its layout.** Query tile,
  key tile, block size, Q residency and chunking are the validated ones. The
  candidates the tuner selects among are layout/schedule variants inside that
  schedule (K-tile padding, phase-1 split width, V stride, pipeline depth) and
  are required to be bit-identical to the reference, which the correctness
  gates check with a byte comparison. The QREG/D256 switches change no
  tolerance or numerical rule.
- **Backend selection is separate.** `boom_fattn` is opt-in
  (`SONICBOOM_ATTN_IMPL=boom_fattn` or `SONICBOOM_ATTN_PREFILL=boom_fattn`;
  `auto` never selects it) and falls through to the native path for shapes
  without an instantiation. The switches above only choose the kernel once
  `boom_fattn` has been selected; they do not make it the default backend.

## Community diagnostics

When reporting a `boom_fattn` run on a device that is not validated yet,
include the two report lines from a *fresh* run
(`SONICBOOM_FATTN_AUTOTUNE=force`) and, separately, the line from an ordinary
run. The first shows what the tuner measured on that device; the second shows
what the cache then serves. They are labelled `measured` and `cache`, so a
reader can tell a measurement made on that machine from a reused record, and
`fallback` records name the reason the preferred candidates could not run.

## Validation status

The tuner, the candidates and their ordering have been validated on one
device: the RTX 3050 8 GB (sm_86, 20 SMs, 101,376 B shared memory per block).
On that device the flow runs at startup and selects the reference candidate
for the prefill classes it was tuned for; a one-query chunk is a separate
class and measured candidate 5 (`k8-v8-s2-seq`) fastest there.
The set is written so that other Ampere-and-later parts are handled by the
same flow — resource checks before compilation, correctness gate before
timing, incumbent protection after it — but no cross-device performance
validation has been done. Reports from other devices are what that step
needs.

What the RTX 3050 run does prove, and what it does not:

- **Probed, not assumed.** Every capability field in the key was read from
  `cudaGetDeviceProperties` on that device, and the report's capability
  component matched those values. The selection cache was cleared, the class
  was measured, and the verified best combination (K-padding, phase-1b 8-warp
  split, BKP=16, QK/PV pipeline) was selected again; a second launch reported
  `cache_status=hit` for the same key. A record written under schema 5 was
  reported as `version-mismatch` and never used.
- **Simulated keys are key-only.** `SONICBOOM_FATTN_FP_SIMULATE` was used to
  show that each component isolates the cache: a changed component produces a
  different key, a fresh measurement, and no false hit, while the legality
  checks still ran against the real device (the sm_86-illegal candidate stayed
  eliminated). This proves the key mechanism, not behaviour on another device.
- **Unverified assumptions awaiting community reports.** (a) That the seven
  capability fields are *sufficient* to separate performance-relevant device
  differences on parts this project has not seen. (b) That two devices with the
  same capability and software components but different product names rank the
  candidates identically. (c) That the driver/runtime/NVRTC version strings are
  the right granularity for binary compatibility (a driver update that does not
  change the reported version is invisible). A report that shows the same key
  on two devices with different behaviour, or a wrong bit on a modified card,
  is the evidence needed to add or drop a field.

The candidate set order (candidate 1 as reference) is a measurement on this
device; the resource check, not the name, is what keeps other parts legal.
