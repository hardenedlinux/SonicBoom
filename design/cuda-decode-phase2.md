# CUDA decode — Phase 2 (kernel) report

**Date: 2026-10-11.** Follows `cuda-decode-phase1.md`. This phase measures the
missing data (2.0) and starts fixing the kernels. Production code changes are
per-sub-phase commits (2.1 rms, 2.2 quant gemv, 2.3 dense gemv); each records
before/after tok/s.

## 2.0 — data completion

### (a) tok/s without nsys — NOT 6.20, it is 4.33

64-step steady-state decode, 3 runs, median: **4.33 tok/s**
(4.3444 / 4.31767 / 4.33359). The `benchmark.md` figure 6.20 tok/s is
clock/thermal variance on the same machine (no production code changed between
the two measurements; `git log` shows only doc commits after the 6.20 run).
The token stream is deterministic and matches the llama.cpp golden
`236761 107 236769 236776`.

### (b) GPU kernel time vs wall time

- Total GPU kernel time over the profiled run = **20.451 s** (nsys gpukernsum;
  hardware timestamps, unaffected by nsys CPU-side overhead).
- CUDA decode steps in that run = 104 (from per-token gemv instance counts).
  → **kernel time ≈ 196.6 ms/token**.
- Steady-state wall (no nsys) = **231 ms/token** (4.33 tok/s).
- **Ratio = 0.85.** The GPU is busy ~85% of wall during decode; ~15%
  (~34 ms/token) is non-kernel — launch gaps, synchronous-memcpy stalls, and the
  CPU-only operators (Embedding, Softcap, host argmax). This is the Phase 3
  budget, and it is exactly where the `cudaMemcpy` synchronous-API overhead
  (correction #3) lives.

### (c) cudaMemcpy / cudaMemset: one-time vs per-token

`gpumemtimesum` / `gpumemsizesum`:

| op | calls | bytes | time | reading |
|---|---|---|---|---|
| HtoD | 3,384 | 2,470 MB | 450 ms | one-time weight upload (max 288 MB = token_embd q3_K) + per-token activations (med 10 KB) |
| memset | 55,466 | 562 MB | 68 ms | y-zero-fill before every `_base` gemv (per-token, not one-time) |
| DtoH | 1,469 | 137 MB | 22 ms | per-token logits (1 MB) + small scalars |
| DtoD | 16,558 | 91 MB | 29 ms | KV-cache device copies |

**Per-token recurring transfers** (every decode step):
- H2D: `embd` [2560 f32 = 10 KB] + per-layer token embed / ple row [~43 KB]
  (the CPU `Embedding` output).
- D2H: `logits` [262144 f32 = 1 MB] (to CPU `Softcap` → host argmax).
- DtoD: KV-cache writes (attention).
- memset: ~533 y-zero-fills/token (one per `_base` gemv) — disappears when 2.2
  replaces `_base` with direct-write kernels.

### (d) lm_head path

`lm_head` = `token_embd` @ hidden, and `token_embd` is **q3_K** [2560, 262144]
→ 10 blocks/row → `gemv_q3_K_base`. It is stored **packed** q3_K (~288 MB on
device), NOT dequantized to f32. `gemv_dense_f32` = `inp_gate` [2560→256] +
`proj` [256→2560] f32; `gemv_bf16` = `per_layer_model_proj` [2560→10752].

### (e) SONICBOOM_GEMV_Q8_1=1

The already-ported q8_1 `__dp4a` path: gemv_bench showed FFN/Q4_K 2.5–9.3× faster
but Q3_K attention shapes 1.5–1.6× slower; per-token gemv 122.5 → 65.67 ms.
End-to-end **7.09 tok/s (1.64× vs 4.33)**, token stream identical to golden.
(The "attention 1.5–1.6× slower" was the §2.2b gemv_bench artifact; with the
fixed tool the attention shapes are 1.5–2.3× *faster* — see §2.2b.)
**Caveat:** the prefill-cuda test fails under q8_1 (batched prefill uses fp32
matmul while the per-token ref uses q8_1 — a test-consistency artifact, not a
decode bug). This needs resolving before q8_1 is enabled by default.

### (f) kernel resource usage (cuobjdump — no perf-counter permission needed)

`cuobjdump --dump-resource-usage libsonicboom.so` gives REG/LOCAL/STACK per
kernel without ncu:

| kernel | REG | STACK | LOCAL |
|---|---|---|---|
| gemv_q3_K_base | **96** | 0 | 0 |
| gemv_q4_K_base | 56 | 0 | 0 |
| gemv_q5_K_base | 96 | 0 | 0 |
| gemv_q3_K_coop | 40 | 16 | 0 |
| gemv_q4_K_coop | 38 | 0 | 0 |
| gemv_q5_K_coop | 40 | 0 | 0 |
| gemv_q3_K_q8_1 / q4_K_q8_1 | 40 / 35 | 0 | 0 |
| rms_sum_sq_kernel | 16 | 0 | 0 (post-2.1) |

**Corrections to phase1.md:**
- The q3_K_base slowdown is **register-bound** (96 regs → 2 blocks/SM → ~25%
  occupancy), **not** a local-memory spill. `unpack_q3_K_scales`' `memcpy` of the
  16×int8 scale array fits in registers (LOCAL:0). phase1.md §2 anomaly #2's
  "local-memory spill" hypothesis is **refuted**.
- `gemv_q4_K_coop` (ffn_down, 28.2%) is clean (38 regs, no spill) yet still only
  12.2 GB/s at 2560 rows — the coop mapping itself is structurally bad, not a
  resource problem. It cannot be a reference template (correction #2 confirmed).

**ncu BLOCKED:** `ncu` needs GPU perf-counter permission
(`ERR_NVGPUCTRPERM`); `sudo` requires a password. The four ncu-only metrics
(achieved occupancy, DRAM %, L1/L2 sectors-per-request, warp stall reasons) are
not obtainable without either `sudo ncu` or
`NVreg_RestrictProfilingToAdminUsers=0`. REG/LOCAL (above) substitutes for
two of the six requested metrics.

## 2.1 — rms_sum_sq_kernel parallel reduction (DONE)

Single-thread `<<<1,1>>>` sequential double sum (127 µs avg, 15% of kernel time)
→ 256-thread warp-shuffle + shared-memory reduction (`<<<1,256>>>`). Accumulation
stays in double with identical per-term `double(v*v)` semantics; only the add
order changes, which is far below the fp32 rounding `rms_inv_kernel` applies —
so the downstream token stream is **bit-identical** (verified: 64-token CUDA
stream unchanged before/after).

- New kernel: REG 16, SHARED 256 B, LOCAL 0.
- **Before/after tok/s (64-step decode, 3-run median): 4.33 → 5.07 (1.17×).**
  The kernel collapsed from ~127 µs single-thread to launch-latency-bound
  (~5 µs, well under the 5% acceptance), removing ~28 ms/token of GPU time
  (231 launches × ~122 µs saved).
- ctest 48/48 pass; CUDA token stream bit-identical before/after.

## 2.2 — quantized gemv (direction: q8_1 全面化)

### 2.2a — prefill/decode quantization consistency (DONE, f134000)

The q8_1 gate `SONICBOOM_GEMV_Q8_1` applied to `matvec_f32_dev` (n=1, decode)
but not to `matmul_f32_dev` (n>1, batched prefill), so prefill ran fp32 while
decode ran q8_1 on the *same* weights. That is a constructive inconsistency, not
a numeric bug: the two paths fed the KV cache from different activation
quantizations, so "prefill == per-token spine" diverged at token ~52. Fix:
`matmul_f32_dev`'s per-column loop now dispatches through the same
`g_q8_enabled()` gate as `matvec_f32_dev`, so prefill and decode use the same
activation quantization. Rule going forward: one weight set ⇒ one activation
quantization for both prefill and decode.

### 2.2b — Q3_K q8_1 small-shape "optimization" (verdict: no kernel rewrite needed)

**The Q3_K q8_1 attention kernels are already fast; the 3.2/2.5 GB/s that
gemv_bench originally reported for attn_q/attn_k (§2.0e) was a measurement
artifact, not a kernel defect. The tool has since been fixed (see below) and the
corrected q8_1 table is at the end of this section.**

Kernel-level effective bandwidth (nsys gpukernsum — hardware timestamps, the
authoritative kernel-only number; the gemv_bench wall clock includes quantize +
launch, see below):

| kernel (shape) | grid | kernel µs | packed bytes | GB/s |
|---|---|---|---|---|
| `gemv_q3_K_q8_1` (attn_k, 512 rows) | 1×64 | 7.5 | 563 KB | **75** |
| `gemv_q3_K_q8_1` (attn_q, 2048 rows) | 1×256 | 23.1 | 2.25 MB | **97.5** |
| `gemv_q4_K_q8_1` (attn_v, 512 rows) | 1×64 | 6.0 | 737 KB | 123 |

**Where 97.5 GB/s comes from.** It is the *pure* `gemv_q3_K_q8_1` kernel time
for attn_q (2048 rows, 2.25 MB packed = 20480 blocks × 110 B) read from
`nsys stats --report gpukernsum` hardware timestamps in the real decode run:
23.1 µs → 2.25 MB / 23.1 µs = 97.5 GB/s. It **excludes** the `quantize_q8_1`
launch (~6 µs, a separate kernel) and the launch/handshake overhead, so it is
the gemv kernel's own bandwidth, not the end-to-end `matvec_f32_dev` rate. The
full `matvec_f32_dev` for attn_q measures 34.1 µs (66.1 GB/s) — quantize + gemv
+ launch — which is the number to compare against llama.cpp's per-op rate; the
97.5 GB/s is the kernel-only figure used to argue the kernel itself is fast
(above the 80 GB/s acceptance target).

**Why the original gemv_bench misreported small shapes (root cause).** Two
interacting bugs in the *tool*, neither in the kernel:

1. **Idle-clock wake-up + single-call adaptive iteration count.** The original
   `run()` warmed up with 5 calls, then timed *one* call to size an adaptive
   `N = 20000/one_us`. A single small kernel launched right after a sync runs at
   the GPU's idle clock (~210 MHz P8 vs ~2.1 GHz boost) and pays SM power-gating
   wake-up (~100–500 µs), so `one_us` was ~10–30× the true kernel time; N shrank
   accordingly and the batch stayed at the low clock. This is what inflated
   attn_q to 505.8 µs, attn_k to 159.2 µs, and attn_v-Q5_K to 367.1 µs in
   phase1 §2 (the fp32 base is 24–78 µs at those shapes, not 155–500 µs).
2. **Stale device-weight cache (host-pointer reuse).** `upload_weight` caches the
   device copy by (host data pointer, byte size). The original `run()` freed each
   shape's weight vector at return, so the next shape's `malloc` reused the
   address when sizes matched (ffn_gate and ffn_up are both Q3_K [2560,10240] =
   11.26 MB). A cache hit then served the *previous* shape's device copy — which
   also skips the H2D upload, shifting the benchmark's device-memory/clock state
   so the small attention shapes that follow measured ~2× their true time
   (attn_k 32 µs) and the Q5_K reference reported an impossible 320 GB/s (above
   the 224 GB/s DRAM ceiling).

**Fix** (committed to `tools/baseline/gemv_bench.cpp`): (a) fixed 2000-call
warmup + 2000-call timed batch with no sync in between — 2000 calls holds the
clock at boost even for the ~15 µs kernels and is short enough that the ~1 ms
fp32 kernels don't throttle; (b) every built weight is kept alive for the whole
run (a `keepalive` vector) so host pointers stay unique and the device cache
never serves stale data. Corrected gemv_bench (q8_1 default):

| tensor | type | µs | GB/s |
|---|---|---|---|
| ffn_gate | Q3_K | 126.8 | 88.9 |
| ffn_up | Q3_K | 127.7 | 88.2 |
| ffn_down | Q4_K | 124.8 | 118.1 |
| attn_q | Q3_K | 34.1 | 66.1 |
| attn_k | Q3_K | 15.8 | 35.7 |
| attn_v | Q4_K | 14.5 | 50.9 |
| attn_output | Q4_K | 31.2 | 94.6 |
| ffn_down (blk 0-1) | Q5_K | 142.5 | 126.5 |
| attn_v (blk 0-1) | Q5_K | 15.1 | 59.8 |

The small attention shapes are **not** regressed: q8_1 attn_k is 15.8 µs vs the
fp32 base 24.4 µs (1.5× faster), attn_q 34.1 vs 78.3 µs (2.3×). The fp32 base
attention shapes are 24–78 µs, not the ~155–500 µs the artifact implied, so the
earlier "q8_1 is ~20× fp32 base" claim was itself the artifact. The real q8_1
speedup over fp32 is ~1.5× (attn_k) to 8× (ffn_down).

**Occupancy + registers** (cuobjdump; `ncu` still blocked by `ERR_NVGPUCTRPERM`):

| kernel | REG | LOCAL | max blocks/SM (register-limited) |
|---|---|---|---|
| `gemv_q3_K_q8_1` | 40 | 0 | 6 (40×256=10240 regs; 65536/10240) |
| `gemv_q4_K_q8_1` | 35 | 0 | 7 |
| `gemv_q5_K_q8_1` | 38 | 0 | 6 |
| `quantize_q8_1` | 49 | 0 | 5 |

block(32,8) = 256 threads. attn_k (64 blocks → 3.2 blocks/SM) = 40% occupancy;
attn_q/ffn (≥256 blocks → 6 blocks/SM) = 75%. The kernel already fills all 20
SMs (64/256 blocks > 20) and is latency/L1-bound (phase1 ncu), not
occupancy-bound — so "one row per multiple warps / more rows per block" would
not move the needle. The remaining per-call cost for small shapes is the
`quantize_q8_1` launch (6 µs kernel + ~5 µs launch), addressed by
quantize-once-per-step reuse (2.2d), not by changing the gemv launch config.

### 2.2c — q8_1 as default (DONE)

The q8_1 mmvq path is now the default decode path. `g_q8_enabled()` returns
true unless `SONICBOOM_GEMV_F32=1` is set, so `matvec_f32` / `matvec_f32_dev` /
`matmul_f32` all quantize x to q8_1 and run the dp4a warp-per-row kernels by
default. The fp32 dequant-on-the-fly base/coop hybrid is **preserved** (not
deleted) behind `SONICBOOM_GEMV_F32=1` as the exact-arithmetic differential
path.

- `test_cuda_quant` (the differential test that compares the CUDA kernel against
  the scalar CPU f32 reference) now forces `SONICBOOM_GEMV_F32=1` at startup, so
  it keeps testing the exact fp32 path. The q8_1 path is approximate (int8
  activation) by design and is validated end-to-end by `test_generate`'s golden
  token stream instead.
- Full ctest **48/48** pass; `test_generate` token stream matches the llama.cpp
  golden `236761 107 236769 236776`.

**Before/after tok/s (64-step decode, spine-cuda): 4.33 → 12.04 (2.78×).**
The q8_1 default gives 12.0431 tok/s vs the fp32 default's 4.33 tok/s, with a
bit-identical continuation (the earlier explicit-gate run and this default run
both yield 236761 107 236769 236776 …).

### 2.2d — quantize-once-per-step reuse (deferred, not a dead end)

Implemented "quantize the activation once and reuse it across the gemvs that
share it" — quantize `h` once for `attn_q/k/v`, quantize `f` once for
`ffn_up/gate` — via a `quantize_x_q8_1` + `matvec_dev` device-resident API
(gated so `SONICBOOM_GEMV_F32=1` still takes the fp32 path). Token stream is
bit-identical (test_generate golden passes). Reverted for now.

**Result: wall time unchanged (12.0431 → 12.0446 tok/s, noise) — but the cause is
host-side, not kernel overlap.** Kernels on the same stream run serially
(quantize → gemv is a data-dependency chain); async launch hides CPU launch
latency only, never GPU execution time. Removing ~126 quantize kernels/token of
GPU work while wall time stays flat means the GPU was idle enough to absorb it —
the critical path is on the **host** (synchronous `cudaMemcpy` ~300×/token +
host embedding/softcap/argmax/planner), not the GPU. This is the same lever as
the coexecution finding: per-call sync / resident activations (Option A), not
more kernel work. **Deferred**: re-evaluate quantize-once-reuse after Phase 3
removes the host-side sync, when the GPU is again the bottleneck.

## 3.0 — orchestration vs kernel decision (nsys, no production code change)

`nsys profile` over a full `test_generate` run (q8_1 default,
`SONICBOOM_GEN_STEPS=64`, `SONICBOOM_SKIP_CPU=1`), `gpukernsum` /
`cudaapisum` / `gpumemtimesum` hardware timestamps. The current decode path is
the **planner spine** (`[spine-cuda]` = `generate_spine_cuda` → `SonicCudaBackend`
per-op graph), 11.89 tok/s (the `[cuda]` section is `generate_reference(Cuda)` →
`cuda_forward_resident`, the single-token *oracle* with no KV cache — a
different, simpler computation, not the decode path).

### (a) per-token wall / GPU busy / util

Wall = 84.1 ms/token (11.89 tok/s). GPU kernel time ≈ 61 ms/token → **util ≈ 73%
(>70%)**. Per-token GPU kernel time by category (planner decode):

| category | ms/token | % of kernel |
|---|---|---|
| **dense gemv** (gemv_bf16 + gemv_dense_f32) | **38.2** | **62** |
| K-quant gemv (q8_1) | 14.0 | 23 |
| lm_head (token_embd q3_K) | 2.7 | 4 |
| attention + norms + quantize + elementwise | 6.5 | 11 |

The dense gemvs are three per block, all `one-thread-per-row` (`gemv_*` at
`core/src/nn/cuda_elementwise.cu:211/225`, grid `grid_for(rows)`):

| dense gemv | shape | grid | µs | why slow |
|---|---|---|---|---|
| `gemv_bf16` | per_layer_model_proj 2560→256 | 1×1×1 | 439.9 | 256 rows = 1 block |
| `gemv_dense_f32` | inp_gate 2560→256 | 1×1×1 | 425.8 | 256 rows = 1 block |
| `gemv_dense_f32` | proj 256→2560 | 10×1×1 | 43.5 | 2560 rows = 10 blocks |

The two single-block ones (per_layer_model_proj, inp_gate — both 2560 cols × 256
rows) are ~10× the proj kernel at *equal* FLOPs, purely because 256 rows fills
only one 256-thread block (1/20 SM) and each thread runs a serial 2560-length
dot. (`per_layer_model_proj` is per-layer 2560→256; phase2 §2.0(d)'s "[2560→10752]"
was the 42-layer total, not one gemv.)

### (b) idle-gap locations (~27% of wall)

Synchronous `cudaMemcpy` host-API stalls (below), the host `Embedding` /
`Softcap` / `argmax` / planner scheduling, and a handful of single-thread
kernels (`rms_inv_kernel` `<<<1,1>>>`, already small after 2.1).

### (c) per-token cudaMemcpy / memset

`gpumemtimesum` (GPU-side) vs `cudaapisum` (host-side) split:

| op | calls | GPU time | host API time |
|---|---|---|---|
| HtoD | 8,088 | 0.445 s | — |
| DtoH | 3,917 | 0.070 s | — |
| DtoD | 65,614 | 0.115 s | — |
| memset | 129,670 | 0.107 s | — |
| **cudaMemcpy (all)** | 77,619 | **0.63 s** | **16.34 s (64.3%)** |

The host spends **16.34 s inside `cudaMemcpy` (median 9.4 µs/call)** while the
GPU does only **0.63 s of actual copy** — a ~26× host-side stall per transfer.
The 53 ms max is the one-time token_embd (288 MB) upload; the per-token
transfers are median ~1–9 µs but synchronous, so the host blocks on each.

### (d) launch count + host compute

867,076 kernel launches (7.52 s host API, 29.6%), ~3,500 launches per decode
token. **Zero** per-op `cudaDeviceSynchronize`/`cudaStreamSynchronize`;
`cudaEventSynchronize` = 67 total (only the timed warmup/timing boundaries). The
host compute (embedding + softcap + argmax + planner) is the rest of the 27%
idle gap.

### Decision

GPU util ≈ 73% **>70%** → **2.3 (dense gemv), not Phase 3.** The GPU is the
bottleneck, and the dense gemvs are 62% of its time as single-block kernels.
Phase 3 (orchestration) stays the follow-up: the ~27% idle gap is real but is
one third the size of the dense-gemv GPU cost, and 2.3's parallelization of the
three dense gemvs is the larger, orthogonal lever.

## 2.3 — dense f32/bf16 gemv (DONE)

### 2.3a — hoist per_layer_model_proj to once-per-token (graph/plan level)

The three dense gemvs are (per block, 42 blocks) `per_layer_model_proj` 2560→256,
`inp_gate` 2560→256, `proj` 256→2560. The first of these is **independent of the
block's hidden state** — it projects the *token embedding*, so all 42 layers
compute the same 2560→256 projection of the same input. Hoisted into a single
**`PerLayerInput`** op in the plan preamble: one full-width 2560→**10752** bf16
gemv + scale (1/√d) + segmented rms_norm (shared `per_layer_proj_norm` [256] over
42 heads) + combine with `per_layer_token_embd` (×√256, then ×1/√2), all folded
into one token-start pass. Each block then reads its [256] segment via a new
**`Slice`** op instead of recomputing the projection.

- `OpKind::PerLayerInput` + `OpKind::Slice` (op_kind.h/.cpp, partition.h,
  resource.cpp), emitted in `gemma4_plan.cpp` `build_gemma4_graph`, executed in
  `SonicCudaBackend::execute` (device-resident: `matvec_bf16_dev` + `scale_dev` +
  `rms_norm_heads_dev` + `layer_combine_dev`); the CPU `SonicBackend` computes the
  same arithmetic in `sonic_backend.cpp`.
- This is a plan/graph change, **not** kernel-level special-casing, and it keeps
  the CPU and CUDA spines on one shared graph (both reproduce the golden token
  stream).

### 2.3b — dense gemv kernels: vectorized + split-K warp-per-row

Rewrote the dense kernels in `nn/cuda_elementwise.cu` (`gemv_dense_f32_v`,
`gemv_bf16_v`): warp-per-row with `k_split` warps cooperating per row via
`atomicAdd` into a zeroed `y`, float4/uint4 (≥128-bit) loads, `__shfl_down`
warp reduction. The few-row shapes (inp_gate 256 rows) now fill the SMs via
split-K instead of one serial thread-block. `matvec_f32_dev` / `matvec_bf16_dev`
dispatch to the vectorized kernel when `cols % 4 == 0` (f32) / `% 8 == 0` (bf16)
and fall back to the scalar kernel otherwise.

### 2.3c — device-weight cache composite key (root-cause OOM fix)

`test_generate` died at step 42 with `cuda=out of memory`: `upload_bytes` /
`upload_weight` keyed the device cache by host pointer **only**, with a bytes
field *inside* the value. The single-token oracle uploads per-block slices of
`per_layer_model_proj` (block 0 = 1310720 B at the base pointer), while the spine's
`PerLayerInput` uploads the full tensor (55050240 B) at the **same base pointer**;
`find(host)` returned the slice entry, the bytes check missed, and `emplace(host,
…)` was a no-op — so the 55 MB upload was re-allocated every step (never cached)
and leaked 54 MiB/step. Fixed by a composite `(host pointer, byte size)` key
(`CacheKey`/`CacheKeyHash`) in both `nn::cuda` and `quant::cuda`.

### 2.3d — gemv_bench dense shapes + cache-staleness in the tool/test

`tools/baseline/gemv_bench.cpp` gained the three dense shapes and the fixed
2000-call warmup/timed methodology (§2.2b). The same host-pointer-reuse
staleness that bit gemv_bench also bit `tests/core/test_cuda_quant.cpp` (its
batched n=1/n=8 calls build identical [64,1024] weights whose freed buffers
reuse one address at the same size); fixed with a `keepalive` vector so every
weight's host pointer stays unique (matching gemv_bench). Pre-existing, not a 2.3
regression.

### Results (RTX 3050, sm_86, q8_1 default, 64-token decode)

**dense gemv microbenchmark** (gemv_bench, device-resident, GB/s = weight bytes / time):

| shape | dtype | µs | GB/s | target |
|---|---|---|---|---|
| per_layer_model_proj [2560→10752] | bf16 | 359.0 | **153.3** | ≥150 |
| inp_gate [2560→256] | f32 | 20.6 | **127.3** | ≥100 |
| proj [256→2560] | f32 | 20.5 | **127.7** | ≥100 |

All three meet target (no ncu needed). Per-token dense gemv dropped **38.2 ms →
~2.1 ms** (18×): 359 µs (×1) + 20.6 µs (×42) + 20.5 µs (×42).

**End-to-end `[spine-cuda]` = 23.9 tok/s** (vs 11.89 tok/s at §3.0, 2.0×;
target ≈20). `ctest` 48/48 pass.

### Divergence vs golden (accumulation order)

The split-K `atomicAdd` and the full-width bf16 gemv reorder the accumulation, so
the CUDA spine's logits drift from the CPU spine by ~1e-3 relative and, at a
near-tie, flip the argmax. First divergence (64-token run): **token 44
(1-based)** — CPU top-1 `236764` (25.397) over top-2 `779` (24.194, gap 1.20);
CUDA swaps them (`779` > `236764`). The two tokens are the same top-2 in both,
just reordered — the anticipated "tiny logit diff" signature, not a real error.

**Caveat:** `atomicAdd` in the `k_split > 1` path is non-deterministic, so the
*exact* flip position shifts a few tokens between runs (~token 38–44). If
reproducible tokens are required, the split-K reduction should be made
deterministic (per-(row,chunk) partial buffer + fixed-order sum) — noted, not
done here.

## 3.0 re-measure — post-2.3 (DONE)

Same full-`test_generate` nsys run as §3.0 (`SONICBOOM_GEN_STEPS=64`,
`SONICBOOM_SKIP_CPU=1`, q8_1 default), `SQL SUM` over
`CUPTI_ACTIVITY_KIND_KERNEL` (`end−start`), the same hardware timestamps. The
`nsys` auto-import of `.qdstrm` is broken on this box ("importer binary …
not found"); workaround is invoking the importer directly —
`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i <x>.qdstrm -o <x>.nsys-rep -f`
— then `nsys stats` as usual.

| metric | §3.0 baseline | post-2.3 | Δ |
|---|---|---|---|
| total GPU kernel time (full run) | 27.17 s | **12.82 s** | **−52.8%** |
| decode-busy GPU util (attn-bracketed) | 84.9 % | 71.7 % | −13 pt |
| `[spine-cuda]` tok/s | 11.89 | **21.14** (nsys) / 23.9 (bare) | ~2.0× |
| `cudaLaunchKernel` (host) | 867,076 (7.52 s) | 780,799 (4.24 s) | −10 % |
| `cudaMemcpy` all (host) | 77,619 (16.34 s) | 77,941 (6.98 s) | +0.4 % calls, −57 % time |
| `cudaMemset` | 129,670 | 116,509 | −10 % |
| `cudaMemcpy` DtoD (GPU) | 65,614 | 65,935 | ~flat |

The dense-gemv rewrite cut the GPU's per-token work roughly in half — exactly
the ~62%-of-kernel cost §3.0 attributed to the three dense gemvs — while the
**orchestration counts are essentially flat** (launches, memcpys, memsets moved
≤10 %). The host `cudaMemcpy` time dropped 57 % purely because each synchronous
transfer now waits on a much less busy GPU, not because there are fewer
transfers.

### Decision → Phase 3

GPU kernel time is no longer the dominant term; the wall-clock is now bounded by
orchestration (launches / memcpys / memsets). This is the Phase 3 budget
(per-call sync / resident activations, Option A in
`generation-integration-bottleneck.md`), not further per-op offload. The exact
per-token budget is measured in §3.1, which corrects the whole-run÷64 figures.

## 3.1 — measurement methodology (NVTX) + corrected per-token counts

The §3.0 numbers ("~12,200 launches / ~1,218 memcpys / ~1,820 memsets per
token") were **whole-run totals ÷ 64** — they swept in the resident path
(`generate_reference(Cuda)`), the prefill spine, and the M4 `generate(Cuda)`
re-routes, none of which is the decode hot path. 3.1 re-measures with **NVTX
ranges** so every "per token" number counts only the steady-state decode-step
window.

### Instrumentation (opt-in)

`SonicCudaBackend::execute` pushes an NVTX range per graph node (op name) and
`generate_spine_cuda` brackets each decode step with `decode_step_N`. Both are
gated by `SONICBOOM_NVTX=1` — OFF by default, because emitting the markers
perturbs host timing (see the split-K note below). Capture:

```
SONICBOOM_GEN_STEPS=64 SONICBOOM_SKIP_CPU=1 SONICBOOM_NVTX=1 \
  nsys profile -t cuda,nvtx,osrt --stats=false -o /tmp/nsys_32 ./tests/core/test_generate
```

The `.qdstrm` is converted with the importer workaround (§3.0), then device
activities are attributed to one `decode_step_N` range by
`globalTid` + `correlationId` (NVTX host range → `CUPTI_ACTIVITY_KIND_RUNTIME`
host call → device `KERNEL`/`MEMCPY`/`MEMSET`).

### One steady-state decode step (decode_step_32 == decode_step_33)

| metric | per token |
|---|---:|
| kernels | **2,005** |
| memcpy | **208** (205 DtoD + 2 HtoD + 1 DtoH) |
| memset | **344** |
| GPU kernel time | 38.9 ms |
| host step wall | 49.2 ms |
| in-step GPU util | ≈ 81 % (kernel 38.9 ms + memcpy/memset 0.8 ms ÷ 49.2 ms) |
| `[spine-cuda]` tok/s | 20.5 (nsys) |

The prior "~12,200 launches / ~1,218 memcpys / ~1,820 memsets per token" are
**6× high**: 780,799 whole-run `cudaLaunchKernel` ÷ 64 also counted the resident
path and the two extra spine re-runs. The real decode step is 2,005 kernels.

> Timing is noisy on this box: the decode GPU contends with an unrelated
> `fm.exe` (Football Manager 2023) holding ~3.6 GB VRAM, so bare `[spine-cuda]`
> tok/s swings (12.5–26 depending on load; 23.9 in §2.3, 20.5 under nsys here).
> The **counts** above are structural (fixed by the graph, not by timing) and are
> unaffected by the contention.

### Kernel list, one decode step (by name)

| kernel | n | GPU ns |
|---|---:|---:|
| quantize_q8_1 | 259 | 3,020,220 |
| rms_sum_sq_kernel | 211 | 959,600 |
| rms_inv_kernel | 211 | 471,872 |
| rms_apply_dev_kernel | 211 | 407,433 |
| gemv_q3_K_q8_1 | 151 | 12,484,249 |
| add_kernel | 126 | 243,875 |
| gemv_q4_K_q8_1 | 104 | 8,325,755 |
| rms_heads_sum_kernel | 91 | 1,589,490 |
| rms_heads_inv_kernel | 91 | 196,100 |
| rms_heads_apply_kernel | 91 | 193,279 |
| mul_kernel | 84 | 198,469 |
| gemv_dense_f32_v | 84 | 2,712,190 |
| gelu_fp16_kernel | 84 | 219,363 |
| rope_neox_heads_kernel | 66 | 147,453 |
| cast_fp16_kernel | 48 | 80,674 |
| scale_kernel | 44 | 81,246 |
| decode_attention_kernel | 42 | 7,074,752 |
| gemv_q5_K_q8_1 | 4 | 206,653 |
| layer_combine_kernel | 1 | 2,176 |
| gemv_bf16_v | 1 | 289,438 |
| cast_bf16_kernel | 1 | 2,208 |
| **total** | **2,005** | **38,906,495** |

The three quantized gemvs (q3+q4+q5 = 23.0 ms) + attention (7.1 ms) + the
quantize kernels (3.0 ms) are 85 % of GPU time — compute, not orchestration.

### memcpy by direction (one decode step)

| direction | n | bytes | GPU ns | host ns |
|---|---:|---:|---:|---:|
| DtoD | 205 | 741,376 | 354,366 | 1,922,078 |
| HtoD | 2 | 53,248 | 8,447 | 2,341,074 |
| DtoH | 1 | 1,048,576 | 159,679 | 10,013,837 |

The **2 HtoD** are the two CPU-side embedding gathers, uploaded each token:

- `token_embd` q3_K row gather — 10,240 B (= d 2560 × f32)
- `per_layer_token_embd` q4_K row gather — 43,008 B (= 42 × 256 × f32)

The **1 DtoH** is the full logits vector (262,144 × f32 = 1 MB) downloaded for
CPU argmax. Its host cost is **10.0 ms** — the synchronous `cudaMemcpy` blocks
until the GPU queue drains. The 205 **DtoD** are the defensive pre-in-place
copies (§3.3); at ~9.4 µs host each they do *not* host-sync (DtoD has no stream
sync), so they are cheap individually but numerous.

### memset by value (one decode step) — 344 total, 48,488 B

| bytes | n | what |
|---|---:|---|
| 8 | 211 | rms_sum accumulator zero (one per sum_sq) |
| 16 | 48 | scalar/accum zero |
| 1024 | 42 | per-layer 256-f32 zero |
| 64 | 42 | 16-f32 zero |
| 336 | 1 | 84-f32 zero |

### Per-layer count: ~48, not ~290

2,005 kernels / 42 blocks ≈ **48 kernels per layer** (the "~290" figure was
~12,200 ÷ 42, the same whole-run inflation). Composition per layer: RMSNorm
3 kernels × ~5 (sum_sq / inv / apply) ≈ 15, quantize_q8_1 ≈ 6, gemv ≈ 8
(q3 3.6 + q4 2.5 + dense 2 + q5 ~0.1), elementwise ≈ 8 (add 3 + mul 2 + gelu 2
+ scale 1), attention 1 + rope 1.6 + rms_heads ≈ 2 + cast ≈ 1. **RMSNorm being
three separate kernels is the single biggest per-layer contributor** and the
obvious fusion target for §3.4 (sum_sq + inv + apply → one kernel, and
norm→q8_1 quantize).

### Split-K `atomicAdd` flakiness — RESOLVED (deterministic two-phase reduction)

Isolating the decode step surfaced a test-stability issue: the live-vs-live CUDA
checks in `test_generate` (`generate(Cuda)` routes, `prefill(Cuda)`,
`generate_prompt(Cuda)`) failed ~1 run in 3 because the split-K **dense** gemv's
`atomicAdd` accumulation order shifted with host timing. The K-quant path was
already deterministic (q8_1 dp4a is the default); the flake came from
`gemv_dense_f32_v` / `gemv_bf16_v` (the `inp_gate` shape: 256 rows → k_split 4).

Fixed by replacing the float `atomicAdd` with a deterministic two-phase
reduction: each split chunk writes its own partial (`dst[row*k_split+chunk]`),
and a second `gemv_reduce_kernel` sums the chunks in fixed order. Verified with
50 consecutive `test_generate` runs — 50/50 green, uniform token stream. The
NVTX opt-in rationale (avoid perturbing the atomic order) no longer applies to
correctness; markers remain opt-in only to keep host timing out of the nsys
measurement window.

### Re-measured after the 3.2 device-resident work (B1–B4)

The 3.2 work (B1–B4) moved the token-embedding gather and the final argmax onto
the device: the next-token id lives in a device buffer, `argmax_kernel` writes
into it, and the two embedding gathers read it back from the device weight —
the only host round-trip left is the 4-byte token id for the loop bookkeeping.
This subsection re-runs the §3.1 attribution on a fresh capture and records the
corrected per-token counts. `decode_step_32 == decode_step_33` still holds
(2,050 kernels / 207 memcpy / 302 memset on both).

| metric | per token (§3.1) | per token (post-3.2) |
|---|---:|---:|
| kernels | 2,005 | **2,050** |
| memcpy | 208 (205 DtoD + 2 HtoD + 1 DtoH) | **207** (206 DtoD + 0 HtoD + 1 DtoH) |
| memset | 344 | **302** |

The goal of 3.2 is met: the **2 per-token HtoD** embedding uploads are gone, and
the **1 DtoH** shrank from the full 1 MB logits vector to a **4-byte** argmax
index. The +45 kernels and −42 memsets are fully accounted for below.

### Kernel list after 3.2 (one decode step, by name)

| kernel | n | GPU ns |
|---|---:|---:|
| quantize_q8_1 | 259 | 1,678,287 |
| rms_sum_sq_kernel | 211 | 957,495 |
| rms_inv_kernel | 211 | 472,684 |
| rms_apply_dev_kernel | 211 | 408,912 |
| gemv_q3_K_q8_1 | 151 | 14,400,102 |
| add_kernel | 126 | 241,847 |
| gemv_q4_K_q8_1 | 104 | 4,425,671 |
| rms_heads_sum_kernel | 91 | 1,210,865 |
| rms_heads_inv_kernel | 91 | 196,759 |
| rms_heads_apply_kernel | 91 | 190,615 |
| mul_kernel | 84 | 194,231 |
| gemv_dense_f32_v | 84 | 1,305,192 |
| gelu_fp16_kernel | 84 | 210,234 |
| rope_neox_heads_kernel | 66 | 148,536 |
| cast_fp16_kernel | 48 | 82,428 |
| scale_kernel | 44 | 76,351 |
| gemv_reduce_kernel | 42 | 113,627 |
| decode_attention_kernel | 42 | 9,051,686 |
| gemv_q5_K_q8_1 | 4 | 207,929 |
| layer_combine_kernel | 1 | 2,240 |
| gemv_bf16_v | 1 | 263,828 |
| gather_q4_K_row | 1 | 6,239 |
| gather_q3_K_row | 1 | 4,032 |
| cast_bf16_kernel | 1 | 1,760 |
| argmax_kernel | 1 | 79,516 |
| **total** | **2,050** | **35,931,066** |

The +45 kernels vs §3.1 split into two independent causes:

- **+3 from 3.2 (B1–B4)** — the two embedding gathers moved from host
  `dequantize_row_f32`+upload onto the device (`gather_q3_K_row` for
  `token_embd`, `gather_q4_K_row` for `per_layer_token_embd`), and the CPU
  argmax became `argmax_kernel`.
- **+42 from the split-K fix (A stream)** — `gemv_reduce_kernel`, one per
  layer's `inp_gate` dense gemv, the deterministic two-phase reduction described
  above.

(GPU-ns columns are as noisy as §3.1's — the same `fm.exe` VRAM contention —
and are listed only for reference; the counts are structural.)

### memcpy by direction after 3.2 (one decode step)

| direction | n | bytes | GPU ns |
|---|---:|---:|---:|
| DtoD | 206 | 741,380 | 357,491 |
| DtoH | 1 | 4 | 1,792 |
| HtoD | 0 | 0 | 0 |

The DtoD count is 205 → 206 (the extra 4-byte device copy from the argmax output
into the resident token buffer), the two HtoD embedding uploads are gone, and
the single DtoH is now just the 4-byte token id. The dominant DtoD traffic is
unchanged: the defensive pre-in-place copies (§3.3) still dominate the count.

### memset by value after 3.2 (one decode step) — 302 total, 5,480 B

| bytes | n | what |
|---|---:|---|
| 8 | 211 | rms_sum accumulator zero (one per sum_sq) |
| 16 | 48 | scalar/accum zero |
| 64 | 42 | 16-f32 zero |
| 336 | 1 | 84-f32 zero |

The only change is the **1024-byte ×42** "per-layer 256-f32 zero" class is gone
(344 → 302): the per-layer token-embedding buffer no longer needs a zero-fill,
because the on-device gather writes the row directly. The remaining memsets are
the rms accumulator / scalar / 16-f32 / 84-f32 zeros, all still on the table for
the §3.4 fusion pass.

