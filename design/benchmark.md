# SonicBoom benchmark — Gemma 4 E4B-It Q3_K_M

**Test date: 2026-10-12** (final v1-performance state; development spanned
2026-10-06 → 2026-10-12). Supersedes the 2026-10-07 v0 baseline.

**Status: v1 performance complete through Step 4.** CUDA decode is at
llama.cpp parity on the 64-step protocol (1.02×) and at long context (1.02× at
depth 512, 0.97× at depth 2048). The remaining gap is one dominated kernel (the
q8_1 gemv); the attention growth at depth ≥ 2048 was traced to split-KV chunk
serialization at large n_ctx and fixed with a `kMaxChunk` cap (§1.2.1).

## Scope

- **Model:** `models/gemma-4-E4B-it-Q3_K_M.gguf` (3.76 GiB, 7.52 B params, 42
  blocks, d=2560, ff=10240, n_heads_q=8, n_heads_kv=2, shared KV, SWA window 512).
- **SonicBoom:** C++23 / g++-13, `SONICBOOM_USE_CUDA=ON`. Decode runs the fused
  planner spine (`generate_spine_cuda`) with the device-resident path + CUDA
  Graph capture + the T1–T5 fused kernels behind `Backend::record_step`.
- **Baseline (llama.cpp):** local checkout, HEAD `43fe9c642`, CUDA build
  (`build-cuda/bin`). Same GGUF, same machine. llama.cpp is the dev-time
  *baseline only* (not an oracle); SonicBoom's reference is its own CPU path.
- **Protocol:** 64-step steady decode = `test_generate` at
  `SONICBOOM_GEN_STEPS=64` (single seed + N greedy steps, no tokenizer, no nsys,
  idle GPU). Long context = the runtime's single-run steady metric
  (`SONICBOOM_GEN_TIMING=1`, "steady tok/s") over a `depth`-step decode at
  `n_ctx = depth`, compared against `llama-bench -p <depth> -n 64 -ngl 99`
  (whose `n_ctx = depth + 64`). Both sides must run at the **same n_ctx** — the
  earlier `-p 1 -n 64` baseline silently measured at `n_ctx = 65`, not the
  stated depth.

## Environment (recorded separately per v1 rules)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 (8 GiB, compute 8.6 / sm_86) |
| Host | Linux 6.8, x86-64 |
| Driver | **595.84** |
| Driver-supported CUDA | **13.2** |
| nvcc (toolkit) | **12.0** (V12.0.140, built Jan 2023) |
| Compiler | g++-13 (C++23, SonicBoom core); nvcc 12.0 (C++20, `.cu` host) |

Note the driver (13.2) is ahead of the nvcc toolkit (12.0); the build targets
`CMAKE_CUDA_ARCHITECTURES=86` (sm_86). cuDNN / MAGMA / cutlass are not used.

## Correctness gate

- **Decode golden:** `test_generate` reproduces the llama.cpp multi-token golden
  stream; `CpuQ8K` and `Cuda` spines are compared per token.
- **Full suite:** `ctest --test-dir build` → **49/49** (CUDA ON); the CUDA-OFF
  build (Step 4) is **45/45** with the CUDA tests compiled out.
- **Op-level consistency:** `test_cuda_elementwise` / `test_cuda_quant` /
  `test_cuda_gather` (device vs CPU reference, fp32 tolerance / bit-exact) all
  pass; `test_nn` (CPU self-consistency) exact.

## 1. Decode throughput (tokens/sec)

### 1.1 End-to-end progression (64-step steady decode, CUDA)

| stage | tok/s | note |
|---|---|---|
| v0 baseline (Phase 1) | 4.33 | unoptimized spine |
| Phase 2.2 q8_1 activation default | 12.04 | q8_1 dp4a gemv |
| Phase 2.3 dense-gemv rewrite | 23.9 | vectorized split-K dense |
| Phase 3.2 device-resident forward | 26.4 | kill per-token H2D/D2H |
| Phase 3.4 split-KV flash-decode attention | 33.4 | attention 6.5→0.9 ms/step |
| Phase 3.5 CUDA Graphs | 36.3 | launch-overhead amortization |
| **Step 1 HAL refactor** | **36.3** | behavior invariant (±0) |
| **T1** quantize-once reuse (CSE) | **37.0** | +1.7% |
| **T2** rms_norm single-pass | **42.4** | +2.1% (own baseline) |
| **T3** rms_norm+quantize fusion | **42.7** | |
| **T4** per-head rms_norm single-pass | **43.4** | |
| **T5** gelu×up→q8_1 fusion | **43.6** | **final** |
| llama.cpp (tg64, ngl 99) | 42.63 | same protocol |

**Reading.** The v0→v1 CUDA decode path gained **10.1×** (4.33 → 43.6 tok/s).
The big levers were the q8_1 dp4a gemv (2.2), the dense-gemv rewrite (2.3), the
resident-forward path (3.2), and split-KV attention (3.4); CUDA Graphs and the
T1–T5 fusions are the tail. At the end SonicBoom CUDA is **1.02× llama.cpp** on
the 64-step protocol.

**Caveat on absolute values:** the T1→T2 absolute jump (37.0 → 42.4) is *not* a
step gain — the two commits were measured in different sessions with different
co-resident GPU load (a sunshine/display process). The trustworthy per-step
numbers are the *deltas* each commit reports against its own same-session
baseline (T1 +1.7%, T2 +2.1%, T3 +0.7%, T4 +1.6%, T5 +0.5% ⇒ ≈ **+6.7%
cumulative** over the pre-T1 spine). The Step 1 HAL refactor was measured as a
behavior invariant (~36.3 tok/s, ±0), as intended.

### 1.2 Long context (steady decode at a fixed KV depth)

| depth | n_ctx | SonicBoom CUDA | llama.cpp CUDA | ratio |
|---|---|---|---|---|
| 512 | 576 | 45.95 tok/s (21.76 ms/step) | 45.23 tok/s (22.11 ms/step) | **1.02×** |
| 2048 | 2112 | 44.02 tok/s (22.71 ms/step) | 45.25 tok/s (22.10 ms/step) | 0.97× |

Re-measured post-T1-T5 with CUDA Graphs ON on an idle GPU, at matching n_ctx
(the 2026-10-07 table above — 34.55/34.78 vs 39.87/37.56 — was measured at
`n_ctx = 65`, i.e. depth 1 not 512/2048, and is **stale**; it understated both
sides and, because llama.cpp's n_ctx also shrank, misstated the ratio).

**Method.** SonicBoom = the runtime's single-run steady metric
(`SONICBOOM_GEN_TIMING=1`, "steady tok/s", first step excluded) over a
`depth`-step decode at `n_ctx = depth` — the *same* decode llama-bench's
`tg64` measures, but averaged over the whole depth range rather than a 64-step
tail (the `longctx_bench` subtraction is thermal-fragile at depth 2048: two
~47 s decodes subtract to a ~1.3 s tail, so a 0.4% GPU-clock drift becomes a
12% tail error). llama.cpp = `llama-bench -p <depth> -n 64 -ngl 99`, whose
`n_ctx = n_prompt + n_gen = depth + 64`; its `tg64` is flat (45.23→45.25)
because its decode is not attention-bound at these depths.

**Reading.** SonicBoom now *leads* llama.cpp at depth 512 (1.02×) and trails
slightly at 2048 (0.97×). The 512→2048 penalty is ~4.6% (21.76→22.71 ms/step)
and is **essentially all attention**: the split kernel grows 1.29→2.36 ms as
the global layers span the full window, while gemv q8_1 (14.9 ms, ~66%) is
unchanged. llama.cpp has no such penalty at these depths, so the long-context
gap widens slowly with depth — this is the same sublinear attention growth §2
quantifies, now visible end-to-end.

### 1.2.1 n_ctx sensitivity at fixed depth (split-KV chunk serialization)

The depth-2048 penalty (0.97×) prompted a sweep of `n_ctx` at a **fixed depth
512** (`n_ctx = 2048/8192/32768`), isolating the attention chunk-size dependence.
Split-KV attention computes `chunk = ceil(window/target_chunks)` with
`target_chunks = sm_count()*4/n_heads_kv ≈ 40`; for global layers `window =
n_ctx`, so **chunk grows linearly with n_ctx** (52 → 205 → 820) while the current
depth stays 512. The active split blocks collapse from 20 (n_ctx 2048) to 2
(n_ctx 32768), serializing the whole 512-key window onto one block per KV head.

| n_ctx | chunk | active split blocks | split (ms) | steady tok/s |
|---|---|---|---|---|
| 2048 | 52 | 20 | 1.95 | 45.19 |
| 8192 | 205 | 6 | 9.32 | 41.24 |
| 32768 | 820 | 2 | 20.36 | 38.49 |

(depth 511, graphs-OFF `nsys nvtxkernsum` for the split; `SONICBOOM_GEN_TIMING=1`
for the steady tok/s.) The growth is real and pathological — split 1.95→20.36 ms
(10×) and tok/s 45.19→38.49 (−15%) as n_ctx 2048→32768 — confirming the
chunk-serialization hypothesis that the depth-2048 §2 sweep had prematurely
refuted (its `n_ctx=2048` held chunk ≤ 52, never reaching the serialization
regime that appears at `n_ctx ≥ 8192`).

**Fix.** Cap `chunk` at `kMaxChunk = 16` so the grid scales with `window`:
`ceil(window/chunk)*n_heads_kv` blocks, with blocks beyond the current `n_keys`
early-exiting. At depth 512 this keeps 32 active chunks (64 blocks, ~3.2/SM) for
*any* n_ctx, matching the SWA layers' parallelism (chunk 13 → 80 blocks). A cap
of 32 is a wave-fill local minimum (16 chunks → 32 blocks, 1.6/SM) and measured
~2.4× slower split (3.20 vs 1.29 ms at depth 512), so 16 was chosen.

After the fix (`chunk = 16`):

| n_ctx | split (ms) | attention split+merge (ms) | steady tok/s |
|---|---|---|---|
| 2048 | 2.72 | 3.21 | 46.40 |
| 8192 | 2.72 | 3.22 | 46.29 |
| 32768 | 2.73 | 3.23 | 46.35 |

The split is now **flat (~2.72 ms)** and tok/s **flat (~46.3)** across n_ctx — the
pathological serialization is gone. The residual split (~2.72 ms at depth 511) is
~1.4 ms above the n_ctx=576 baseline (1.29 ms, chunk 15 → 70 blocks) but does *not*
grow with n_ctx — a memory/cache effect of the larger KV cache, ≤6% of the step,
not a serialization bottleneck.

Reproduce: `SONICBOOM_GEN_TIMING=1 tools/baseline/longctx_bench --depth 512 --steps 1 --n-ctx {2048,8192,32768}`.

### 1.3 CPU decode (reference)

| path | tok/s |
|---|---|
| SonicBoom CPU q8_K (thread pool) | 4.37 |
| llama.cpp CPU 4 threads | 1.53 |
| llama.cpp CPU 12 threads | 2.02 |

SonicBoom's single straight-line spine beats llama.cpp's scheduler at equal
threads; the CPU path is the numerical oracle for the CUDA path, not a perf
target.

### 1.4 Prefill (prompt processing) — host-bound, ~85–100× behind llama.cpp

Prompt-processing throughput (pp, tokens/sec) of the one-shot batched prefill
(`generate_prefill_spine_cuda` → `SonicCudaBackend::prefill`), at a fixed prompt
length, vs `llama-bench -p <n> -n 0 -ngl 99`:

| prompt | SonicBoom CUDA pp | llama.cpp CUDA pp | ratio |
|---|---|---|---|
| 128 | ~20.4 tok/s | 1745.5 tok/s | 0.012× |
| 512 | ~19.8 tok/s | 2019.0 tok/s | 0.010× |
| 2048 | **not supported** | 2014.2 tok/s | — |

(`tools/baseline/prefill_bench --prompt N --reps 3`, marginal pp — the ~constant
per-call plan/graph overhead is subtracted via a 1-token reference; llama-bench
`-r 3 -n 0 -ngl 99`.)

**pp2048 is not supported.** The one-shot prefill requires the prompt to land in a
straight, wrap-free run of every KV cache, and the SWA layers are a 512-slot ring
(`sliding_window == 512`), so `n > 512` is rejected — the batched prefill has no
ring-wrap handling (llama.cpp's chunked/ubatch prefill does).

**Root cause — the prefill is not actually batched.** `matmul_f32` with `n > 1`
does *not* launch a batched GEMM: it loops over the `n` columns, gathering each
into a scratch buffer, running the single-column q8_1 gemv (the decode kernel),
and scattering back (`gather_col → gemv → scatter_col`). So the prefill is `n ×`
the decode path plus gather/scatter, with no batching parallelism. On top of that
the transposes, column-wise rms-norms, and per-layer gates run on the *host*
through the thread pool (per-position scalar matvecs). The GPU is nearly idle: at
n=128 the batched-matmul GPU kernel time is ~114 ms of the ~6387 ms prefill
(`cuda-matvec-prof`), the rest is host thread-pool work (`threadpool-prof`: 83 s
work / 16.8 s handshake summed across workers).

**Reading.** Prefill is ~85–100× behind llama.cpp — a far larger gap than decode
(1.02×). It is a host-bound + column-loop gap, not a kernel gap: the same
single-column q8_1 gemv that makes decode competitive is reused verbatim for
prefill, where it serializes the prompt. The fix is a real batched quantized GEMM
(or cuBLAS) plus moving the host-side per-position work to the device; deferred
(measure-only item, see §5).

## 2. Per-kernel GPU time (decode @ depth 512, CUDA Graphs ON, post-Phase-3)

nsys `gpukernsum` of the decode spine — GPU kernel-time share by family:

| family | share |
|---|---|
| gemv q8_1 dp4a (q3_K 44.8 + q4_K 18.7 + q5_K 0.9) | **64.4%** |
| rms_norm (layer 8.1 + heads 6.9) | 14.9% |
| quantize_q8_1 | 7.2% |
| dense gemv (f32 5.5 + bf16 1.1) | 6.6% |
| elementwise (add/gelu/mul/scale/cast) | ~3.5% |
| attention split/merge | ~1.4% |
| rope + kv_append + argmax + gemv_reduce | ~1.9% |

**GPU utilization.** Before CUDA Graphs the decode step showed ≈80% in-step GPU
util with a ~7.5 ms host-launch idle gap (~20%); after moving the launch
overhead behind `Backend::record_step`, the ~13.76 ms/step of host launch cost is
hidden under GPU execution and the step is GPU-bound (~31 ms/step at depth 512).
The launch overhead is eliminated; it was never the wall-clock bottleneck at this
depth.

**What T1–T5 changed.** The rms_norm (14.9%) was three kernels per pass
(sum/inv/apply), quantize_q8_1 (7.2%) ran 1:1 with the gemvs, and elementwise
was ~3.5%. T1 deduplicated quantize, T2/T4 collapsed the rms_norm trios into one
pass, T3 folded norm→quantize, T5 folded gelu×up→quantize. That removes most of
the ~25% "glue", but because gemv is 64% and unchanged, the end-to-end tok/s only
moved the observed +6.7%.

## 3. Numerical divergence + SWA ring-buffer wraparound

### 3.1 CUDA vs CPU divergence (token 35)

The CUDA spine's logits drift from the CPU spine by ~1e-3 relative (fp32
accumulation order), and at a near-tie flip the argmax. First divergence in the
64-token run: **token 35** (1-based). The decisive finding is that
**llama.cpp's own CPU vs CUDA paths diverge at exactly token 35** (token ids
`2934` vs `3847` — the same tokens and position as SonicBoom), while
**SonicBoom's CUDA spine matches llama.cpp CUDA token-for-token across all 64
tokens**. The divergence is a property of the model's near-tie token, not a
SonicBoom defect. (An earlier Phase-2.3 run showed the same signature at token
44 with logit gap 1.20 — CPU top-1 `236764` 25.397 vs top-2 `779` 24.194, CUDA
swapping them. That Phase-2.3 decode path still used split-K `atomicAdd`, so the
flip position shifted run-to-run; the current decode path is deterministic —
fixed-order split-KV merge (991a1bb) and two-phase split-K dense gemv (0318093)
— so the flip position is now fixed.)

### 3.2 SWA ring-buffer wraparound

The SWA KV cache is a fixed-size ring (`n_slots == sliding_window == 512`);
global layers span the full decode length. The split-KV decode kernel matches the
CPU oracle to **max_abs ~1e-8 / rms ~1e-9** at every probed position — the SWA
wraparound boundary (pos 511/512/513/1000 at n_slots=512) and global full-length
decode out to pos 4096. Wraparound (`% n_slots`) and the chunked/merged online
softmax are both exact.

### 3.3 prefill(Cuda) vs per-token(Cuda) divergence (near-tie, token 37)

The `prefill(Cuda)` test compares two *CUDA* paths over a 3-token prompt: the
batched prefill (`SonicCudaBackend::prefill` — batched matmul + flash attention,
one-shot KV fill) vs the per-token reference (three split-KV decode steps). Both
paths use **q8_1** activation quantization for every K-quant matmul — the 2.2a
fix (f134000) is intact, and `matmul_f32_dev`'s `q8_1_enabled()` gate is what
both the prefill FFN/projection and the decode spine go through. The two paths
differ only in how the 3-row KV cache is built: flash attention (prefill) vs
split-KV decode attention (per-token), which accumulate in different fp32 orders.

That ~1e-3 KV-cache difference flips a near-tie. First divergence (64-token run,
seed 2): **step 36** (0-based; token 37 1-based), where the two paths hold the
same top-2 tokens and merely swap them:

| path       | top-1 | raw logit | top-2 | raw logit | margin |
|------------|-------|-----------|-------|-----------|--------|
| prefill    | 506   | 37.81     | 496   | 37.60     | 0.21   |
| per-token  | 496   | 37.58     | 506   | 37.55     | 0.03   |

(raw = pre-softcap logit; final_logit_softcapping = 30.) The margin is < 0.25
raw, i.e. a near-tie, and the flip is the same class as §3.1's CUDA-vs-CPU token
35: a property of the model's near-tie token, not a SonicBoom defect. The decode
path is deterministic — the split-KV merge walks the chunk loop in fixed order
with no atomics (991a1bb), the dense gemv uses a deterministic two-phase split-K
reduction (0318093), and every q8_1 mmvq / attention / fused-T1–T5 reduction is a
warp `__shfl`. 10× 64-step `generate_spine_cuda` runs produce the identical token
stream, so this flip is a fixed, reproducible near-tie from the flash-attention
(prefill) vs split-KV (per-token) KV-cache numerics, not run-to-run jitter.

## 4. Operator-level: quantized gemv vs cuBLAS (unchanged from v0)

The dominant FFN matmul (`ffn_gate_up`, `ffn_down`) quantized q*_K gemv already
matches or beats cuBLAS fp16 (1.09×) and beats fp32 (1.8×) at decode shape —
quantization reads 0.43 B/weight vs 2 B/weight and the large shapes are
bandwidth-bound. The attention projections were the weak spot (7–10× behind
cuBLAS fp16), which is why the split-KV + resident work in Phase 3 targeted
attention. See the phase docs for the full table.

## 5. Remaining bottlenecks and next steps (tiered by precedence)

The decode step is now dominated by a single family: **gemv q8_1 dp4a = 64.4% of
GPU time**. Everything below is ranked by the project's implementation precedence
(MLIR > graph optimization > hand-written kernels), but the top lever happens to
be hand-written because the K-quant format is SonicBoom-owned.

**New top lever — prefill (§1.4).** The one-shot prefill runs ~20 tok/s vs
llama.cpp's ~2000 tok/s (~85–100×), host-bound with a per-column (non-batched)
matmul, and it caps out at `n = 512` (SWA ring-wrap). That gap dwarfs every
decode lever below and is the first thing to fix for any long-prompt workload;
it is a measure-only finding for now (the fix is a real batched quantized GEMM /
cuBLAS plus device-side norms and gates, and lifting the SWA wrap constraint).

1. **手写 (hand-written) — the q8_1 gemv is the floor and the lever.** The gemv
   is 64% and is already the full llama.cpp mmvq port (q8_1 int8 activation +
   warp-per-row iqs slice + `__shfl` scale broadcast + `__dp4a`, 3.84× over the
   fp32 kernels). Native-torch and MLIR have *no* ggml K-quant kernels, so this
   is necessarily hand-written. **ncu (2026-10-12, base clock) resolved the
   "DRAM vs L1/LSU" probe: shared-memory tiling is not warranted.** The q8_1 port
   already eliminated the L1/LSU bound — `lg_throttle` collapsed from ~75% (the
   fp32 kernels' dominant stall) to ~0.5% in both q8_1 kernels. The two FFN
   shapes now diverge:
   - `gemv_q4_K_q8_1` (ffn_down, bpr=40): **DRAM-bound but not at the floor** —
     78% DRAM + `long_scoreboard` 60% means in-flight requests are insufficient,
     so a few % is still on the table (more super-blocks per warp, sw-pipeline /
     cp.async prefetch of the next round's weights, wider aligned loads, checking
     whether register usage caps occupancy). Recorded as a *possible* follow-up,
     not being done now.
   - `gemv_q3_K_q8_1` (ffn_gate/up, bpr=10): **instruction/compute-bound** (sm
     79%, `math_pipe_throttle` 17%, DRAM 50%) — Q3_K's bit-unpack overhead. Item 1
     (cooperative `__shfl` scale unpack, one unpack per lane then broadcast) landed:
     ffn_gate 121.5→134.5 GB/s (−9.7%) and ffn_up 119.4→132.8 GB/s (−10.1%), still
     below the Q4_K 160 GB/s DRAM floor, so Q3_K remains instruction-bound (the
     150 GB/s / DRAM≥70% target was not met — bit-unpack still dominates). Bit-exact
     (err unchanged). End-to-end +3.6% (64-step) / +4.5% (depth-512).
   - **Item 2 — small-matrix gemv (attn_k/attn_v, 512 rows): measured, no cheap
     win.** attn_k/attn_v run at ~53/~72 GB/s (well below the FFN shapes) because
     512 warps fill only ~53% of the 20 SMs. Two options were tried:
     * multi-warp-per-row / split-K was implemented and **reverted as a net loss**
       (attn_k 53→44.5, attn_v 72→60 GB/s): Q3_K is instruction-bound (the split-K
       `__shfl` reduction + ~2 µs reduce kernel exceed the latency-hiding gain) and
       bpr=10 caps the split at 5 warps/row (2560 warps, far short of saturation).
       Recorded as do-not-retry.
     * row-concat of q/k/v into one gemv is the remaining lever (the three
       projections are separate `QuantizedMatmul` nodes reading the same
       activation), but it needs planner fusion + output slicing + a mixed
       Q3_K/Q4_K kernel — estimated ~1.2% e2e (q+k) to ~4.4% e2e (q+k+v). Deferred
       as a follow-up, not a do-not-retry.
   Neither limiter is the L1/LSU data path a cooperative shared-memory tiling
   would relieve, so that rewrite is a do-not-retry on the q8_1 path (conditions:
   `lg_throttle` ~0.5%, DRAM ~78%, `long_scoreboard` ~60%). The do-not-retry list
   (widen-loads no-op, dp4a-on-fp32, warp-per-row at small bpr, shared-memory
   tiling on q8_1) is recorded in the phase-6a notes.

2. **MLIR (future codegen) — generated fused glue.** The hand-written T1–T5
   fused kernels are the numerical/perf reference for the v1 MLIR GPU lowering
   path (linalg → gpu → nvvm → cubin, JIT-compiled and cached by shape
   signature). Once a dequant-on-the-fly / K-quant custom op is added to the
   lowering, the remaining glue (rms_norm, quantize, residual add, RoPE) can be
   *generated* rather than hand-written. This is where the precedence rule pays
   off going forward — new fusions become generated, not hand-coded.

3. **图优化 (planner) — diminishing returns.** T1–T5 exhausted the large fusions
   (quantize-CSE, norm+quantize, gelu×up→q8_1). The remaining graph-level wins
   are smaller: attention↔projection fusion, KV-cache-aware memory scheduling,
   and dead-computation/invariant hoisting. They are below the gemv rewrite in
   payoff.

## Reproduce

```sh
# 64-step steady decode (SonicBoom)
SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
  SONICBOOM_GEN_STEPS=64 ./build/tests/core/test_generate

# long-context steady decode (SonicBoom; read the "steady tok/s" line)
SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
  SONICBOOM_GEN_TIMING=1 tools/baseline/longctx_bench --depth 512 --steps 1

# baseline (llama.cpp; n_ctx = depth + 64)
$LLAMA_ROOT/build-cuda/bin/llama-bench \
  -m models/gemma-4-E4B-it-Q3_K_M.gguf -p 512 -n 64 -ngl 99 -r 1

# full correctness suite
ctest --test-dir build
```
