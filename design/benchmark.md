# SonicBoom benchmark — Gemma 4 E4B-It Q3_K_M

> **Canonical numbers moved to `tools/bench.sh` (2026-10-16).** The document
> below is the *history* of the v1 performance work: its numbers were collected
> with harnesses that were re-derived per section, and several of them were
> later found to be measured with a different warm-up or at a different depth
> than the label claimed. They are kept because the per-kernel analysis, the
> divergence records and the negative results are still what explains the
> current design — **not** as a comparison table. Reproducible, same-session,
> median-of-3 numbers are now produced only by `tools/bench.sh`, and the ones
> current as of **`boom-next` (2026-10-19)** are in the README's
> [Numbers](../README.md#numbers) section: Gemma 53.6 / 53.5 tok/s decode
> (1.20× / 1.21× llama.cpp), 2084 tok/s prefill (1.04×), 57.4 ms prefix reuse;
> Qwen2.5-0.5B 345.9 / 307.0 tok/s (1.13× / 1.01×), 15301 tok/s prefill
> (1.09×), 13.7 ms reuse. (v0.0.3 was 53.0 / 52.1, 1.18×, 1990 tok/s, 65.2 ms;
> 344.7 / 296.4, 1.12× / 0.98×, 15346 tok/s, 13.7 ms.)

**Test date: 2026-10-12** (final v1-performance state; development spanned
2026-10-06 → 2026-10-12). Supersedes the 2026-10-07 v0 baseline.

**Status at v1 wrap-up (history).** CUDA decode was **ahead** of llama.cpp on
the 64-step protocol (1.14×) and at long context (1.11× at depth 512, 1.04× at
depth 2048); the chunked batched prefill was ~0.53–0.59× llama.cpp (§1.4).
The remaining decode gap was one dominated kernel (the q8_1 gemv); the
attention growth at depth ≥ 2048 was traced to split-KV chunk serialization at
large n_ctx and fixed with a `kMaxChunk` cap (§1.2.1). Prefill was closed later
— v0.0.3 is at 1.00× on Gemma 4 and 1.09× on Qwen2.5 (README).

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
  `n_ctx = depth`, compared against
  `llama-bench -p 0 -n 64 -d <depth> -ngl 99` (depth is set with `-d`). Both
  sides must run at the **same depth** — the earlier `-p 1 -n 64` baseline
  silently measured at depth 1, not the stated depth.

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
| llama.cpp (tg64, depth 0, ngl 99) | 45.22 | `llama-bench -p 0 -n 64 -d 0` |

**Reading.** The v0→v1 CUDA decode path gained **10.1×** (4.33 → 43.6 tok/s).
The big levers were the q8_1 dp4a gemv (2.2), the dense-gemv rewrite (2.3), the
resident-forward path (3.2), and split-KV attention (3.4); CUDA Graphs and the
T1–T5 fusions are the tail. At the end of T5 SonicBoom CUDA was **1.02×
llama.cpp** on the 64-step protocol; the release number (re-measured on HEAD,
idle GPU) is **51.65 tok/s (1.14× llama.cpp)** — the post-T5 device-resident
position-major layout + fused-quantize work (§1.4) also sped up decode.

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
| 512 | 576 | 49.96 tok/s (20.02 ms/step) | 44.76 tok/s (22.34 ms/step) | **1.12×** |
| 2048 | 2112 | 47.13 tok/s (21.22 ms/step) | 44.08 tok/s (22.69 ms/step) | **1.07×** |

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
12% tail error). llama.cpp = `llama-bench -p 0 -n 64 -d <depth> -ngl 99`,
with its depth set by `-d`; its `tg64` changed from 44.76 to 44.08 tok/s over
depths 512→2048.

**Reading.** SonicBoom now *leads* llama.cpp at both depths (1.12× at 512,
1.07× at 2048). The 512→2048 penalty is ~6.0% (20.02→21.22 ms/step) and is
**essentially all attention**: the split kernel grows 1.29→2.36 ms as the
global layers span the full window, while gemv q8_1 (~14.9 ms, ~64%) is
unchanged. llama.cpp also slows modestly over these depths (44.76→44.08), so the long-context lead narrows slowly with depth — the same
sublinear attention growth §2 quantifies, now visible end-to-end.

### 1.2.1 n_ctx sensitivity at fixed depth (split-KV chunk serialization)

The 512→2048 slowdown (the §1.2 penalty) prompted a sweep of `n_ctx` at a
**fixed depth 512** (`n_ctx = 2048/8192/32768`), isolating the attention
chunk-size dependence.
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

### 1.4 Prefill (prompt processing) — chunked + batched, ~0.5–0.6× llama.cpp

Prompt-processing throughput (pp, tokens/sec) of the device-resident chunked
prefill (`generate_prefill_spine_cuda` → `SonicCudaBackend::prefill`, chunked at
n > 512), at a fixed prompt length, vs `llama-bench -p <n> -n 0 -ngl 99`:

| prompt | SonicBoom CUDA pp | llama.cpp CUDA pp | ratio |
|---|---|---|---|
| 512 | 1148 tok/s (445.9 ms) | 1949 tok/s | 0.59× |
| 2048 | 1058 tok/s (1935.2 ms) | 1993 tok/s | 0.53× |
| 4100 | 1051 tok/s (3901.0 ms) | 1944 tok/s | 0.54× |

(`tools/baseline/prefill_bench --prompt N --reps 3`, wall pp; llama-bench
`-r 1 -n 0 -ngl 99`, build `43fe9c642`.)

**Chunked prefill.** n > 512 is handled by chunked prefill (B.2): the prompt is
prefilled in ≤512-token chunks, each landing in a wrap-free SWA run, with the KV
cache advanced chunk-by-chunk. This lifts the old one-shot prefill's n ≤ 512
(SWA ring) constraint.

**What fixed it.** The prefill is now a real batched quantized matmul: dequant +
cuBLAS fp16 batched GEMM for the K-quant projections (C.1), MMQ int8 tensor-core
matmul (C.2), position-major `[n, dim]` activations so the fused rms-norm/quantize
runs column-parallel without transposes (C.3), and mmq fused-quantize kernels
(C.4). The old path looped `gather_col → gemv → scatter_col` per column with
host-side transposes/norms/gates; the GPU was nearly idle. That column-loop and
host-bound path is gone.

**Reading.** Prefill is now ~0.53–0.59× llama.cpp (was 0.010–0.012×): the wall pp
is roughly flat at ~1050–1150 tok/s across 512–4100 (the ~constant per-call
plan/graph overhead dominates at small n). The residual gap is kernel efficiency
in the batched K-quant matmul (MMQ int8 / cuBLAS fp16) versus llama.cpp's tuned
q*_K batched GEMM — not the host-bound column-loop of v0.

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

**Prefill (§1.4) — now ~0.53–0.59× llama.cpp (was 0.010–0.012×).** The chunked
device-resident batched prefill runs ~1050–1150 tok/s vs llama.cpp's
~1950–1990 tok/s, no longer the host-bound per-column gap of v0. The residual
prefill gap is kernel efficiency in the batched K-quant matmul (MMQ int8 /
cuBLAS fp16); decode's q8_1 gemv (§1.1) is again the top lever below.

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

## v1 release

SonicBoom v1 (wrapped 2026-10-13) closes the performance scope on Gemma 4
E4B-It Q3_K_M: CUDA decode ahead of llama.cpp, a chunked batched prefill path
at ~0.5× llama.cpp, and prefix-cache KV reuse that makes long-prefix
continuation cheap. This section is the release-time summary of the numbers, the
baseline, and the reproduction environment; the per-section detail is above.

**Decode (CUDA).** 51.65 tok/s on the 64-step steady protocol — **1.14×
llama.cpp** (45.22 tok/s, `tg64`/`ngl 99`, same GGUF + machine). Long-context
steady decode: **1.12×** at depth 512 (49.96 vs 44.76 tok/s) and **1.07×** at
depth 2048 (47.13 vs 44.08 tok/s). The step is GPU-bound; the q8_1 dp4a gemv
holds ~64% of GPU time (§1.1, §1.2, §2).

**Prefill (CUDA).** Chunked device-resident batched prefill: `pp512` 445.9 ms /
1148 tok/s, `pp2048` 1935.2 ms / 1058 tok/s, `pp4100` 3901.0 ms / 1051 tok/s
vs llama.cpp ~1949/1993/1944 tok/s (**0.53–0.59×**) — a real batched quantized
matmul (MMQ int8 tensor-core + cuBLAS fp16), no longer the host-bound
per-column path of v0. The batched prefill reproduces the per-token path
token-for-token (§1.4).

**Prefix reuse (CUDA).** `generate_prefill_spine_cuda_prefix` matches the
shared prompt prefix against the token history and prefills only the suffix.
First-token latency at prefix 4000 + 100 new: **502.6 ms reuse vs 3903.1 ms
full prefill → 7.8×**. Reuse continuation is token-identical to the full
prefill over 32 tokens at every acceptance prefix (512/600/1000/1040/3584/
3600/3996/4000/4095/4097 +100, and 4000 +600); `test_prefix_cache` 11/11.

**Baseline (llama.cpp).** local checkout HEAD **`43fe9c642`**, CUDA build
(`build-cuda/bin`); the dev-time *baseline only* (not an oracle).

**Environment.** NVIDIA RTX 3050 (8 GiB, sm_86); driver **595.84** (CUDA
13.2); nvcc **12.0** (C++20 host); g++-13 / C++23 core; Linux 6.8. See
§Environment for the full table.

**Reproduce.** `## Reproduce` (below) has the exact command lines for the
64-step decode, long-context decode, llama.cpp baseline, and the full suite.

## post-v1 (deferred / blocked)

Recorded at v1 wrap-up; none of these block the release.

- **Qwen2 decode — was blocked at v1; later resolved.** The arch-generic
  loader (`detect_arch` + `load_config`) reads a Qwen2.5 GGUF
  (`general.architecture` == "qwen2") and derives a Qwen2 `Gemma4Config`;
  `test_qwen2_config` asserts the frozen 0.5B facts (24 blocks, d 896, ff 4864,
  14/2 heads, head_dim 64, context 32768, vocab 151936, rope 1e6, no
  SWA/shared-KV/per-layer-input/softcap). At v1 wrap-up, weight binding, the
  Qwen plan builder and end-to-end verification were deferred because Qwen2.5
  Q4_K_M uses Q5_0/Q8_0/Q6_K quant types and attention q/k/v biases SonicBoom
  did not yet model. All of it landed after v1: Qwen2.5-0.5B is one of the two
  verified families in v0.0.3 (README). The paragraph is kept as the v1 record.
- **CUDA plugin split — deferred (#58).** Splitting the CUDA backend into a
  runtime-loadable plugin (Step 4 Deliverable 2) awaits an amendment decision.

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

## GGUF load-time quant fallback validation (2026-10-14)

The user-provided `models/gemma-4-E4B-it-Q4_K_M.gguf` is the Unsloth Q4_K_M
quantization. Its SHA-256, supplied by the user and verified locally, is
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`. The
GGUF is local validation data and is not part of the repository.

Loading converts 42 `q6_K` tensors to Q5_K:

```text
sonicboom: converted 42 tensors to Q5_K at load time (types: q6_K x42); native kernels pending
```

On the RTX 3050, `test_generate` completed its 64-token CUDA run at **51.526
tok/s**. The standalone load-plus-generation measurement was **34.631 s** for
Reader, model weights, vocabulary, and setup, followed by **51.501 tok/s** for
tg64. Generated text was coherent; the prompt requested the derivative of
`f(x) = x^2`, and the output gave a solution using the power rule. The CUDA
prefill benchmark (`--prompt 512 --reps 3`) reported **446.850 ms** for 512
tokens, **1145.8 tok/s wall**, and **1670.8 tok/s marginal** (best of 3).

The Q3_K_M unchanged-path golden token sequence remained
`236761 107 236769 236776`. After the final loader adjustment allowing
already-supported Q3_K/Q4_K/Q5_K blocks in a Q4_K_M file, the rebuilt
`test_generate` passed its 64-token run and the full CTest run passed **53/53**
tests. `test_prefix_cache` passed all 11 cases, and
`test_cuda_elementwise` passed its batch invariance checks.

## Q4_K_M (native Q6_K) (2026-10-14)

The validation file is the user-provided Unsloth quantization at
`models/gemma-4-E4B-it-Q4_K_M.gguf`, SHA-256
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`.
Q6_K is native; loading emitted no `sonicboom: converted ...` summary. The
Q6_K MMQ weight kernel remains disabled. Q6_K prefill uses D4_Q8_1 activation
scales followed by fp16 dequantization and cuBLAS; decode uses the q8_1 MMVQ
kernel. A 64-token prompt asking for the derivative of `f(x) = x^2` generated
a coherent answer using the power rule with default `SONICBOOM_MMQ=1`.

| Measurement | SonicBoom CUDA | llama.cpp CUDA | Ratio |
|---|---:|---:|---:|
| tg64, depth 0 | 49.34 tok/s | 54.99 tok/s | 0.90x |
| tg64, depth 512 (n_ctx 576) | 48.29 tok/s | 54.30 tok/s | 0.89x |
| tg64, depth 2048 (n_ctx 2112) | 46.30 tok/s | 53.32 tok/s | 0.87x |
| pp512 | 1082.6 tok/s (472.950 ms) | 2048.02 tok/s | 0.53x |
| pp2048 | 989.2 tok/s (2070.286 ms) | 2084.09 tok/s | 0.47x |
| pp4100 | 986.6 tok/s (4155.569 ms) | 2035.26 tok/s | 0.48x |

SonicBoom decode used `longctx_bench` with `SONICBOOM_GEN_TIMING=1`; the
steady-rate output excludes plan setup and the first decode step. For depth
512/2048, `n_ctx` was pinned to 576/2112. Prefill is the chunked prefix-session
path; `prefill_bench` reports wall throughput and uses three repetitions (best
run). llama.cpp is checkout `43fe9c642`; decode used
`llama-bench -p 0 -n 64 -d 0,512,2048 -ngl 99 -r 1`; prefill used
`-p 512/2048/4100 -n 0 -ngl 99 -r 1`.

The measured load sequence was 4.243 s in `Reader::load`, 0.269 ms in
`load_gemma4`, and 0.213 s in vocabulary loading (4.456 s total). The GGUF
remains local validation data and is not part of the repository.

## Qwen2.5-0.5B-Instruct Q4_K_M (2026-10-16)

SonicBoom's second verified model family. The file is the Qwen2.5-0.5B-Instruct
Q4_K_M GGUF (local validation data, not part of the repository), SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`; it reports
`general.architecture = qwen2` (d 896, 24 blocks, 14/2 heads, head_dim 64, rope
1e6, no sliding window, tied output embedding). Numbers below are measurement
only — nothing was tuned for this model. Same machine, same CUDA spine, same
tools as the Gemma numbers above (`longctx_bench`, `prefill_bench`,
`prefix_cache_bench`); llama.cpp is the same checkout, `43fe9c642`.

| Measurement | SonicBoom CUDA | llama.cpp CUDA | Ratio |
|---|---:|---:|---:|
| tg64, depth 0 | 255.7 tok/s (250.3 ms) | 307.89 ± 1.43 tok/s | 0.83x |
| tg64, depth 512 | 224.2 tok/s (285.5 ms) | 304.10 ± 3.67 tok/s | 0.74x |
| pp512 | 4036.6 tok/s (126.84 ms) | 14505.81 ± 1063.36 tok/s | 0.28x |
| prefix reuse, 512 + 64 | 68.4 ms first token (full: 299.3 ms) | — | 4.4x vs own full prefill |

Method. Decode is `longctx_bench`, which subtracts an incremental warm decode
from the full one; at depth 512 the three runs agreed to 0.3% (224.48 / 223.81 /
224.16 tok/s), so the tail number is reported there rather than the
whole-range steady average (257.9 tok/s over a 513-step decode, i.e. averaged
from depth 0 — that figure answers "how fast is a long decode", not "how fast is
a decode at depth 512"). Prefill is `prefill_bench --prompt 512 --reps 3` (best
of 3), the chunked prefix-session path: 126.840 ms of 512 tokens = 4036.6 tok/s
wall, 4665.3 tok/s marginal (the N=1 baseline in the same run was 17.308 ms).
llama.cpp used `-p 0 -n 64 -ngl 99 -r 10` / `-d 512` / `-p 512 -n 0 -r 10`.
The reuse number is `prefix_cache_bench --prefix 512 --new 64 --steps 4
--reps 3` over a real token file: reusing a resident 512-token prefix and
prefilling 64 new tokens takes 68.4 ms to the first token against 299.3 ms for a
full 576-token prefill, and the greedy continuation after reuse matches the full
prefill's for all four steps. llama-bench has no prefix-reuse measurement, so
the reuse row is against SonicBoom's own full prefill.

Reading. Decode is 0.74–0.83× llama.cpp on this file, and the depth-0 → depth-512
gap (255.7 → 224.2 tok/s, −12%) is attention at 512 keys, since Qwen2 has no
sliding window and the whole prefix is attended. Prefill is 0.28×: at 512 tokens
the batched K-quant matmul is the same bottleneck as in §1.4, and this file's
`ffn_down` is Q6_K, whose MMQ weight kernel is still disabled (see the Q6_K
entries above) — Q6_K prefill goes through D4_Q8_1 activation scales, fp16
dequantization and cuBLAS. The reuse path is the strongest result here (4.4×
TTFT): Qwen2 has no SWA ring, so a resident prefix is a plain KV slice and
needs no wrap handling.
