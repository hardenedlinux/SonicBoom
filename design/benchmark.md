# SonicBoom benchmark — Gemma 4 E4B-It Q3_K_M

**Test date: 2026-10-12** (final v1-performance state; development spanned
2026-10-06 → 2026-10-12). Supersedes the 2026-10-07 v0 baseline.

**Status: v1 performance complete through Step 4.** CUDA decode is now at
llama.cpp parity on the 64-step protocol (1.02×) and ~0.87× at long context
(depth 512). The remaining gap is one dominated kernel, not architecture.

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
  idle GPU). Long context = `tools/baseline/longctx_bench` (warm to `depth`, then
  time the tail) and `llama-bench -p 1 -n 64 -ngl 99`.

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

| depth | SonicBoom CUDA | llama.cpp CUDA | ratio |
|---|---|---|---|
| 512 | 34.55 tok/s | 39.87 tok/s | **0.87×** |
| 2048 | 34.78 tok/s | 37.56 tok/s | 0.93× |

SonicBoom is flat from 512→2048 (global attention is <1% of the step after the
split-KV rewrite). Measured at the pre-CUDA-Graphs/pre-T1-T5 commit
(`6867b2f`); the later CUDA Graphs work pushed depth-512 to ≈34.8 tok/s. The
long-context gap to llama.cpp (≈13% at 512) is the gemv + full-SWA attention,
which the T1–T5 fusions do not remove (they target the elementwise/quantize
glue, a smaller share at depth 512).

### 1.3 CPU decode (reference)

| path | tok/s |
|---|---|
| SonicBoom CPU q8_K (thread pool) | 4.37 |
| llama.cpp CPU 4 threads | 1.53 |
| llama.cpp CPU 12 threads | 2.02 |

SonicBoom's single straight-line spine beats llama.cpp's scheduler at equal
threads; the CPU path is the numerical oracle for the CUDA path, not a perf
target.

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
swapping them; the exact flip position shifts a few tokens run-to-run because
split-K `atomicAdd` accumulation is non-deterministic.)

### 3.2 SWA ring-buffer wraparound

The SWA KV cache is a fixed-size ring (`n_slots == sliding_window == 512`);
global layers span the full decode length. The split-KV decode kernel matches the
CPU oracle to **max_abs ~1e-8 / rms ~1e-9** at every probed position — the SWA
wraparound boundary (pos 511/512/513/1000 at n_slots=512) and global full-length
decode out to pos 4096. Wraparound (`% n_slots`) and the chunked/merged online
softmax are both exact.

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

1. **手写 (hand-written) — the q8_1 gemv is the floor and the lever.** The gemv
   is 64% and is already the full llama.cpp mmvq port (q8_1 int8 activation +
   warp-per-row iqs slice + `__shfl` scale broadcast + `__dp4a`, 3.84× over the
   fp32 kernels). Native-torch and MLIR have *no* ggml K-quant kernels, so this
   is necessarily hand-written. The next probe is ncu on the q8_1 kernel: if it
   is now DRAM-bound it is at the matmul floor (stop); if it is still
   L1/LSU-bound (as the fp32 kernels were — L1/TEX 97%, `math_pipe_throttle`
   0.2%), the remaining rewrite is a cooperative shared-memory tiling that loads
   each super-block once and broadcasts scales, which reduces *load traffic*
   rather than compute. The do-not-retry list (widen-loads no-op, dp4a-on-fp32,
   warp-per-row at small bpr) is recorded in the phase-6a notes.

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

# long-context steady decode (SonicBoom)
tools/baseline/longctx_bench --depth 512 --steps 64

# baseline (llama.cpp)
$LLAMA_ROOT/build-cuda/bin/llama-bench \
  -m models/gemma-4-E4B-it-Q3_K_M.gguf -p 1 -n 64 -ngl 99 -r 2

# full correctness suite
ctest --test-dir build
```
