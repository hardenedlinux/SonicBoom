# SonicBoom v0 benchmark — Gemma 4 E4B-It Q3_K_M

**Test date: 2026-10-07.**

**Status: v0 — the first runnable version.** Everything below is the *initial*
baseline: all functionality is in place and bit-checked, but **no performance
work has been done**. These numbers are the starting point, not a target or a
claim. Optimization is explicitly deferred until the v0 correctness is settled
(per the "compute through the core first, optimize later" principle).

## Scope

- **Model:** `models/gemma-4-E4B-it-Q3_K_M.gguf` (3.76 GiB, 7.52 B params, 42
  blocks, d=2560, ff=10240, n_heads_q=8, n_heads_kv=2, shared KV).
- **SonicBoom:** C++23 / g++-13, `SONICBOOM_USE_CUDA=ON`. Decode runs the fused
  planner spine (`generate_spine` / `generate_spine_cuda`); CPU prefill runs the
  batched prefill graph through the spine (`generate_prefill_spine`).
- **Baseline (llama.cpp):** `$LLAMA_ROOT` (local checkout), HEAD `43fe9c642`,
  CUDA build (`build-cuda/bin`). Same GGUF, same machine.
- **cuBLAS reference (PyTorch):** `torch 2.14.1+cu130`, used only for the
  operator-level matmul comparison — not a runtime dependency.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 (8 GiB, compute 8.6) |
| Host | Linux 6.8, x86-64 |
| CUDA | 13.0 |

## Correctness gate (what "runnable" means)

- **Decode:** `generate_spine` / `generate_spine_cuda` reproduce the
  llama.cpp multi-token golden stream `236761 107 236769 236776 …` (multi-token
  KV cache; the single-token oracle yields `236761 236761 …` by design).
- **CPU prefill == per-token spine:** the batched prefill graph and the
  per-token decode produce the identical continuation stream (verified at prompt
  lengths n = 1, 3, 5).
- **Full suite:** `ctest --test-dir build` → **48/48 pass**.

## 1. End-to-end decode (tokens/sec)

Autoregressive greedy decode, single seed token + N decode steps (no tokenizer).

| path | SonicBoom | llama.cpp | gap |
|---|---|---|---|
| CUDA decode | **6.20 tok/s** (spine-cuda, 64 steps) | **42.63 tok/s** (tg64, ngl 99) | 6.9× |
| CPU decode | **3.90 tok/s** (spine, 64 steps) | **6.54 tok/s** (tg32, 4 threads) | 1.7× |

Reference paths (single-token oracle, no KV cache — not the production path):
SonicBoom `[cpu q8_K]` 4.21 tok/s, `[cuda]` 6.66 tok/s.

llama.cpp numbers are `llama-bench` (`-p 1 -n 64 -ngl 99` / `-p 1 -n 32 -ngl 0
-t 4`); SonicBoom numbers are `test_generate` at `SONICBOOM_GEN_STEPS=64`.

## 2. Operator-level: quantized gemv vs cuBLAS matmul

`gemv_bench` (SonicBoom `cuda::matvec_f32_dev`, pure device kernel time) vs
`cublas_bench.py` (`torch.matmul`, pure kernel time), same shapes and layout
`W[rows, cols] @ x[cols]`.

### n=1 (decode shape) — latency per op (µs)

| shape (cols→rows) | Q3_K | cuBLAS fp16 | cuBLAS fp32 | Q3_K / fp16 |
|---|---|---|---|---|
| ffn_gate_up 2560→10240 | 329 | 303 | 604 | **1.09×** |
| ffn_down 10240→2560 | 582 | 322 | 583 | **1.81×** |
| attn_q 2560→2048 | 504 | 67 | 123 | **7.5×** |
| attn_o 2048→2560 | 475 | 69 | 125 | **6.9×** |
| attn_kv 2560→512 | 159 | 16 | 34 | **9.9×** |

Throughput at these points: Q3_K runs 159 GF (ffn_gate_up) down to ~17 GF
(attn_kv); cuBLAS fp16 is flat ~155–173 GF (bandwidth-bound); cuBLAS fp32 is
~77–90 GF.

**Reading:**

- **FFN matrices (the dominant matmul work) — quantization already pays off.**
  `ffn_gate_up` Q3_K ≈ cuBLAS fp16 (1.09×) and **beats fp32 1.8×**, because Q3_K
  reads 0.43 B/weight vs fp16's 2 B/weight and the large shapes are
  bandwidth-bound.
- **Attention projections — the weak spot.** `attn_q/o/kv` are 7–10× behind
  cuBLAS fp16: small shapes (2048/512 rows, 10 blocks/row) are kernel/L1-bound,
  not bandwidth-bound, so dequant overhead dominates.

### n=32 (prefill shape) — cuBLAS switches to tensor-core GEMM (µs)

| shape (cols→rows) | cuBLAS fp16 (GF) | cuBLAS fp32 (GF) |
|---|---|---|
| ffn_gate_up 2560→10240 | 321 µs (5229) | 1485 µs (1130) |
| ffn_down 10240→2560 | 359 µs (4678) | 669 µs (2508) |
| attn_q 2560→2048 | 74 µs (4564) | 163 µs (2064) |
| attn_o 2048→2560 | 71 µs (4695) | 150 µs (2241) |
| attn_kv 2560→512 | 21 µs (4044) | 62 µs (1347) |

SonicBoom's batched path (`cuda::matmul_f32`) is **per-column gemv** ("one gemv
per column" in `quantized_matmul_cuda.h`), so n=32 ≈ 32× the n=1 cost — it never
reuses the weight across columns. cuBLAS tiles the GEMM and reuses the weight,
which is why prefill (batched prompt) is where the largest gap sits.

## Conclusions (v0, unoptimized)

1. **CPU decode** is within 1.7× of llama.cpp — a normal, tunable gap.
2. **CUDA decode** is 6.9× behind llama.cpp. The operator data locates the gap:
   FFN quantized gemv is already competitive with cuBLAS; the attention
   projections (7–10×) and the per-column-gemv prefill (no weight reuse) are the
   structural costs. These are kernel-level, not correctness, issues.
3. **No optimization has been attempted.** This is the honest starting point for
   the later "make it fast" phase.

## Reproduce

```sh
# end-to-end decode (SonicBoom)
SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
  SONICBOOM_GEN_STEPS=64 ./build/tests/core/test_generate

# end-to-end decode (llama.cpp baseline; $LLAMA_ROOT = the local checkout)
$LLAMA_ROOT/build-cuda/bin/llama-bench \
  -m models/gemma-4-E4B-it-Q3_K_M.gguf -p 1 -n 64 -ngl 99 -r 2

# operator-level gemv (SonicBoom)
tools/baseline/gemv_bench

# operator-level matmul (cuBLAS / PyTorch)
models/.venv/bin/python tools/baseline/cublas_bench.py
```
