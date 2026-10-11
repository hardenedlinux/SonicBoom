# CUDA decode — Phase 1 measurement report

**Date: 2026-10-11.** No production code changed in this phase; the only code
change is `tools/baseline/gemv_bench.cpp` (measurement tool: effective GB/s +
per-tensor quant types). This report is the data baseline that Phase 2 (kernel)
and Phase 3 (orchestration) build on.

## Hardware / target

- RTX 3050 (8 GiB, sm_86), CUDA 13.0, ~224 GB/s DRAM.
- Model: Gemma 4 E4B-It Q3_K_M (Unsloth), 42 blocks, d=2560, ff=10240.
- End-to-end: SonicBoom spine-cuda **4.33 tok/s** (test_generate, 64 steps) vs
  llama.cpp **42.63 tok/s** → **9.8× gap**. (benchmark.md's earlier 6.20 tok/s is
  clock/thermal variance on the same machine.)

## 1. nsys GPU kernel breakdown (the authoritative relative data)

`nsys profile --stats=true` over a full `test_generate` run. Percentages are
fractions of total GPU kernel time.

| % | kernel | notes |
|---|---|---|
| 28.2 | `gemv_q4_K_coop` | ffn_down (bpr=40 → warp-per-row) |
| 24.8 | `gemv_q3_K_base` | ffn_gate/up + attn_q/k + **lm_head** |
| 15.0 | `rms_sum_sq_kernel` | **grid 1×1×1, block 1×1×1 — single thread** |
| 12.5 | `gemv_dense_f32` | inp_gate/proj + lm_head f32 |
| 11.9 | `gemv_bf16` | per-layer model proj (bf16 weights) |
| 3.1 | `gemv_q4_K_base` | attn_v/attn_output |
| 1.5 | `gemv_q5_K_coop` | ffn_down/attn_v blocks 0-1 |
| 1.2 | `decode_attention_kernel` | — |
| ~2 | rms_heads_* / rope / elementwise / add | many tiny kernels |

**Quantized gemv total ≈ 57.6%.** `rms_sum_sq` + `gemv_dense_f32` + `gemv_bf16`
are three *secondary* costs (≈39% combined) that were **not** in the given
analysis.

### API counts (sync / transfer / alloc)

| API | calls | total time | reading |
|---|---|---|---|
| `cudaDeviceSynchronize` | **0** | — | no per-op sync |
| `cudaStreamSynchronize` | **0** | — | no per-op sync |
| `cudaMemcpy` | 21,411 | 13.74 s | activation H2D/D2H + one-time weight uploads |
| `cudaMemset` | 55,466 | 0.96 s | y-zero-fill before every `_base` gemv |
| `cudaMalloc` | 2,082 | 0.19 s | one-time weight upload + pool growth (median 4.8 µs) |
| `cudaFree` | 11 | ~0 | one-time teardown |

## 2. gemv_bench — effective GB/s per actual quant type

The GGUF was dumped to get the **real** per-tensor types (Unsloth Q3_K_M):
q3_K for `token_embd`/`attn_q`/`attn_k`/`ffn_gate`/`ffn_up`; q4_K for
`per_layer_token_embd`/`attn_output`/`attn_v`/`ffn_down`; `attn_v`/`ffn_down`
are q5_K in blocks 0-1 only. Effective bandwidth = packed weight bytes /
kernel time (the correct metric for n=1 gemv, *not* GFLOPS).

| tensor | shape (cols→rows) | type | µs | **GB/s** |
|---|---|---|---|---|
| ffn_gate | 2560→10240 | Q3_K | 346.4 | 32.5 |
| ffn_up | 2560→10240 | Q3_K | 346.2 | 32.5 |
| ffn_down | 10240→2560 | Q4_K | 994.5 | 14.8 |
| attn_q | 2560→2048 | Q3_K | 78.3 | 28.8 |
| attn_k | 2560→512 | Q3_K | 24.4 | 23.1 |
| attn_v | 2560→512 | Q4_K | 24.5 | 30.1 |
| attn_output | 2048→2560 | Q4_K | 82.4 | 35.8 |
| ffn_down (blk 0-1) | 10240→2560 | Q5_K | 1053.9 | 17.1 |
| attn_v (blk 0-1) | 2560→512 | Q5_K | 25.7 | 35.1 |

**Effective bandwidth is 14.8–35.8 GB/s — 7–16% of the 224 GB/s ceiling.** The
gemv is **not** DRAM-bandwidth-bound; it is latency/occupancy-bound (consistent
with the prior ncu finding of L1/TEX ~97%, math_pipe ~0.2%). The task's given
premise "n=1 gemv is bandwidth-bound" is **refuted by measurement**.

> **This table supersedes the original (corrected 2026-10-11).** The original
> reported attn_q 505.8 µs, attn_k 159.2 µs, and attn_v-Q5_K 367.1 µs — those
> three numbers were a gemv_bench measurement artifact (idle-clock wake-up +
> a single-call adaptive iteration count), not real kernel time. The corrected
> fp32 values here are from the fixed gemv_bench (`SONICBOOM_GEMV_F32=1`); see
> phase2 §2.2b for the root cause and the fixed methodology.

Two anomalies that shape Phase 2 (REVISED — the two original anomalies were
mostly that same artifact):

1. **Q3_K base degrades *mildly* at small shapes.** Per-block throughput:
   ffn_gate/up (10,240 rows) ≈ 296 blocks/µs, attn_q (2,048 rows) ≈ 261
   blocks/µs, attn_k (512 rows) ≈ 210 blocks/µs — a ~1.4× drop, not the ~7×
   "collapse" the original table implied (the original 40 blocks/µs for attn_q
   was the artifact). Q4_K base is flat (≈ 206–209 blocks/µs from 10,240 to 512
   rows).
2. **Q3_K vs Q4_K base at the *same* shape** (2560→512): 24.4 µs vs 24.5 µs —
   essentially **equal**. The original 159.2 vs 28.0 (a "5.7× gap") was the
   artifact; there is no large Q3_K-vs-Q4_K base gap at equal shape. (The
   register-bound occupancy difference in phase2 §2.2f is real but does not
   translate into a 5.7× time gap at these shapes.)

## 3. matvec_f32_dev thread mapping (what the code actually does)

`core/src/quant/cuda_quant_matmul.cu`. Dispatch is on `blocks_per_row = cols/256`:

- **`_coop` (bpr ≥ 20, e.g. ffn_down=40):** warp-per-row. block `(32, 8)`, grid
  `(1, num_rows/8)`. The 32 lanes cover a row via the iqs-slice mapping — lane's
  `iqs` picks one 16/32-element slice sharing a scale, `ib = lane>>…` picks which
  block — so packed-weight loads coalesce/broadcast within the warp and a
  `__shfl_down` reduction produces the row dot (no per-block atomicAdd).
- **`_base` (bpr < 20, e.g. attn_* = 8-10, ffn_gate/up = 10):** **one thread per
  super-block**, not per row. block `(256)`, grid `(num_blocks/256)`; 8-way ILP
  accumulators, `atomicAdd(&y[row], s)` fan-in. `cudaMemset(yp, 0, …)` runs
  before every `_base` launch (→ the 55k memsets). Dequant is **on-the-fly**
  fused into the FMA — no full row is ever materialized in f32.

Activation handling (default fp32 path): `x` stays f32 and is re-read per block;
there is **no** q8_1 quantization on the default path. A complete q8_1 mmvq port
(`quantize_q8_1` + `__dp4a` `vec_dot_*_K_q8_1` + warp-per-row kernels) already
exists behind `SONICBOOM_GEMV_Q8_1=1` — Phase 2 is *validate/enable/improve* it,
not a from-scratch rewrite.

## 4. Suspect resolution (the given (a)–(d))

| suspect | verdict |
|---|---|
| (a) one thread per row, uncoalesced, SM under-filled | **Refuted in part.** `_base` is one-thread-per-super-block, `_coop` is warp-per-row+coalesced. But "SM under-filled" is **true** for attention shapes (few rows → few blocks → low occupancy). |
| (b) q8_K activation quant on GPU then f32 dot | **Off by default.** The q8_1 path exists but is gated; the default re-reads f32 x. |
| (c) dequant whole row to f32 then dot | **Refuted.** Dequant is fused on-the-fly; no f32 row materialized. |
| (d) different slow branches per type | **Confirmed.** q3_K base is the slow branch (see anomaly #1/#2). |
| RuntimeExecutor `cudaDeviceSynchronize` per task | **Refuted.** 0 sync calls. |
| device_pool `cudaMalloc/cudaFree` in hot path | **Refuted.** 11 cudaFree total; cudaMalloc is one-time setup. |

## 5. Conclusions → Phase 2 / Phase 3

1. **The gap is kernel-bound, not orchestration-bound.** GPU spends ~95% of its
   time in kernels; quantized gemv is 57.6% of that. There are **zero** sync
   calls and negligible hot-path allocation. This **refutes** the coexecution
   doc's §7 conclusion ("per-step host↔device handshake is the bottleneck").
2. **Primary Phase 2 target: the q3_K base kernel** (ffn_gate/up + attn_q/k +
   lm_head ≈ 24.8% + the q3 part of attn). It collapses at low occupancy and is
   5.7× slower than q4_K base at equal shape. Fix = q8_1/dp4a path (already
   present) or a register/coalescing rewrite of the fp32 base kernel.
3. **Secondary Phase 2 targets:** `rms_sum_sq` (15%, single-thread reduction over
   n=2560), `gemv_dense_f32` (12.5%), `gemv_bf16` (11.9%). These were not in the
   given analysis and are individually as large as any single attention gemv.
4. **Phase 3 targets remain real but secondary:** 55k `cudaMemset` y-zero-fills,
   and the 21k `cudaMemcpy` activation transfers — worth removing once the
   kernels are fixed (memset disappears if `_base` is replaced by `_coop`/q8_1).

## Reproduce

```sh
# end-to-end (SonicBoom)
SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
  SONICBOOM_GEN_STEPS=64 ./build/tests/core/test_generate

# profile
nsys profile --stats=true -o /tmp/sb_phase1 ./build/tests/core/test_generate
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i /tmp/sb_phase1.qdstrm -f
nsys stats --report gpukernsum /tmp/sb_phase1.nsys-rep
nsys stats --report cudaapisum /tmp/sb_phase1.nsys-rep

# operator-level gemv (effective GB/s, per-tensor types)
bash tools/baseline/build_gemv_bench.sh && tools/baseline/gemv_bench
```
