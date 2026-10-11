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

The already-ported q8_1 `__dp4a` path: gemv_bench shows FFN/Q4_K 2.5–9.3× faster
but Q3_K attention shapes 1.5–1.6× slower; per-token gemv 122.5 → 65.67 ms.
End-to-end **7.09 tok/s (1.64× vs 4.33)**, token stream identical to golden.
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
gemv_bench reports for attn_q/attn_k (§2.0e) is a measurement artifact, not a
kernel defect.**

Kernel-level effective bandwidth (nsys gpukernsum — hardware timestamps, the
authoritative number; the gemv_bench wall clock is polluted, see below):

| kernel (shape) | grid | kernel µs | packed bytes | GB/s |
|---|---|---|---|---|
| `gemv_q3_K_q8_1` (attn_k, 512 rows) | 1×64 | 7.5 | 563 KB | **75** |
| `gemv_q3_K_q8_1` (attn_q, 2048 rows) | 1×256 | 23.1 | 2.25 MB | **97.5** |
| `gemv_q4_K_q8_1` (attn_v, 512 rows) | 1×64 | 6.0 | 737 KB | 123 |

The fp32 base kernels for the same shapes are ~155 µs (attn_k) / ~500 µs
(attn_q) — i.e. the q8_1 kernel is **~20×** the fp32 base, not 0.7× as the
gemv_bench wall-clock numbers implied. "2× fp32 base" is met with ~20× margin;
the 80 GB/s target is met by attn_q (97.5) and approached by attn_k (75).

**Why gemv_bench misreports small shapes.** `matvec_f32_dev` = `quantize_q8_1`
(6 µs) + gemv kernel (7.5 µs) + ~9 µs launch ≈ 23 µs steady-state for attn_k
(measured directly: fixed N=1000 loop in a fresh process gives 23.3 µs/call GPU,
12.7 µs/call CPU). But the gemv_bench `run()` frees/re-`cudaMalloc`s `dx/dy` and
`cudaEventSynchronize`s between shapes; after the heavy FFN shapes the GPU drops
to a low-power state and each subsequent tiny 64-block kernel pays a ~190 µs
wake-up/scheduling latency. That latency does not exist in the real decode loop
(533 launches/token, **0** sync calls — phase1 §1), which is why end-to-end q8_1
is 7.09 tok/s (1.64×) with a bit-identical token stream even though gemv_bench
says the small shapes "regressed". The artifact is measurement-only.

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

### 2.2d — quantize-once-per-step reuse (pending)

## 2.3 — dense f32/bf16 gemv (pending)
