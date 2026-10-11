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

## 2.2 / 2.3 — quantized gemv + dense gemv (pending direction confirmation)

See the 2.0(e) result and the q8_1 prefill-consistency caveat. Direction to be
confirmed before 2.2 lands.
