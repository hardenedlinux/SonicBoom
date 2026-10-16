# SonicBoom tuning-parameter inventory

**Created 2026-10-16** (pre-release hardening, before the MLIR planner
migration). Every hand-tuned constant on the CUDA path, in one place, with what
it *should* depend on, how its value was arrived at, and the values that were
measured and rejected.

This is an inventory, not a specification: the constants are correct today, and
several of them encode measurements that only hold for the shapes and hardware
in the "Measured on" column. Two uses are intended:

1. **A reviewer's map** — when a new model or GPU makes a kernel look wrong,
   this table says which knob exists and what would justify moving it.
2. **An autotuner's search space** — §10 lists which of these are
   shape-parameterised enough to be searched per model/shape rather than fixed.

"Measured on" is the desktop reference machine (RTX 3050 8 GB, sm_86, AMD
Ryzen 5 5500) with Gemma 4 E4B Q3_K_M (d_model 2560, gqa 4) and Qwen2.5-0.5B
Q4_K_M (d_model 896, gqa 7) unless stated. Per `CLAUDE.md`, a do-not-retry
entry states its conditions: when they change, it is re-evaluated, not obeyed.

Environment overrides (dev-time only, read once and cached) are noted per row;
they exist so these values can be re-measured without a rebuild
(`SONICBOOM_GEMV_COOP_MIN_BPR`, `SONICBOOM_MMQ_MIN_N`, `SONICBOOM_CUBLAS_MIN_N`,
`SONICBOOM_DENSE_CUBLAS_MIN_N`, `SONICBOOM_MMQ`, `SONICBOOM_CUBLAS`,
`SONICBOOM_CUDA_GRAPHS`).

---

## 1. Decode attention (split-KV over the KV ring)

`core/src/nn/cuda_elementwise.cu`. The window is the SWA ring (512) for SWA
layers and `n_ctx` for global layers; the split kernel derives the real start
and key count from `*pos` at run time, so grid/smem/chunk are fixed at the
window so the decode step stays CUDA-Graph-capturable.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `target_chunks` | `cuda_elementwise.cu:2304-2305` | `max(1, sm_count()*4 / n_heads_kv)` | SM count, `n_heads_kv` (parallelism available) | Chosen so `n_chunks` fills ≥ 4 waves of SMs at the reference shapes |
| `kMaxChunk` | `cuda_elementwise.cu:2320` | `16` keys | window (SWA 512), SM count, `n_heads_kv` | Swept against `n_ctx` 2048/8192/32768 and chunk caps 16/32 |
| `chunk` | `cuda_elementwise.cu:2321-2323` | `clamp(ceil(window/target_chunks), 1, kMaxChunk)` | window, target_chunks | derived |
| partial buffer | `:2330-2333` | `n_chunks * n_heads_q * (head_dim+2)` floats | window, chunk, head_dim | sized, not tuned |
| smem | `:2337` | `group_q * chunk * sizeof(float)` | chunk, group_q | sized, not tuned |
| split launch | `:2338-2339` | grid `n_chunks*n_heads_kv`, block `group_q*32` | chunk, heads | derived |
| merge launch | `:2343` | grid `n_heads_q`, block `head_dim` | head_dim (1 thread per output element) | derived |
| `#pragma unroll 4` | `:536` | 4 keys in flight | head_dim/registers | not swept |
| `dims_per_lane`, `acc[16]` | `:574-575` | `head_dim >> 5`, 16 | head_dim | sized by the lane-owned slice |

**Known bad (do-not-retry as-is):**

- **Uncapped chunk** (`kMaxChunk = ∞`, i.e. the pre-fix code): chunk grows with
  `n_ctx` (820 at n_ctx 32768), so at short depth one block per KV head is
  active and the split kernel is latency-bound. Measured split-KV
  **1.95 → 9.32 → 20.36 ms** at n_ctx 2048/8192/32768; token rate 45.2 → 38.5.
  With the cap: flat 2.72 ms / 46.3 tok/s. Condition: any model with a windowed
  KV ring and `n_ctx` ≥ 8192.
- **`kMaxChunk = 32`**: a wave-fill local minimum at the reference geometry
  (16 chunks → 32 blocks, 1.6 blocks/SM) — measured **~2.4× slower split**
  (3.20 vs 1.29 ms at depth 512). Condition: RTX 3050 (20 SMs), `n_heads_kv`
  2 (Gemma). On a much larger GPU the wave-fill argument would move and this
  should be re-swept.

## 2. Prefill flash attention (tiled tensor-core kernel)

`core/src/nn/cuda_elementwise.cu`. One instantiations table, keyed by
`(head_dim, gqa)`; anything else falls through to the scalar chunked kernel.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| template | `:2396` | `flash_attention_tiled_kernel<HEAD_DIM, BK, NW, QB, GQA>` | (head_dim, gqa) | — |
| instantiation | `:2450-2453` | `<256, 64, 2, 32, 4>` | Gemma-family SWA: gqa 4, head 256 | nsys: prefill FA ≈ 5.6% at n=512 |
| instantiation | `:2454-2457` | `<512, 32, 4, 16, 4>` | Gemma-family global: gqa 4, head 512 | same |
| instantiation | `:2458-2461` | `<64, 64, 2, 32, 7>` | SwiGLU families: gqa 7, head 64 (Qwen2.5-0.5B) | added with the qwen-opt line, measured on pp512 |
| fallthrough | `:2464-2467` | `flash_attention_chunk_pos_kernel`, block `n_heads_q*32` | any other (head_dim, gqa) | correct-but-slow; requires `n_heads_q ≤ 32` (one warp per query head) |
| `FA_BK` | `:669` | `32` keys/tile | head_dim, smem budget | not swept |
| `__launch_bounds__` | `:1086` | `NW*GQA*(QB/16)*32` | the template's own geometry | derived, ≤ 1024 threads |
| `V_STRIDE = BK+2` | `:1093` | 34 | BK | padding against bank conflicts |

**Known bad (do-not-retry as-is):**

- **fp16 O accumulator** in the tiled FA: incorrect (prefix 4097 diverges) *and*
  slower (2.35 s vs 1.74 s). The fix must be a two-phase QK^T / PV split, not a
  register trick. Condition: this kernel structure and these shapes.
- **Splitting QK^T keys across all warps**: +14% (1.225 s vs 1.072 s) — the
  kernel is tensor-core-bound, not warp-bound. Condition: sm_86, these shapes.

## 3. Dense / matvec (non-quantized)

`core/src/nn/cuda_elementwise.cu`.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `kGemvWarpsPerBlock` | `:1417` | `8` (256 threads) | rows | chosen to keep SMs busy for few-row shapes |
| `kGemvTargetWarps` | `:1418` | `1024` | rows, SM count | `gemv_k_split(rows) = rows >= 1024 ? 1 : ceil(1024/rows)` |
| `kDenseCublasWidth` | `:1629` | `128` columns | *nothing* — deliberately fixed | see below |
| `kDenseCublasMinN` | `:2615` | `1` | *nothing* — deliberately fixed | see below |
| `kDenseCublasWorkspaceBytes` | `:2630` | 4 MiB | largest cuBLAS call | sized so cuBLAS never lazily allocates inside a capture |

`kDenseCublasWidth` / `kDenseCublasMinN` are **numerics** constants, not
performance ones: every dense cuBLAS call is shaped `(m=128, k=cols,
lda=ldc=128)` with `n == 1` packing one real column plus 127 pad columns, so the
result does not depend on the batch size or on the column offset. That
independence is what makes prefix-cache reuse bit-exact (up to fp16 rounding).
Measured effect of the fix: dense 12.2 s → 1.5 s at 4100-token prefill. Do not
"optimise" these two away.

## 4. Quantized GEMV (decode, K-quant + legacy 32-element)

`core/src/quant/cuda_quant_matmul.cu` (dispatch) and
`core/src/quant/ggml_derived/mmvq_q8_1.cuh` (kernels). All decode gemv paths are
warp-per-row, 8 warps per block.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `_coop` block | `cuda_quant_matmul.cu:1227-1228` | `dim3(32,8)`, grid `(1, ceil(rows/8))` | rows | measured ~1.6× win on the large FFN shapes |
| `_base` block | `:1255-1256` | 256 threads, one per (row, super-block), atomic accumulate | rows, blocks_per_row | — |
| ILP accumulators | `:224/269/312/356` | q3_K `acc[8]`, q4_K/q5_K `acc[4]`, q6_K serial | super-block count per row | hides the block-load latency on small attention shapes |
| `kCoopMinBlocksPerRow` | `:1162` | `20` blocks/row | blocks_per_row (= cols / block_size), L1 capacity | crossover between two measured clusters (below); env `SONICBOOM_GEMV_COOP_MIN_BPR` |
| `GEMV_BATCH_T` | `mmvq_q8_1.cuh:593` | `8` positions per warp | batch n, register budget | weight block read once and reused across 8 columns (prefill dp4a path) |
| batched block | `cuda_quant_matmul.cu:1398-1399` | `dim3(32,8)`, grid `(1, ceil(rows/8))` | rows | — |
| legacy gemv | `:1184-1219` | scalar loops, no dp4a | block size 32 | "the block is only 32 elements, there is nothing to amortize" |

**Known bad (do-not-retry as-is):**

- `_base` one-thread-per-block on a large `blocks_per_row`: ffn_down (40
  blocks/row) spills x out of L1 and serializes a 40-way atomic fan-in —
  **788 µs** where the cooperative kernel does **496 µs**. Condition: sm_86,
  L1 capacity, this row count.
- Hybrid dispatch by `blocks_per_row` (the `kCoopMinBlocksPerRow` mechanism)
  is not the "~15.6%" figure that circulated in earlier notes — no constant or
  measurement in the tree carries that number; the two clusters it sits between
  are above.

## 5. Activation quantize + fused glue emitters

`core/src/quant/cuda_quant_matmul.cu`; `kQuantizeBlockThreads` in
`mmvq_q8_1.cuh`; mmq geometry in `ggml_derived/mmq.cuh`.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `kQuantizeBlockThreads` | `mmvq_q8_1.cuh:128` | `256` (8 warps) | block count | one warp per 32-element q8_1 block, so a decode row is short instead of latency-bound |
| `q8_1` fused norm | `cuda_quant_matmul.cu:1491/1541` | `__launch_bounds__(1024)`, `<<<1,1024>>>` | row length (`n`) | block pinned so the reduction tree matches the standalone norm and `inv` is bit-identical |
| `q8_1` fused quant loop | `:1516` | one warp per 32-element block | n/32 | two coalesced 128 B loads instead of 32 strided lines per warp |
| `gelu×up→q8_1` | `:1579` | 256 threads, 1 thread per 32-el block | element count | — |
| `silu×up→q8_1` | `:1596/1615-1616` | one warp per 32-el block, `kQuantizeBlockThreads` | nblocks | a row (ff 4864 = 152 blocks) fills the GPU instead of one thread's serial chain |
| `rms_norm_inv_kernel` | `:2064`, launch `:2131` | `__launch_bounds__(256)`, `<<<n,256>>>` | dim, rows | 256 threads = the standalone batched norm's reduction tree (`rms_heads_fused_kernel`), so the fused and unfused emitters write identical bytes |
| `scale_quantize_mmq_q8_1_kernel` | `:2097`, launch `:2134-2140` | `__launch_bounds__(128)`, grid `(n, ceil((dim/128)/4))` | dim (multiple of 128), n | the standalone `quantize_activation` geometry: 128 threads, one 128-element block per warp |
| `gelu_mul_quantize_mmq_q8_1_kernel` | `:2150`, launch `:2177-2183` | `<<<n, 256>>>`, 128-el block per warp | dim, n | same geometry |
| `silu_mul_quantize_mmq_q8_1_kernel` | `:2194`, launch `:2223-2229` | `<<<n, 256>>>` | dim, n | same geometry |
| `CUDA_QUANTIZE_BLOCK_SIZE_MMQ` | `mmq.cuh:99` | `128` | the q8_1_mmq block (fixed) | not tuned; the block *is* 128 values |
| `kCublasWorkspaceBytes` | `:571` | 16 MiB | largest fp16 GEMM | fixed so no lazy alloc inside a capture |
| `kWeightTailPad` | `:606` | `256` bytes | K-iteration width (256 values) | mmq reads whole 256-value K-iterations; a non-multiple-of-256 K over-reads up to 4 legacy blocks |

**Known bad (do-not-retry as-is):**

- **Single-kernel fused rms_norm→mmq_q8_1**: one 1024-thread block per token
  with one 128-element block per warp for the quantize. At Qwen's d_model 896
  only 7 of its 32 warps had quantize work, and its 1024-thread
  double-precision reduction cost ~4× the 256-thread one; measured **45 GB/s**
  against the standalone quantizer's **226 GB/s**, and *slower than not fusing
  at all* at dims 896–2560 (Stage C microbench, n=512: 89.5 µs vs 52.7 µs at
  896). Condition: this kernel shape. The two-phase form above replaced it:
  41.2 µs at 896 (1.28× the unfused pair), 130.7 µs at 4096 (1.45×), with
  byte-identical output.
- **Deleting the split's phase A and folding `inv` into phase B** would restore
  the 1024-thread tree and both regressions above.
- `Q6_K` through mmq: `TODO(q6k-mmq)` at `:2251` — incoherent output on Gemma
  Q4_K_M, disabled (Q6_K falls back to the cuBLAS fp16 path). Not a tuning knob;
  a correctness blocker.

## 6. MMQ tile selection (prefill int8 tensor-core matmul)

`core/src/quant/ggml_derived/mmq.cuh` — ported from ggml's
`mmq-config-ampere.cuh` with `fallback = false`.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `Js[]` | `:1052` | `{8,16,24,32,40,48,64,80,96,112,128}` | batch n | `select_J(n)` takes the smallest J that minimises `ceil(n/J)` |
| `I` | `:96` | `128` rows/tile | M (fixed) | the q8_1_mmq block is 128 values |
| `MMQ_ITER_K` | `:95` | `256` | weight K-quant block (256) | fixed |
| `NWARPS`, `WARP_SIZE` | `:97-98` | `8`, `32` | — | 256 threads/block |
| `rows_per_warp` | `:559/628/694/781/858` | `(J >= 48 && J % 16 == 0) ? 32 : 16` | J | register pressure per output tile |
| `threads_per_row` | `:506-507` (q8_0), `:274-275` (q3_K), `:319-320` (q4_K), `:405-406` (q6_K), `:453-454` (q5_0) | q3_K 16, others 32 | MMQ_ITER_K / (4·QR) | `q8_0` deviates on purpose (see comment) |
| `SRAM_STRIDE_Q3_K / _Q8_1 / _LEGACY_32 / _Q6_K` | `:107-111` | 84 / 68 / 76 / 76 int32 | the quant layout | layout arithmetic |
| smem | `:1069` | `(PAD(J*36, 256) + I*sram_stride)*4` bytes | J, quant type | set via `cudaFuncSetAttribute` |
| `mmq_q8_planes` | `:1014-1017` | `p + (p & 1)` | K | pads to an even plane count so the 256-value K-iteration is whole |
| `mmq_q8_blocks` | `:1023` | `planes*n + I` | K, n | the `+I` is the J_max=128 tail over-read guard |

## 7. Dispatch thresholds (which algorithm runs)

`core/src/quant/cuda_quant_matmul.cu`.

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `kMmqMinN` | `:1339` | `2` | *numerics*, not the perf crossover | microbench crossover is n≈6 (dp4a wins below n=6); fixed at 2 so a prefill position's algorithm does not change with segment size (the dp4a batched gemv and mmq disagree numerically). Cost at n=2 is ~0.13 ms/matmul — negligible |
| `kCublasMinN` | `:1306` | `32` | batch n | fp16 dequant + upload is not amortized over fewer columns |
| `mmq_enabled()` / `cublas_enabled()` | `:1322+` | env `SONICBOOM_MMQ` / `SONICBOOM_CUBLAS` | — | dev switches |

## 8. Planner and model-shape constants

| constant | file:line | value | should depend on | how measured |
|---|---|---|---|---|
| `kMmqBlockBytes` | `planner/rewrites.cpp:62` | `144` | the q8_1_mmq block layout | sizing check for the pass's f32 buffer |
| `kMmqTailGuardBlocks` | `:63` | `128` | J_max = I | tail over-read guard |
| plane rounding | `:70-72` | `dim/128 + ((dim/128) & 1)` | dim (multiple of 128 enforced) | layout |
| `ds` attribute | `model/gemma4_plan.cpp:559-568` | `1` for Q4_K/Q5_K, `0` for every other format with an mmq activation layout, `-1` for Q6_K | whether the *weight format* has an mmq activation layout | capability, not speed: the fusion passes read `ds` only to decide eligibility (see the emitter rows in §5 for why speed is not part of this decision) |
| `kDefaultPrefillChunk` | `model/gemma4_plan.cpp:91` | `512` tokens | model (sliding window) | any positive value is correct; 512 matches the segment size the windowed models use |
| chunk selection | `:163`, `:1356-1357` | `sliding_window > 0 ? sliding_window : kDefaultPrefillChunk` | SWA ring size | a segment is then a wrap-free run |
| mmq gate | `:1386-1387` | `nseg >= mmq_min_n()` | dispatch threshold | keeps the fused norm+quantize passes off for 1-token segments |
| CUDA-graph replay | `:1431` | `use_graphs && n > chunk && start_pos == 0` | prompt length, segment size | prefill capture+replay measured ~0.8% (noise) — prefill is matmul-GPU-bound, not launch-bound |
| `kAlignment` | `planner/device_pool.cpp:33` | `256` bytes | device requirement | not tuned |
| `kFallbackHostBudgetBytes` | `planner/resource.cpp:177` | 1 GiB | host RAM | not tuned |

## 9. Summary of "do not retry" (with conditions)

Full entries above; the list a future optimizer should read before starting:

1. Uncapped decode-attention split chunk (windowed KV, n_ctx ≥ 8192).
2. `kMaxChunk = 32` (RTX 3050, n_heads_kv 2, SWA 512).
3. fp16 O accumulator in the tiled flash attention (this kernel structure).
4. QK^T keys split across all warps in FA (sm_86, these shapes).
5. `_base` gemv on large blocks_per_row (sm_86, L1-bound).
6. The single-kernel fused rms_norm→mmq_q8_1 emitter (any dim ≤ 4096).
7. Q6_K through mmq (correctness blocker, not a tuning question).
8. Moving `kDenseCublasWidth` / `kDenseCublasMinN` for speed (they are numerics
   constants; prefix-cache bit-exactness depends on them).

## 10. Autotuner search space

Candidate parameters are the ones whose correct value is a function of *shape*
(batch, dim, rows, head geometry) rather than of the model's numerics:

- `select_J` over `Js[]` per (n, quant type, M) — the only one already
  shape-parameterised; the natural first search space.
- `kCoopMinBlocksPerRow` per (blocks_per_row, L1 size) — two measured clusters
  today, a 1-D threshold.
- `kMaxChunk` / `target_chunks` per (window, sm_count, n_heads_kv).
- The `(head_dim, gqa)` instantiation table (currently hand-written; the
  fallthrough exists so a wrong key is slow, not wrong).
- `GEMV_BATCH_T` per batch n and register budget.
- `kGemvWarpsPerBlock` / `kGemvTargetWarps` per rows.
- `select_J`-style selection for the batched mmq vs dp4a vs cuBLAS thresholds —
  *except* `kMmqMinN`, which is fixed by numerics and must stay out of the
  search space.

Not searchable (correctness/numerics, listed so a tuner does not "win" by
changing them): `kMmqMinN`, `kDenseCublasWidth`, `kDenseCublasMinN`, the
`kWeightTailPad`/plane-padding arithmetic, `mmq_q8_planes`, and the reduction
trees of the norm emitters (byte-identity with the unfused path).
