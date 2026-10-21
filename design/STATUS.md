# Project Status

- Release: **v0.0.3** (`3a8c8d8`, fast-forwarded from `boom-pp`; base
  `5ee7b3f`), with the release documentation on top.
- Development branch: **`boom-next`** (base `ef6ed36` = the release + docs,
  code tip `3258d75`, unreleased): lossless speculative decoding, generated decode
  attention as the default, and the generated prefill-attention efficiency
  work.
- Scope and canonical numbers: [README](../README.md#numbers) and
  [Scope of v0.0.3](../README.md#scope-of-v003).
- Measurement record for the boom-next run: `/tmp/boom-next-report.md` (raw
  logs referenced there; not part of the repository).

## Completed

- **Boom IR is the only rewrite path.** The model emitters build Boom IR, the
  planner's rewrite phase runs MLIR passes over it, and `SONICBOOM_IR_PASS_DISABLE`
  is the ablation switch. The decode loop itself is lowered from `boom.program`
  to the step executor; the hand-written loop is behind `SONICBOOM_IR_PROGRAM=0`.
- **Generated GPU kernels** (MLIR `linalg → gpu → nvvm`, JIT-compiled and cached
  by shape signature): fused glue regions (norm, quantize, activation,
  residual, RoPE), FA-2 prefill attention, split-KV decode attention and a
  bf16 dense GEMM with an `mma.m16n8k16` core. The prefill attention now uses
  a 2–3-stage cp.async K/V pipeline (autotuned), padded shared-memory tiles
  and a conditional online-softmax rescale that is exactly `exp(0)` when
  skipped, so it is bit-identical.
- **Lossless speculative decoding.** A `boom.program` runs draft → (plain step
  or verify) → accept → emit → stop-check. The n-gram drafter is CPU-only and
  deterministic; the verify phase runs per-row decode kernels at their own
  positions with the same split constants as one plain step; rejected KV is
  rolled back by valid length, with SWA ring headroom so dead future slots
  cannot alias a live one. `SONICBOOM_SPEC=off|on|auto`; unset runs the V2
  startup probe (verify/plain ratio per SM, auto below 1.5, off otherwise; the
  sm_86 measurement is ~1.9, so the default is the plain loop), and an explicit
  value bypasses it. `auto` picks k from a measured c(k) curve and an
  acceptance EMA (α0 = 0.5).
  `test_spec_invariance` (k=1..7, both models, depth 0/512), `test_spec_rollback`
  (forced rejections 0/3/7, valid KV bitwise equal to plain decode) and the
  workload token hashes all pass. It is bit-identical to greedy by
  construction.
- **Per-GPU runtime autotune** for attention prefill/decode, MMQ tile and
  decomposition, and the dense GEMM vs cuBLASLt choice. One service, one disk
  cache (`~/.cache/sonicboom/tune/<gpu>-sm<arch>-<driver>-reduce-<mode>/<op>.json`,
  versioned JSON written by temp file + rename), keyed by GPU, operator and
  shape class — never by batch size or position. Choices need a 3% win to
  replace a default. `SONICBOOM_AUTOTUNE=0` uses the static defaults; `force`
  retunes. The generated-kernel (JIT candidate) tuner is on by default too:
  every probe is bit-identity gated against the static default, cold tuning is
  capped by one process-wide budget (`SONICBOOM_AUTOTUNE_BUDGET_MS`, default
  10 s, `<= 0` unlimited), and picks live in a versioned, flock-protected
  per-SM file (`~/.cache/sonicboom/autotune/sm_<NN>.bin`).
  `SONICBOOM_AUTOTUNE_JIT=0` disables only the JIT service. The MMQ static rule is upstream's
  smallest-J-minimizing-`ceil(n/J)` over `{8..128}` (including 56, 72, 88, 104
  and 120), and every one of those Js has a dispatch case.
- **MMQ at upstream build parity:** wide tiles use
  `__launch_bounds__(256,1)`, and the block order matches llama.cpp. At pp512,
  Q3_K is 112.1 ms against llama.cpp's 111.0 ms; aggregate MMQ is 167.5 vs
  164.0 ms.
- **Generated decode attention is the default** and reproduces the hand-written
  split-KV pair's token stream bit-for-bit (same chunk rule, FFMA rounding and
  reduction order); `SONICBOOM_ATTN_DECODE=own` selects the hand-written pair
  as the ablation and numerical reference.
- **ggml imported flash attention retired.** Attention is SonicBoom's own
  tiled kernel or generated code; the old `SONICBOOM_ATTN_IMPL=ggml` value is
  an alias for `auto`.
- **Scale folds:** the projection scale is folded into the dense GEMM epilogue,
  and the remaining scale into the `rms_norm_add` glue epilogue
  (`ScaleIntoDensePass`, `FoldScaleIntoGluePass`), removing launches from the
  prefill plan.
- Two model families run end to end (Gemma 4 E4B-it Q3_K_M, Qwen2.5-0.5B
  Q4_K_M) on CPU and CUDA, with prefix reuse, an OpenAI-compatible server and
  the Guile binding. `test_prefix_cache` is 14/14 bitwise; the separate
  `--long` enumeration adds three 8K-context cases (3/3).

## Current numbers

Medians of 3 process runs, same session, llama.cpp `43fe9c64281`, RTX 3050
sm_86 (full table and conditions in the README):

| model | tg64 | d512 | pp512 | reuse |
|---|---:|---:|---:|---:|
| Gemma 4 E4B Q3_K_M | 53.6 tok/s (1.20×) | 53.5 tok/s (1.21×) | 2084 tok/s (1.04×) | 57.4 ms |
| Qwen2.5-0.5B Q4_K_M | 345.9 tok/s (1.13×) | 307.0 tok/s (1.01×) | 15301 tok/s (1.09×) | 13.7 ms |

SM clock medians: 1912 MHz (Gemma), 1927 MHz (Qwen). pp512 kernel time on
Gemma is 266.9 ms / 1543 launches against llama.cpp's 242.1 ms / 1253
(1.103× kernel time, 1.231× launches); attention is 25.72 ms vs 8.25.

Speculation: `auto` chooses k=0 on every benchmark trace, so off and auto are
within noise (workload hashes identical; Gemma tg64 53.645 vs 53.533). Fixed
k=7 is correct but slower on these traces (Gemma tg64 25.2).

## Known issues

- **Prefill attention is the dominant remaining gap** at pp512: 25.7 ms
  against llama.cpp's 8.25 ms, about 70% of the residual kernel-time delta.
  The generated FA-2 is warp/scalar with no tensor cores; llama uses
  `flash_attn_ext_f16<256,256,16,4>` / `<512,512,16,4>`. The next lever is a
  generated tensor-core FA-2 with llama-class tiles.
- **Generated d64 prefill is gated off for token identity**, and it is the
  largest measured win not in the default path: the opt-in arm
  (`SONICBOOM_ATTN_TUNE_NONIDENTICAL`) selects b64/bc32/dps1/as2 and gives
  Qwen pp512 15301 → 16855 (+10.2%) and reuse 13.7 → 10.6 ms (−23%). It needs
  the hand tiled kernel's S reduction order, the same treatment C7 gave decode.
- **Generated decode wins end to end but loses in isolation at long context
  and d64**: 634 vs 246 µs at d512/4097 keys, 33.8 vs 15.0 µs for d64 at 513
  keys; it re-reads K per query head on a fixed grid. It still wins at depth
  512 (Gemma d512 +2.8%, Qwen +3.9%).
- **Generated decode versus prefill choice gap at d512.** The autotuner keeps
  codegen for the d512 prefill class, while the shape bench measures own about
  12% faster at n=512; the class probe's query length differs from the
  in-model call. The choice is stable and cached; the class definition is an
  open question.
- **The speculative `auto` policy is inert.** It proposes nothing on any of
  the four workload traces, including the copy-heavy code-edit trace (0 drafts
  over 11,247 emitted tokens). That looks like a drafter/history-coverage
  defect rather than genuine no-match, and it blocks any speculation win.
- **The opt-in exp2 softmax arm is not default-eligible**: it is
  performance-neutral and misses the d64 fp64-error bound (2.665e-4 vs
  2.301e-4).
- **Q6_K MMQ is disabled** (`TODO(q6k-mmq)`). Q6_K loads and decodes natively,
  but prefill dequantizes to fp16 and uses cuBLAS.
- **Sampling is greedy only.**
- **CUDA is tested on sm_86 only.** The CPU path is a correctness reference,
  not a fast path.

## Next

1. Token-identical generated d64 prefill (+10% Qwen pp512, −23% reuse).
2. Generated tensor-core FA-2 with llama-class tiles (the largest gap).
3. Fix the attention class probe (d512 prefill) and the speculative
   drafter/policy defect, then re-measure the workload traces.
4. Generated f32 dense GEMM, replacing the remaining chunked cuBLASLt
   projections and the 128-column blocking used for batch invariance.
5. Q6_K MMQ; further model families.
6. GPU-independent targeting (compute capability discovered at run time,
   SPIR-V) and the step-loop migration to Boom IR control flow.
