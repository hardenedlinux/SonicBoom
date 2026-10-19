# Project Status

- Release: **v0.0.3**.
- Branch: `master`. The release content is `3a8c8d8` (fast-forwarded from the
  `boom-pp` branch; base `5ee7b3f`), with the release documentation on top.
- Scope and canonical numbers: [README](../README.md#numbers) and
  [Scope of v0.0.3](../README.md#scope-of-v003).
- Measurement record for this release: `/tmp/boom-pp-report.md` (raw logs
  referenced there; not part of the repository).

## Completed

- **Boom IR is the only rewrite path.** The model emitters build Boom IR, the
  planner's rewrite phase runs MLIR passes over it, and `SONICBOOM_IR_PASS_DISABLE`
  is the ablation switch. The old planner `Graph` rewrite path and its switch
  are gone.
- **Generated GPU kernels** (MLIR `linalg → gpu → nvvm`, JIT-compiled and cached
  by shape signature): fused glue regions (norm, quantize, activation,
  residual, RoPE), FA-2 prefill attention, split-KV decode attention and a
  bf16 dense GEMM with an `mma.m16n8k16` core.
- **Per-GPU runtime autotune** for attention prefill/decode, MMQ tile and
  decomposition, and the dense GEMM vs cuBLASLt choice. One service, one disk
  cache (`~/.cache/sonicboom/tune/<gpu>-sm<arch>-<driver>/<op>.json`), keyed
  by GPU, operator and shape class — never by batch size or position. Choices
  need a 3% win to replace a default. `SONICBOOM_AUTOTUNE=0` uses the static
  defaults; `force` retunes.
- **MMQ at upstream build parity:** wide tiles use
  `__launch_bounds__(256,1)`, and the block order matches llama.cpp. At pp512,
  Q3_K is 112.4 ms against llama.cpp's 111.2 ms; the aggregate MMQ delta is
  about 1%.
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
| Gemma 4 E4B Q3_K_M | 53.0 tok/s (1.18×) | 52.1 tok/s (1.18×) | 1990 tok/s (1.00×) | 65.2 ms |
| Qwen2.5-0.5B Q4_K_M | 344.7 tok/s (1.12×) | 296.4 tok/s (0.98×) | 15346 tok/s (1.09×) | 13.7 ms |

SM clock medians: 1912 MHz (Gemma), 1942 MHz (Qwen). pp512 kernel time is
277.8 ms / 1543 launches against llama.cpp's 242.1 ms / 1253 launches
(1.148× kernel time, 1.231× launches).

## Known issues

- **Prefill attention is the dominant remaining gap** at pp512: 37.3 ms
  against llama.cpp's 8.35 ms (about 81% of the residual kernel-time delta).
  Both SonicBoom paths now iterate only the valid causal/window-minimal KV;
  the generated FA-2 is warp/scalar and the hand-written tiled kernel is the
  faster one at d512. The next lever is a generated tensor-core (mma) FA-2.
- **Generated split-KV decode attention is the default.** Its chunk rule,
  fused-multiply-add rounding and reduction order match the hand-written pair,
  so the token stream is bit-identical; measured +1% (tg64) to +3-4% (decode
  at depth 512) end to end for Gemma 4 and Qwen2.5 on the RTX 3050.
  `SONICBOOM_ATTN_DECODE=own` keeps the hand-written pair as the ablation and
  numerical reference.
- **Generated decode versus prefill choice gap at d512.** The autotuner keeps
  codegen for the d512 prefill class, while a 256-query sweep measures the own
  kernel about 12% faster; the class probe's query length differs from the
  in-model call. The choice is stable and cached; the class definition is an
  open question.
- **Q6_K MMQ is disabled** (`TODO(q6k-mmq)`). Q6_K loads and decodes natively,
  but prefill dequantizes to fp16 and uses cuBLAS.
- **Sampling is greedy only.**
- **CUDA is tested on sm_86 only.** The CPU path is a correctness reference,
  not a fast path.

## Next

1. Generated tensor-core FA-2 for prefill attention (the largest gap).
2. Generated f32 dense GEMM, replacing the remaining chunked cuBLASLt
   projections and the 128-column blocking used for batch invariance.
3. Generated decode attention with a bit-identical summation order, or an
   explicit tolerance policy for decode.
4. Q6_K MMQ; further model families.
5. GPU-independent targeting (compute capability discovered at run time,
   SPIR-V) and the step-loop migration to Boom IR control flow.
