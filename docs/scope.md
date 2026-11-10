# Scope

SonicBoom v0.0.6 is a working runtime, not a drop-in replacement for
llama.cpp. It adds speculative decoding through every entry point — the C ABI,
Guile and the OpenAI-compatible server — to the v0.0.5 runtime. Its limits:

- **Four benchmarked language-model families:** Gemma 4 E4B-it Q3_K_M,
  Qwen2.5-0.5B-Instruct Q4_K_M, DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M and
  Qwen2.5-7B-Instruct-1M Q4_K_M. All run end to end — load, tokenize, prefill
  and decode on CPU and CUDA, prefix reuse, OpenAI-compatible serving. Gemma 4
  and Qwen2.5-0.5B are checked against llama.cpp by the committed golden
  suites; DeepSeek is covered by the engine regression suite; Qwen2.5-7B is
  benchmarked for speed and is not part of the golden set.

  | | file | sha256 |
  |---|---|---|
  | Gemma 4 E4B-it | `gemma-4-E4B-it-Q3_K_M.gguf` (4,058,137,728 bytes) | `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4` |
  | Qwen2.5-0.5B-Instruct | `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes) | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
  | DeepSeek-R1-Distill-Qwen-1.5B | `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (1,117,321,312 bytes) | `f3bdf9cf31dee4b57ae4e455a1cb0d01b5c2c1b50d72d3112141c195506c2840` |
  | Qwen2.5-7B-Instruct-1M | `qwen2.5-7b-instruct-1m-q4_k_m.gguf` (4,683,073,888 bytes) | `6dba997502a27b5f0c840346f45d9d62c0a04e6eb3cf72adc6c7ceb18804f380` |

  The GGUF's architecture string selects the plan and the chat format. There is
  no `if (arch == ...)` in the planner or the backends, and no model-specific
  code outside the GGUF loader and the plan emitter — the Qwen2 differences
  (a SiLU-gated FFN, full attention with no sliding window, a bias on each
  Q/K/V projection, an attention scale) are all generic ops and attributes, and
  the fallback to the token embedding when a file omits `output.weight` is
  generic too. DeepSeek and the 1M variant load through that same qwen2 path;
  their vocabulary, context length and RoPE base come from the GGUF.
- **Quant formats:** Q3_K, Q4_K, Q5_K, Q6_K, Q5_0 and Q8_0. `f16` and `q4_0`
  tensors are converted to Q5_K at load time, which requires the row length to
  be a multiple of 256; anything else is refused rather than converted. Q6_K
  loads and decodes natively, but its prefill does not use the int8 tensor-core
  weight kernel yet — it dequantizes to fp16 and uses cuBLAS, so a Q6_K
  `ffn_down` makes prefill slower than the same shape in Q4_K.
- **Compiler tier:** Boom IR is the only rewrite path, and it now generates
  CUDA kernels through MLIR/NVVM for fused glue regions (norm, quantize,
  activation, residual, RoPE), split-KV decode attention and a bf16 dense
  GEMM. Generated kernels are checked against the
  hand-written ones they replace and are selected per shape class by a runtime
  tuner; the first tune of a shape class runs during warm-up (never inside
  CUDA-graph capture) and is cached under `~/.cache/sonicboom/`, keyed by GPU,
  operator and shape class — never by prompt length or position. The JIT
  candidate tuner is on by default: each probe is bit-identity gated against
  the static default, cold tuning shares one process-wide budget
  (`SONICBOOM_AUTOTUNE_BUDGET_MS`, default 10 s, `<= 0` unlimited), and the
  picks live in a versioned, flock-protected per-SM file. `SONICBOOM_AUTOTUNE=0`
  uses the static defaults and `force` retunes (both services);
  `SONICBOOM_AUTOTUNE_JIT=0` disables only the JIT service. Generated
  decode attention is the default decode path: it uses the hand-written
  split-KV pair's chunk rule, fused-multiply-add rounding and reduction order,
  so it reproduces the token stream bit-for-bit (`SONICBOOM_ATTN_DECODE=own`
  selects the hand-written pair, which remains the numerical reference).
  Prefill attention is the hand-written `boom_fattn` kernel and the only
  implementation: one kernel covers d64/GQA-7, d128/GQA-6 (strict exact order)
  and d128/GQA-7, gathering f32, f16 and q8 KV rows into a f16 workspace, with
  tiled MMA, an online softmax, GQA query grouping, cp.async staging and a
  per-shape-class autotuned layout ([Fattn autotune](fattn-autotune.md)). The
  KV gather launches a fixed key-lane grid and loops over key tiles, so the
  recorded plan body does not depend on `n_ctx`. The generated prefill route
  (and its exp2 arm) is not reachable: that exp2 arm changed the token stream
  and did not meet the d64 fp64-error bound.
- **Speculative decoding:** a CPU n-gram/prompt-lookup drafter proposes draft
  tokens, a verify pass runs the decode graph over the drafted rows with the
  same per-row kernels and reductions as plain decode, and KV entries computed
  from rejected drafts are rolled back by valid length. A long-match drafter is
  the default and targets code and document edits; `SONICBOOM_SPEC_DRAFTER=legacy`
  selects the original 2–4-gram drafter. `SONICBOOM_SPEC_MAX_DRAFTS` sets the
  ceiling on drafts per verify (0–63; a fixed `on` defaults to 15), so a verify
  can carry up to 64 rows. The emitted stream is bit-identical to greedy by
  construction (`test_guarantees` compares spec on/off on the public models;
  the exhaustive invariance and rollback sweeps are engine regression suites
  maintained separately). `SONICBOOM_SPEC=off|on|auto|reckless`; unset runs a
  per-SM startup probe (verify/plain ratio) and picks `auto` below the
  threshold. `auto` chooses the draft length from a measured verify-cost curve
  c(k) and an acceptance-rate EMA. `reckless` runs the verify rows through the
  batch prefill kernels; it is faster on the batched matmuls but gives up
  row-invariance, and is refused under the strict numerics profile.
- **GPU:** CUDA on sm_86. `CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the
  GPU in the build machine) and is resolved at configure time; other
  architectures have not been tested. Whatever was compiled is reported by
  `sb_build_info()` / `tools/sb-report.sh`, so a binary never has to be taken on
  trust about which SMs it contains. One community report ran on an RTX 4060 Ti
  (Ada) with decode tokens identical to the sm_86 results, see the README's
  *Community results*. A CPU path exists but is a reference implementation, not
  a fast one.
- **Sampling:** greedy is the zero-valued default and stays bit-identical to
  the historical argmax path. `sb_llm_sampling` adds seeded temperature, top-k,
  top-p, min-p and repeat/presence/frequency penalties, drawing from the top
  256 candidates; the same seed, position and history reproduce the same token
  independently of KV-cache capacity or prefix reuse. A penalty window above
  1024 tokens is rejected.
- **Constrained decoding:** a session may carry a GBNF grammar or a JSON
  Schema, compiled to a token mask applied before sampling. The supported
  JSON-Schema subset is in `core/include/sonicboom/vocab/grammar.h`; an
  unsupported keyword is rejected by name. Both are exposed through the C ABI
  and the OpenAI server's grammar / `response_format` fields. On the LM head,
  only the rows the grammar allows are computed.
- **Numerics profiles:** `strict` keeps the historical f32-KV arithmetic;
  `standard` uses f16 KV and must stay inside its acceptance gate (perplexity
  delta, KL, same-top, top-1 agreement, determinism and prefix-reuse identity).
  A q8 KV cache is opt-in.
- **Image generation:** MOSS-Image text-to-image is exposed through the C ABI
  (`sb_image_*`), Guile and `/v1/images/generations`. Only the language models
  above are benchmarked in the README.
- **Chat template:** the serving layer picks the format from the loaded
  architecture — Gemma's `<|turn>…` turn markers, or Qwen2's ChatML — and takes
  the end-of-turn tokens from the model's metadata. `generate` itself still
  takes raw text, so a caller that does not go through the server formats its
  own turns.
