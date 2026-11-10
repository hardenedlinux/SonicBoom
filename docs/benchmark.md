# SonicBoom benchmark

How SonicBoom is measured, what its numbers mean, and the numbers current as of
this commit. The canonical harness is [`tools/bench.sh`](../tools/bench.sh); it
reprints every table below on any machine, and
[`tools/sb-report.sh`](../tools/sb-report.sh) prints the hardware and the exact
build that produced it.

## What is measured

The canonical metric set is fixed so that numbers from different sessions stay
comparable. All of it is the CUDA spine with the whole model resident on the GPU
(llama.cpp side: `-ngl 99`):

| metric | meaning |
|---|---|
| `tg64` | decode 64 tokens from position 0 |
| `d512` | decode 64 tokens at KV depth 512 |
| `pp512` | prefill a 512-token prompt, reported *marginal* (per-token prefill cost) |
| `reuse` | 512-token prefix + 64-token continuation, first-token latency through the prefix cache |

The decode tools warm the exact `(model, n_ctx)` runtime before the timed
region, so graph/plan build and CUDA-graph capture are never counted. `pp512`
uses the llama-bench `-p n -n 0` semantics (per-token marginal cost), not the
wall time of a first cold prefill.

### Prefill session lifecycle

A session owns its KV cache, and every execution runtime is bound to that cache
and keyed by its identity. What a prefill number means therefore depends on how
many sessions the run creates and where the one-time build is charged.
[`prefill_bench`](../tools/baseline/prefill_bench.cpp) separates the two
scenarios and never mixes them; both use the same code on every revision of the
library, so old and new measurements stay comparable.

- **reused** (`RESULT prefill ... session=reused`) is the canonical throughput.
  One session is created before measurement; its KV cache, decode runtime,
  prefill segment plans and captured full-chunk graph are built once
  (`first_prefill_ms`), then every repetition runs the same full prefill with
  only the logical prefill state reset (`filled_tokens` cleared — no KV
  reallocation and no runtime rebuild). The timed region contains prefill
  execution only.
  - `pp_wall_tok_s` = `N / t_warm`, the steady-state throughput.
  - `pp_marginal_tok_s` = `(N - n0) / (t_N - t_n0)`, all times from that
    session. The cancellation baseline `n0` takes the *same execution path*:
    `n0 = n/2` for `n <= chunk` (both one uncaptured batched segment) and
    `n0 = n - chunk` for `n > chunk` (the delta is exactly one full chunk,
    with the final partial segment identical in both runs). A one-token call
    is never used: it takes the single-column kernels and skips the prefill
    segment/capture path, so it cancels costs that were never added.
- **fresh** (`RESULT prefill_fresh*`) reports session lifecycle cost, not
  throughput. Every repetition creates and destroys a session exactly as
  `sb_llm_session_new`/`sb_llm_session_free` do. `session_create_ms` is the KV
  allocation plus the session-bound runtime warm-up; `first_prefill_ms` is that
  session's first prefill (segment plan build, capture when `N > chunk`,
  execution); `e2e_ms` is their sum — what a server creating one session per
  request pays. Runtimes are released between repetitions, so the measurement
  does not accumulate sessions or device memory.
- An untimed warm-up session/prefill precedes both sections, so
  `first_prefill_ms` means the session's own build (segment plan, capture), not
  one process's first kernel load.

Compare numbers produced by the same mode. Reading a per-request measurement
(session creation inside the timed region) against a steady-state one is what
makes a session-ownership change look like a kernel regression; `tools/bench.sh`
uses the reused mode.

## Baselines

llama.cpp is the reference baseline, not an oracle: the same GGUF file, the same
machine, the same session, `llama-bench -ngl 99`. SonicBoom's correctness
reference is its own CPU path (and the committed goldens). A number without a
baseline row is a regression check, not a claim.

## Run protocol

- Every metric is the **median of independent process runs** (`--runs`,
  default 3). Each underlying tool ends its output with one machine-readable
  `RESULT ...` line; the script parses that, never free text.
- Both runtimes use the same GGUF, on the same machine, in the same session.
- The GPU clock is pinned for the run (`nvidia-smi -lgc`, released on exit with
  `-rgc`); an unpinned clock drifts by more than the effects being compared.
  When pinning is impossible the run is not clock-homogeneous and the script
  reports the SM clock it actually ran at (min/median/max).
- Measurement jobs run alone on an idle GPU under `/tmp/sb-bench.lock`; a run
  that overlapped other load is re-measured rather than reported.

## Current numbers

Medians of 3 process runs, both runtimes on the same GGUF in the same session,
on the machine in [Conditions](#conditions). Every claim is bounded by that: one
GPU, one quant file.

Two language models are benchmarked: Gemma 4 E4B (d64 dense) and
Qwen2.5-7B-Instruct-1M (d128 grouped-query attention, GQA-7). The Gemma table
below was recorded with the Q3_K_M file; the file the tools default to, and the
model list in [Conditions](#conditions), name Q5_K_M.

**Gemma 4 E4B-it Q3_K_M** (RTX 3050 8 GB, sm_86)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **53.3 tok/s** | 44.9 tok/s | **1.19×** |
| Decode at KV depth 512 | **52.7 tok/s** | 44.3 tok/s | **1.19×** |
| Prefill, 512-token prompt (marginal) | 2261.7 tok/s | 1998.7 tok/s | 1.13× |
| Prefix reuse, 512-token prefix + 64 new | 51.1 ms to first token | — | — |

**Qwen2.5-7B-Instruct-1M Q4_K_M** (d128/GQA-7, same machine)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | 41.0 tok/s | 43.5 tok/s | 0.94× |
| Decode at KV depth 512 | 40.6 tok/s | 43.2 tok/s | 0.94× |
| Prefill, 512-token prompt (marginal) | 1598.2 tok/s | 1589.9 tok/s | 1.01× |
| Prefix reuse, 512-token prefix + 64 new | 62.7 ms to first token | — | — |

Gemma decodes 1.19× ahead of llama.cpp at depth 0, and Qwen2.5-7B is 6%
behind at depths 0 and 512. Prefill is ahead on Gemma (1.13×, the converged
`boom_fattn` path) and level on Qwen2.5-7B at 512 tokens (1.01×); it falls
behind as the prompt grows (see [Long-context tiers](#long-context-tiers)).
Prefix reuse has no llama.cpp counterpart
(`llama-bench` cannot measure it), so it is reported as first-token latency
only; the exactness claim beside it is a test, not a benchmark — a reuse
continuation is token-identical to a full prefill (`test_guarantees` covers the
public contract; the exhaustive prefix-length sweep is an engine regression
suite maintained separately).

The v0.0.3 values for the two original families are kept in the
[README numbers table](../README.md#numbers) so the release-over-release change
stays visible; the doc here reports the latest published measurement of each
model (Gemma 4 from v0.0.4; Qwen2.5-7B from the v0.0.6 run).

SM clock medians over the published runs: 1927 MHz (Gemma, v0.0.4), 1867 MHz
(Qwen2.5-7B, code-edit run) and 1882 MHz (Qwen2.5-7B, long-context run); the
script prints its own clock readings per run.

## Long-context tiers

The same model and session, `tools/bench.sh --long`: decode is 64 tokens at
the given KV depth, prefill is the whole prompt. Medians of 3 process runs;
both runtimes on the same GGUF in the same session.

**Qwen2.5-7B-Instruct-1M Q4_K_M** (d128/GQA-7)

| tier | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| d4096 | 38.3 tok/s | 41.4 tok/s | 0.92× |
| d8192 | 36.6 tok/s | 39.6 tok/s | 0.93× |
| d16384 | 33.2 tok/s | 36.4 tok/s | 0.91× |
| d32768 | 28.5 tok/s | 31.3 tok/s | 0.91× |
| pp4096 | 1234.2 tok/s | 1517.7 tok/s | 0.81× |
| pp8192 | 1007.6 tok/s | 1425.0 tok/s | 0.71× |
| pp16384 | 701.7 tok/s | 1272.5 tok/s | 0.55× |
| pp32768 | 442.6 tok/s | 1046.9 tok/s | 0.42× |

Decode stays within 0.91–0.93× of llama.cpp out to 32K. Prefill is level at
512 tokens and diverges as the prompt grows: the d128 batched path is the
remaining gap, and it is where the next prefill work lands.

## Code editing

The code-editing measurement is ten frozen edit requests in
[`tools/bench/codeedit`](../tools/bench/codeedit). `manifest.tsv` lists each
turn's index, source path, sha256 and `add_special` flag; `NN.instruction` is
the edit request and `NN.file` the pasted source. A turn's prompt is the
instruction bytes followed by the file bytes (only the first turn carries a
special token), and its generation length is the pasted file's own token count,
so the reply has to return the whole file.

`codeedit_bench` runs the turns through the public C ABI on one session
(`--warm-session`, `--warmup`, greedy, seed 1234), two prefills per turn the
way a stateless request API re-sends the file: prefill(prompt), decode one
token, prefill(prompt + that token), decode the file's token count. The
speculative arm is chosen with `SONICBOOM_SPEC=off|auto`; `--ctx-only` prints
the smallest context that holds the largest request with its generation,
`max_t(prompt_tokens_t + gen_tokens_t + 1) + 2`.

`tools/bench/llama-codeedit.scm` runs the same requests through llama-server:

```sh
llama-server -m GGUF -ngl 99 -fa on -np 1 --no-webui --seed 1234 \
             -c CTX --port PORT [--spec-type ngram-mod]
```

with every other speculative option at its server default. Each turn posts to
`/completion` with `temperature` 0, `top_k` 1, `top_p` 1.0, `min_p` 0.0,
`repeat_penalty` 1.0, `seed` 1234, `cache_prompt` true, `ignore_eos` true,
`stream` true, the manifest's `add_special`, and the turn's `n_predict`;
`prompt_n`, `predicted_n` and `predicted_ms` are read from the reply's
`timings`.

The arms are SonicBoom spec off (1 rep, deterministic), SonicBoom spec auto
(`--runs` reps), llama.cpp no spec (1 rep) and llama.cpp ngram-mod (`--runs`
reps, a fresh server per rep because the n-gram container is per server).
Identity is per turn: a SonicBoom auto rep is identical to the off arm when all
ten token hashes match, and a llama.cpp ngram-mod rep is identical when all ten
generated texts match the no-spec arm byte for byte. `tools/bench.sh MODEL
--spec` runs all four arms and prints the table.

On a coding-chat workload, speculation makes SonicBoom 4.4× faster than its own
plain decode (176.8 tok/s), with the same output. In the same test, llama.cpp's
non-speculative output also differed between repeated runs on 5 of 10 turns.

## Correctness gates

A performance number is only reported for a build that passes:

- the full test suite (`ctest --test-dir build`), CUDA on and off;
- the public guarantee check (`test_guarantees`): determinism, prefix-reuse
  identity, KV-capacity independence and speculative-decoding identity on the
  published models;
- the decode golden (engine regression suite): the committed llama.cpp
  multi-token stream is reproduced on the CPU path, and the CUDA path is
  compared per token (GPU lanes may diverge only at near-ties, with the first
  divergent step and top-1/top-2 margin recorded);
- the numerics gate for the active profile (`tools/sb-numerics`): strict is the
  historical f32-KV path; standard (f16 KV) must stay inside its acceptance
  gate — perplexity delta, KL, same-top, top-1 agreement, determinism and
  prefix-reuse identity.

SonicBoom guarantees exactly three properties: (1) determinism — same input,
same output, every run; (2) prefix-reuse identity — output does not depend on
whether, or where, a prefix was reused; (3) KV-capacity independence — output
does not depend on `n_ctx`. Prefill-vs-decode identity is **not** guaranteed:
a batched prefill and the same tokens fed one at a time may use different
arithmetic. Speculative decoding on/off is nevertheless **bit-identical**: the
verify step uses decode arithmetic for every row (`test_guarantees` compares
on/off on the public models).

## Conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB, compute 8.6 |
| Driver / CUDA | 595.84 (CUDA 13.2 driver API); nvcc 12.0.140 |
| CPU | AMD Ryzen 5 5500 (6C/12T, Zen 3), PCIe 3.0 |
| RAM | 32 GB DDR4-2400, single channel |
| Host | Desktop, Ubuntu 24.04, Linux 6.8, x86-64, g++-13, build type Release |
| Gemma model | [`unsloth/gemma-4-e4b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF) → `gemma-4-E4B-it-Q5_K_M.gguf` (5,481,798,784 bytes, sha256 `a5d6e634db151368d2caa4270fd32ab604d96c2d1291950586989f46d8e17d36`) |
| Qwen2.5-7B model | [`ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF`](https://huggingface.co/ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF) → `qwen2.5-7b-instruct-1m-q4_k_m.gguf` (4,683,073,888 bytes, sha256 `6dba997502a27b5f0c840346f45d9d62c0a04e6eb3cf72adc6c7ceb18804f380`) |
| llama.cpp | commit `43fe9c64281`, CUDA build, `llama-bench -ngl 99` |
| SonicBoom build | `tools/sb-report.sh`: git commit, CUDA archs actually compiled, LLVM version |

The comparison is against llama.cpp at the commit above; newer versions may be
faster or slower. A greedy token stream agrees with llama.cpp except at
llama.cpp's own near-ties, where a different prefill batch geometry flips the
last-place logits (first divergent step and top-1/top-2 margin recorded by the
tests).

## Reproduce

```sh
# canonical table for one model (SonicBoom; set LLAMA_BENCH to add the
# llama.cpp column and the ratios)
LLAMA_BENCH=$LLAMA_ROOT/build-cuda/bin/llama-bench \
  tools/bench.sh models/gemma-4-E4B-it-Q3_K_M.gguf

# one metric by hand, e.g. decode at depth 512 (model from SONICBOOM_GEMMA_GGUF)
SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
  tools/baseline/longctx_bench --depth 512 --steps 64

# baseline (llama.cpp, same GGUF and machine)
$LLAMA_ROOT/build-cuda/bin/llama-bench \
  -m models/gemma-4-E4B-it-Q3_K_M.gguf -p 512 -n 64 -ngl 99 -r 1

# extended-context tiers: decode at depth 4K/8K/16K/32K and prefill 4K–32K
LLAMA_BENCH=$LLAMA_ROOT/build-cuda/bin/llama-bench \
  tools/bench.sh models/gemma-4-E4B-it-Q3_K_M.gguf --long

# correctness
ctest --test-dir build
```

## Standard run

The standard run on Qwen2.5-7B-Instruct-1M, in three commands:

```sh
export CUDA_VISIBLE_DEVICES=0
MODEL=models/qwen2.5-7b-instruct-1m-q4_k_m.gguf
LLAMA=/path/to/llama.cpp/build/bin

# 1. What is being measured
tools/sb-report.sh --model "$MODEL" --no-bench
git -C /path/to/llama.cpp rev-parse --short HEAD

# 2. Code editing, speculation off and on, SonicBoom and llama.cpp
LLAMA_BIN="$LLAMA" tools/bench.sh "$MODEL" --runs 3 --spec

# 3. Plain decode and prefill, short and long context
LLAMA_BIN="$LLAMA" tools/bench.sh "$MODEL" --runs 3 --long
```

The optional export/import pair checks cross-machine identity of the code-edit
token streams:

```sh
tools/bench.sh "$MODEL" --spec --tune-export qwen7b-tune.tar
tools/bench.sh "$MODEL" --runs 1 --spec --tune-import qwen7b-tune.tar
```

## Community runs

Numbers from other people's hardware are collected in the README's *Community
results* section, as reported and marked not reproduced by us. The copy-paste
standard run — hardware/build record, canonical bench, extended-context tiers,
clock pinning and what to send — is
[README: Reproduce the table](../README.md#reproduce-the-table).

Earlier per-release results threads:

- 0.0.3: [#1](https://github.com/hardenedlinux/SonicBoom/issues/1),
  [#2](https://github.com/hardenedlinux/SonicBoom/issues/2)
- 0.0.4: [#3](https://github.com/hardenedlinux/SonicBoom/issues/3)
- 0.0.5: [#4](https://github.com/hardenedlinux/SonicBoom/issues/4)

A community submission follows the same protocol as the published numbers:

- both runtimes on the same GGUF, on the same machine, in the same session;
- medians of 3 independent process runs (`--runs 3`);
- one GPU at a time, idle, with no other measurement running;
- the SM clock either pinned or reported as min/median/max;
- the hardware, driver, SonicBoom build and llama.cpp commit included with the
  numbers.

A submission that omits the conditions is reported as a regression check, not
as a published ratio.
