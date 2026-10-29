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

Four language-model families are benchmarked: d64 dense (Gemma 4, Qwen2.5-0.5B)
and d128 grouped-query attention (DeepSeek-R1-Distill-1.5B GQA-6, Qwen2.5-7B
GQA-7). DeepSeek and Qwen2.5-7B are benchmark-only: they carry no public
token-stream golden.

**Gemma 4 E4B-it Q3_K_M** (RTX 3050 8 GB, sm_86)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **53.3 tok/s** | 44.9 tok/s | **1.19×** |
| Decode at KV depth 512 | **52.7 tok/s** | 44.3 tok/s | **1.19×** |
| Prefill, 512-token prompt (marginal) | 2261.7 tok/s | 1998.7 tok/s | 1.13× |
| Prefix reuse, 512-token prefix + 64 new | 51.1 ms to first token | — | — |

**Qwen2.5-0.5B-Instruct Q4_K_M** (same machine, same session)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **350.7 tok/s** | 306.9 tok/s | **1.14×** |
| Decode at KV depth 512 | 297.9 tok/s | 303.2 tok/s | 0.98× |
| Prefill, 512-token prompt (marginal) | 17932 tok/s | 14017.3 tok/s | 1.28× |
| Prefix reuse, 512-token prefix + 64 new | 11.7 ms to first token | — | — |

**DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M** (d128/GQA-6, same machine)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **154.0 tok/s** | 149.0 tok/s | **1.03×** |
| Decode at KV depth 512 | 138.4 tok/s | 147.1 tok/s | 0.94× |
| Prefill, 512-token prompt (marginal) | 4795.7 tok/s | 5815.0 tok/s | 0.82× |
| Prefix reuse, 512-token prefix + 64 new | 33.1 ms to first token | — | — |

**Qwen2.5-7B-Instruct-1M Q4_K_M** (d128/GQA-7, same machine)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | 41.1 tok/s | 43.5 tok/s | 0.94× |
| Decode at KV depth 512 | 39.7 tok/s | 43.2 tok/s | 0.92× |
| Prefill, 512-token prompt (marginal) | 1475.4 tok/s | 1587.9 tok/s | 0.93× |
| Prefix reuse, 512-token prefix + 64 new | 101.7 ms to first token | — | — |

Decode leads on the two d64 families at depth 0 (1.19× Gemma, 1.14× Qwen2.5-0.5B).
Qwen2.5-0.5B has no sliding window, so it attends all 512 keys while Gemma's
ring buffer wraps — that is why it is level at depth 512. DeepSeek is at decode
parity (1.03×) and Qwen2.5-7B is 6–8% behind. Prefill is ahead on both d64
families (1.13× / 1.28×, the converged `boom_fattn` path) and behind on the d128
families (0.82× / 0.93×). Prefix reuse has no llama.cpp counterpart
(`llama-bench` cannot measure it), so it is reported as first-token latency
only; the exactness claim beside it is a test, not a benchmark — a reuse
continuation is token-identical to a full prefill (`test_guarantees` covers the
public contract; the exhaustive prefix-length sweep is an engine regression
suite maintained separately).

The v0.0.3 values for the two original families are kept in the
[README numbers table](../README.md#numbers) so the release-over-release change
stays visible; the doc here reports the current release only.

SM clock medians over the published runs: 1927 MHz (Gemma), 1942 MHz
(Qwen2.5-0.5B), 1935 MHz (DeepSeek) and 1852 MHz (Qwen2.5-7B); the script
prints its own clock readings per run.

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
| Gemma model | [`unsloth/gemma-4-e4b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF) → `gemma-4-E4B-it-Q3_K_M.gguf` (4,058,137,728 bytes, sha256 `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4`) |
| Qwen2.5-0.5B model | [`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) → `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes, sha256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`) |
| DeepSeek model | [`unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) → `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (1,117,321,312 bytes, sha256 `f3bdf9cf31dee4b57ae4e455a1cb0d01b5c2c1b50d72d3112141c195506c2840`) |
| Qwen2.5-7B model | [`ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF`](https://huggingface.co/ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF) → `qwen2.5-7b-instruct-1m-q4_k_m.gguf` (4,683,073,888 bytes, sha256 `6dba997502a27b5f0c840346f45d9d62c0a04e6eb3cf72adc6c7ceb18804f380`) |
| llama.cpp | commit `43fe9c64281`, CUDA build, `llama-bench -ngl 99` |
| SonicBoom build | `tools/sb-report.sh`: git commit, CUDA archs actually compiled, LLVM version |

The comparison is against llama.cpp at the commit above; newer versions may be
faster or slower. The Qwen2.5-0.5B greedy token stream agrees with llama.cpp
except at llama.cpp's own near-ties, where a different prefill batch geometry
flips the last-place logits (first divergent step and top-1/top-2 margin
recorded by the tests). DeepSeek and Qwen2.5-7B are benchmark-only and carry no
token-stream claim.

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

## Community runs

Numbers from other people's hardware are collected in the README's *Community
results* section, as reported and marked not reproduced by us. The copy-paste
standard run — hardware/build record, canonical bench, extended-context tiers,
clock pinning and what to send — is
[README: Standard run](../README.md#standard-run).

A community submission follows the same protocol as the published numbers:

- both runtimes on the same GGUF, on the same machine, in the same session;
- medians of 3 independent process runs (`--runs 3`);
- one GPU at a time, idle, with no other measurement running;
- the SM clock either pinned or reported as min/median/max;
- the hardware, driver, SonicBoom build and llama.cpp commit included with the
  numbers.

A submission that omits the conditions is reported as a regression check, not
as a published ratio.
