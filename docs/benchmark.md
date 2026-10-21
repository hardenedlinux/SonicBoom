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

`tools/bench.sh --workload` covers two session traces the per-request set cannot
express: an eight-turn agent session (shared 1000-token system prompt, prefix
cache on, reported as per-turn TTFT, per-turn decode rate and wall time beside a
fresh-session-per-turn variant) and a coding request (2048-token prompt,
256-token generation). `tools/bench.sh --ablate` runs the canonical set once per
disabled Boom IR pass group or runtime feature and reports what each is worth.

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
- A/B mode (`--ab BASE_LIB CUR_LIB`) stages two `libsonicboom.so` builds and
  runs them in **alternating** order per repetition, so drift lands on both
  sides. The noise floor is measured, not assumed: the same library is run on
  both sides for `--control-runs` pairs and a metric's gate is
  `|median(control deltas)| + k·SD(control deltas)` (default `k=2`).
- Measurement jobs run alone on an idle GPU under `/tmp/sb-bench.lock`; a run
  that overlapped other load is re-measured rather than reported.

## Current numbers

Medians of 3 process runs, both runtimes on the same GGUF in the same session,
on the machine in [Conditions](#conditions). Every claim is bounded by that: one
GPU, one quant file.

**Gemma 4 E4B-it Q3_K_M** (RTX 3050 8 GB, sm_86)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **53.6 tok/s** | 44.8 tok/s | **1.20×** |
| Decode at KV depth 512 | **53.5 tok/s** | 44.1 tok/s | **1.21×** |
| Prefill, 512-token prompt (marginal) | 2084 tok/s | 1995 tok/s | 1.04× |
| Prefix reuse, 512-token prefix + 64 new | 57.4 ms to first token | — | — |

**Qwen2.5-0.5B-Instruct Q4_K_M** (same machine, same session)

| metric | SonicBoom | llama.cpp | ratio |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **345.9 tok/s** | 306.9 tok/s | **1.13×** |
| Decode at KV depth 512 | **307.0 tok/s** | 302.8 tok/s | **1.01×** |
| Prefill, 512-token prompt (marginal) | 15301 tok/s | 14027 tok/s | 1.09× |
| Prefix reuse, 512-token prefix + 64 new | 13.7 ms to first token | — | — |

Decode leads on both families at depth 0 and depth 512. Qwen2 has no sliding
window, so it attends all 512 keys while Gemma's ring buffer wraps — that is
why Qwen2 is level at depth 512. Prefill is at parity on Gemma and ahead on
Qwen. Prefix reuse has no llama.cpp counterpart (`llama-bench` cannot measure
it), so it is reported as first-token latency only; the exactness claim beside
it is a test, not a benchmark — a reuse continuation is token-identical to a
full prefill at every tested prefix length (`test_prefix_cache`).

SM clock medians over the published runs: 1912 MHz (Gemma) and 1927 MHz
(Qwen); the script prints its own clock readings per run.

## Correctness gates

A performance number is only reported for a build that passes:

- the full test suite (`ctest --test-dir build`), CUDA on and off;
- the decode golden: `test_generate` reproduces the committed llama.cpp
  multi-token stream on the CPU path, and the CUDA path is compared per token
  (GPU lanes may diverge only at near-ties, with the first divergent step and
  top-1/top-2 margin recorded);
- the numerics gate for the active profile (`tools/sb-numerics`): strict is the
  historical f32-KV path; standard (f16 KV) must stay inside its acceptance
  gate — perplexity delta, top-1 agreement, determinism, batch invariance and
  prefix-reuse identity.

## Conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB, compute 8.6 |
| Driver / CUDA | 595.84 (CUDA 13.2 driver API); nvcc 12.0.140 |
| CPU | AMD Ryzen 5 5500 (6C/12T, Zen 3), PCIe 3.0 |
| RAM | 32 GB DDR4-2400, single channel |
| Host | Ubuntu 24.04, Linux 6.8, x86-64, g++-13, build type Release |
| Gemma model | [`unsloth/gemma-4-e4b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF) → `gemma-4-E4B-it-Q3_K_M.gguf` (4,058,137,728 bytes, sha256 `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4`) |
| Qwen model | `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes, sha256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`) |
| llama.cpp | commit `43fe9c64281`, CUDA build, `llama-bench -ngl 99` |
| SonicBoom build | `tools/sb-report.sh`: git commit, CUDA archs actually compiled, LLVM version |

The comparison is against llama.cpp at the commit above; newer versions may be
faster or slower. The Qwen2 greedy token stream agrees with llama.cpp except at
llama.cpp's own near-ties, where a different prefill batch geometry flips the
last-place logits (first divergent step and top-1/top-2 margin recorded by the
tests).

## Reproduce

```sh
# canonical table for one model (SonicBoom; set LLAMA_BENCH to add the
# llama.cpp column and the ratios)
LLAMA_BENCH=$LLAMA_ROOT/build-cuda/bin/llama-bench \
  tools/bench.sh models/gemma-4-E4B-it-Q3_K_M.gguf

# one metric by hand, e.g. decode at depth 512
tools/baseline/longctx_bench --model models/gemma-4-E4B-it-Q3_K_M.gguf \
  --depth 512 --steps 64

# baseline (llama.cpp, same GGUF and machine)
$LLAMA_ROOT/build-cuda/bin/llama-bench \
  -m models/gemma-4-E4B-it-Q3_K_M.gguf -p 512 -n 64 -ngl 99 -r 1

# two builds of libsonicboom.so against each other, with the measured noise floor
tools/bench.sh models/gemma-4-E4B-it-Q3_K_M.gguf \
  --ab build-base/libsonicboom.so build/libsonicboom.so

# correctness
ctest --test-dir build
```

Numbers from other people's hardware are collected in the README's *Community
results* section, as reported and marked not reproduced by us.
