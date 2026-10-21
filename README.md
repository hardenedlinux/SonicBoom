# SonicBoom

<img src="assets/sonic-boom-logo.webp" height="600">

**A native tensor runtime. No Python. One `.so`.**

SonicBoom is an AI framework written from the ground up in C++ and CUDA,
for workloads where performance matters. Everything is exposed through a
stable C ABI, and GNU Guile is its first-class language. You load
`libsonicboom.so` from whatever language you use. There is no Python
interpreter, no pip, and no wheel tree on the inference path.

SonicBoom targets local LLM inference on consumer GPUs. On a laptop
RTX 3050 it **decodes 13–20% faster than llama.cpp, using the same quantized
kernels** — 1.20× on Gemma 4 E4B-it Q3_K_M, 1.13× on Qwen2.5-0.5B-Instruct
Q4_K_M ([numbers](#numbers), same machine, medians of 3) — and its prefill is
at llama.cpp parity on Gemma 4 (1.04×) and 9% ahead on Qwen2.5. On decode the hot
kernels are the same, so the lead comes from the layers above them: a graph
planner, and behind it an MLIR compiler tier that now also generates GPU
kernels. Those layers are what llama.cpp's architecture does not have, and they
are where SonicBoom keeps improving.

> **Status: v0.0.3 released; development branch `boom-next`.** Two model
> families are verified end to end — Gemma 4 E4B-it (Q3_K_M) and Qwen2.5-0.5B
> Instruct (Q4_K_M) — on CUDA sm_86, both checked against llama.cpp. v0.0.3
> makes Boom IR the only rewrite path, adds generated GPU kernels (fused glue,
> attention and a bf16 dense GEMM), per-GPU runtime autotuning, MMQ at
> upstream build parity, and brings prefill to llama.cpp parity. `boom-next`
> adds lossless speculative decoding (off by default), makes generated decode
> attention the default (token-identical to the hand-written pair), and closes
> part of the generated prefill-attention gap with a cp.async K/V pipeline,
> padded shared tiles and conditional softmax rescaling. Read
> [Scope of v0.0.3](#scope-of-v003) before trying it; the numbers below are
> from `boom-next` (2026-10-19).

## The architecture

![SonicBoom architecture](assets/sonicboom-architecture-light.svg#gh-light-mode-only)
![SonicBoom architecture](assets/sonicboom-architecture-dark.svg#gh-dark-mode-only)

## How this is built

- Human architectured
- Agent: Claude Code
- Worker: Deepseek-flash-v4.1

---

## Why another runtime?

**For people who need to know what they are running.** The core is
26k lines of C++/CUDA. It is GPL-3.0-or-later, and the boundary is a plain
C ABI that you can audit, pin and link against. It has no Python dependency
chain to vet, and nothing is downloaded at run time. Python is not needed to
build it either: PyTorch's Python code generator, torchgen, is replaced by
[SonicCross](https://github.com/hardenedlinux/SonicCross), written in Guile.

**Fast because of the runtime, not only the kernels.** Each decode step runs
as a single CUDA Graph replay. The KV cache stays on the device. Planner passes
fuse norm, activation and quantization. Prefix-cache reuse is exact: a
continuation is token-identical to a full recompute.

**More than an LLM runner.** SonicBoom is a general tensor framework: a
better PyTorch, with no Python in it. You write models in Scheme, as eagerly as
in PyTorch, and they are traced and compiled into graphs. Production models are
defined in the C++ core and served from any language. LLM inference is the first
benchmark, not the limit of the project.

---

## Numbers

All numbers below are **medians of 3 process runs**, measured on the machine in
[Benchmark conditions](#benchmark-conditions), with **both runtimes on the same
GGUF file in the same session** — SonicBoom through
[`tools/bench.sh`](tools/bench.sh), llama.cpp through `llama-bench -ngl 99`
(commit `43fe9c64281`). Every claim in this section is bounded by that: one
model, one GPU, one quant file. `tools/bench.sh` reprints the whole table on any
machine, and [`tools/sb-report.sh`](tools/sb-report.sh) prints the hardware and
the build that produced it.

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

- **Decode is ahead on both families**, at depth 0 and at depth 512: 1.20×/1.21×
  on Gemma and 1.13×/1.01× on Qwen. Qwen2 at depth 512 is level with llama.cpp
  (1.01×): it has no sliding window, so all 512 keys are attended while Gemma's
  ring buffer wraps.
- **Prefill is at parity on Gemma (1.04×) and ahead on Qwen (1.09×).** The
  path is int8 tensor-core MMQ plus a generated bf16 dense GEMM. The remaining
  prefill cost is attention: 25.7 ms against llama.cpp's 8.25 ms at pp512
  (3.12×), about 70% of the residual kernel-time gap. The generated FA-2 now
  uses a 2–3-stage cp.async K/V pipeline, padded shared-memory tiles and a
  conditional online-softmax rescale; a tensor-core path with llama-class tiles
  is the next lever.
- **Prefix reuse** has no llama.cpp counterpart (`llama-bench` cannot measure
  it), so it is reported as first-token latency in milliseconds and nothing
  else. The exactness claim beside it is a test, not a benchmark: a
  prefix-reuse continuation is token-identical to a full prefill at every
  tested prefix length (`test_prefix_cache`, 14/14; three more 8K-context cases
  under `--long`).
- SM clock medians over the published runs: 1912 MHz (Gemma) and 1927 MHz
  (Qwen); `tools/bench.sh` prints its own clock readings, because the GPU is
  not power-capped on this host. Decode moves by about 1% between runs at
  these clocks.

### Benchmark conditions

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

### Community results

Numbers from other people's hardware belong here, next to the command that
produced them. To add a row, run both tools on your machine and paste the
output into an issue or a PR:

```sh
tools/sb-report.sh --gemma your-model.gguf --no-golden --no-bench  # hardware + exact build
LLAMA_BENCH=... tools/bench.sh your-model.gguf --label your-model
```

`sb-report.sh --json` gives the same facts as one machine-readable object, so a
result can carry its own conditions. Results are published as reported, with
the reporter credited, and marked as not reproduced by us.

| GPU | Driver / CUDA | SonicBoom build | Model | SonicBoom decode | llama.cpp tg64 | Ratio | Decode tokens vs sm_86 golden | Reporter |
|---|---|---|---|---:|---:|---:|---|---|
| RTX 4060 Ti 16 GB (sm_89) | 550.163.01 / 12.4 | `1462b04` (sm_86 code, before tuning) | Gemma 4 E4B-it Q3_K_M | 73.6 tok/s | 78 tok/s | 0.94 | identical (64/64) | [@svenha](https://github.com/hardenedlinux/SonicBoom/issues/1) |

The 4060 Ti row predates `tools/bench.sh` and the build-for-the-detected-GPU
default: it measures sm_86 code running on Ada with `test_generate`, against
llama.cpp `43fe9c64281` built for the card. Both numbers come from the
reporter's machine.

---

## Quick start

### Requirements

- Linux x86-64, g++-13, CMake, Ninja (optional), libyaml-dev. The CPU
  quantized-matmul kernels are AVX2 (with a scalar fallback); an aarch64 build
  is not wired up yet — the AVX2 translation unit has no architecture guard, so
  it does not compile there as it stands. The port is a guard plus a NEON
  sibling of the same fixed-point kernel.
- CUDA toolkit 12.0 or newer and an NVIDIA GPU, compute capability 6.1 or
  newer (`__dp4a`) — 7.5 or newer for the tensor-core MMQ path. v0.0.3 is built
  and tested for **sm_86** only (RTX 30-series).
- GNU Guile 3.0 and
  [guile-json](https://github.com/aconchillo/guile-json) (the `(json)` module,
  used by the Guile binding for build info and by the serve front end).
  `apt install guile-3.0 guile-json` on Debian/Ubuntu.
- [SonicCross](https://github.com/hardenedlinux/SonicCross), a required
  build-time dependency (see below)

### Build

SonicBoom's operator library, native-torch, is generated at build time from
PyTorch's `native_functions.yaml`, as PyTorch itself does with torchgen.
SonicBoom uses **SonicCross**, a torchgen written in Guile, which turns the YAML
into native-torch's C++ declaration, registration and dispatch sources (more
than 7,000 per-operator headers). SonicCross runs only at build time and is
not linked into `libsonicboom.so`. Install SonicCross **v0.0.2** first, the version pinned in
[`tools/generation/SONICCROSS.lock`](tools/generation/SONICCROSS.lock):

```sh
git clone https://github.com/hardenedlinux/SonicCross
cd SonicCross
git checkout v0.0.2
./bootstrap && ./configure && make && sudo make install
cd ..
```

This installs `guild torchgen`. If `guild` is not on your `PATH`, pass
`-DSONICCROSS_GUILD=/path/to/guild` to CMake. Then build SonicBoom:

```sh
git clone --recursive https://github.com/hardenedlinux/SonicBoom
cd SonicBoom

# One-time step: build the pinned LLVM/MLIR into build/llvm-mlir-install.
# It is statically linked into libsonicboom.so and is not a runtime dependency.
# The third_party/llvm-project submodule is only this script's input.
tools/mlir/build-mlir.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSONICBOOM_USE_CUDA=ON
cmake --build build -j
```

The LLVM submodule is large, so the first clone and build take a while.

An LLVM/MLIR already installed elsewhere can be used instead — the submodule
stays empty and is not needed at all:

```sh
cmake -S . -B build -DSONICBOOM_USE_CUDA=ON \
      -DMLIR_DIR=/opt/llvm/lib/cmake/mlir -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
```

A different LLVM major is accepted with a warning, naming the version SonicBoom
is tested against (23). MLIR is required — the compiler library it carries is
the only rewrite path the runtime has — and the error message when it is missing
lists both routes.

Whatever was configured is compiled into the library and can be read back — from
C with `sb_build_info()` (`capi/include/sonicboom/capi.h`), from Guile with
`(build-info)` (`bindings/guile/README.md`), and from the command line with
[`tools/sb-report.sh`](tools/sb-report.sh), which prints it beside the host, the
GPU and the toolchain:

```sh
tools/sb-report.sh --gemma your-model.gguf --no-golden --no-bench  # --json for machines
```

### Run (Guile)

Download the model and verify it:

```sh
mkdir -p models
curl -L -o models/gemma-4-E4B-it-Q3_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-E4B-it-Q3_K_M.gguf
echo "0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4  models/gemma-4-E4B-it-Q3_K_M.gguf" | sha256sum -c
```

Any GGUF the runtime supports runs the same way — for instance
`qwen2.5-0.5b-instruct-q4_k_m.gguf`, the other verified model (both files and
their checksums are in [Scope of v0.0.3](#scope-of-v003)).

Then start Guile:

```sh
LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme
```

```scheme
(use-modules (sonicboom inference))

(define model   (load-model "models/gemma-4-E4B-it-Q3_K_M.gguf"))
(define session (make-session model 4096))

(display (generate model session "The three laws of thermodynamics are" 64))

;; Prefix reuse is automatic: when a later prompt on the same session starts
;; with the same tokens, only the new suffix is prefilled.
(prefill! session (tokenize model "..."))   ; returns the number of reused tokens

(free-session! session)
(free-model! model)
```

The same functions are available from C through
[`capi/include/sonicboom/capi.h`](capi/include/sonicboom/capi.h) (`sb_llm_*`).

### Test

```sh
ctest --test-dir build -LE long --output-on-failure   # per-commit suite
ctest --test-dir build -L long                        # nightly/weekly suite
```

Tests labelled `long` verify correctness slowly (model loads and the pure-Scheme
reference interpreter), so they run in the nightly/weekly job rather than on
every change. Tests that need the model skip themselves when the GGUF is not
present, and the two ResNet-18 end-to-end tests skip (CTest
`SKIP_RETURN_CODE 77`) when their generated sidecar artifacts are absent, which
is the state of a fresh checkout.

### Benchmark

```sh
LLAMA_BENCH=/path/to/llama-bench tools/bench.sh your-model.gguf
LLAMA_BENCH=/path/to/llama-bench tools/sb-report.sh --gemma your-model.gguf
```

The second command runs the same canonical bench (3 process runs by default)
and the golden checks through ctest, and prints the hardware and the build
behind the numbers; `--qwen PATH` adds the Qwen goldens.

Every metric is the median of 3 process runs and both runtimes are measured in
the same session; `tools/bench.sh` is the only benchmark SonicBoom publishes
numbers from.

---

## Scope of v0.0.3

v0.0.3 is a working runtime, not a drop-in replacement for llama.cpp. It adds
GPU kernel generation and per-GPU runtime autotuning to the v0.0.2 runtime.
Its limits:

- **Two verified model families:** Gemma 4 E4B-it Q3_K_M and Qwen2.5-0.5B-Instruct
  Q4_K_M. Both run end to end — load, tokenize, prefill and decode on CPU and
  CUDA, prefix reuse, OpenAI-compatible serving — and both are checked against
  llama.cpp on the same hardware.

  | | file | sha256 |
  |---|---|---|
  | Gemma 4 E4B-it | `gemma-4-E4B-it-Q3_K_M.gguf` (4,058,137,728 bytes) | `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4` |
  | Qwen2.5-0.5B-Instruct | `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes) | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |

  The GGUF's architecture string selects the plan and the chat format. There is
  no `if (arch == ...)` in the planner or the backends, and no model-specific
  code outside the GGUF loader and the plan emitter — the Qwen2 differences
  (a SiLU-gated FFN, full attention with no sliding window, a bias on each
  Q/K/V projection, an attention scale) are all generic ops and attributes, and
  the fallback to the token embedding when a file omits `output.weight` is
  generic too.
- **Quant formats:** Q3_K, Q4_K, Q5_K, Q6_K, Q5_0 and Q8_0. `f16` and `q4_0`
  tensors are converted to Q5_K at load time, which requires the row length to
  be a multiple of 256; anything else is refused rather than converted. Q6_K
  loads and decodes natively, but its prefill does not use the int8 tensor-core
  weight kernel yet — it dequantizes to fp16 and uses cuBLAS, so a Q6_K
  `ffn_down` makes prefill slower than the same shape in Q4_K.
- **Compiler tier:** Boom IR is the only rewrite path, and it now generates
  CUDA kernels through MLIR/NVVM for fused glue regions (norm, quantize,
  activation, residual, RoPE), prefill flash attention, split-KV decode
  attention and a bf16 dense GEMM. Generated kernels are checked against the
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
  Generated prefill attention stages K/V through a 2–3-stage cp.async pipeline
  (autotuned), pads the shared tiles, and skips the online-softmax rescale when
  no row maximum grew — the factor is exactly `exp(0)`, so that skip is
  bit-identical. An exp2 arm exists but is opt-in: it changes the token stream
  and does not meet the d64 fp64-error bound.
- **Speculative decoding (`boom-next`):** a CPU n-gram/prompt-lookup drafter
  proposes up to 7 tokens, a verify pass runs the decode graph over k+1 query
  rows with the same per-row kernels and reductions as plain decode, and KV
  entries computed from rejected drafts are rolled back by valid length. The
  emitted stream is bit-identical to greedy by construction
  (`test_spec_invariance`, `test_spec_rollback`). `SONICBOOM_SPEC=off|on|auto`;
  unset runs a per-SM startup probe (verify/plain ratio; auto below 1.5, off
  otherwise) and an explicit value bypasses it. `auto` chooses k from a
  measured verify-cost curve c(k) and an acceptance-rate EMA. On sm_86 the
  probe measures ~1.9, so the default is plain decode. On the benchmark traces
  `auto` selects k=0, so
  it does not change any number in this README; fixed k=7 is slower than plain
  decode on them.
- **GPU:** CUDA on sm_86. `CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the
  GPU in the build machine) and is resolved at configure time; other
  architectures have not been tested. Whatever was compiled is reported by
  `sb_build_info()` / `tools/sb-report.sh`, so a binary never has to be taken on
  trust about which SMs it contains. One community report ran on an RTX 4060 Ti
  (Ada) with decode tokens identical to the sm_86 results, see
  [Community results](#community-results). A CPU path exists but is a reference
  implementation, not a fast one.
- **Sampling:** greedy only. `temperature`, `top_p` and `top_k` return
  `SB_ERR_UNSUPPORTED`.
- **Chat template:** the serving layer picks the format from the loaded
  architecture — Gemma's `<|turn>…` turn markers, or Qwen2's ChatML — and takes
  the end-of-turn tokens from the model's metadata. `generate` itself still
  takes raw text, so a caller that does not go through the server formats its
  own turns.

---

## How much of this is ggml?

SonicBoom ports a small set of kernels from ggml/llama.cpp under the MIT
license. They live in `ggml_derived/` directories, with the details in
[`NOTICE`](NOTICE) and [`LICENSES/ggml-MIT.txt`](LICENSES/ggml-MIT.txt).

| | Lines of code |
|---|---:|
| ggml/llama.cpp-derived (`core/src/quant/ggml_derived`, `core/src/vocab/ggml_derived`) | 3,301 |
| SonicBoom core total (`core/src`, `core/include`) | 25,638 |
| **Derived share** | **~12.9%** |

*Counted with `cloc` 1.98. These are code lines only, excluding blanks and
comments. The C ABI, the Guile bindings and the dependencies are not
counted.*

**Ported:** quantized matrix multiply (MMVQ for decode, MMQ for prefill),
q8_1 activation quantization, the legacy Q5_0/Q8_0 block formats, and the BPE
tokenizers (SPM-style for Gemma 4, byte-level with the qwen2 pre-tokenizer,
plus the Unicode classes and byte maps they split on).

**SonicBoom's own:** everything else. That includes the planner and Boom IR
compiler passes, operator fusion, the generated GPU kernels (glue, attention
and bf16 dense GEMM), per-GPU runtime autotuning, CUDA Graph capture and
replay, the device-resident KV cache, split-KV flash decoding, tiled flash
attention for prefill, chunked prefill, prefix-cache reuse, and batch-invariant
dense GEMM.

Decode uses the same quantized GEMV kernels as llama.cpp and is still 13–20%
faster on the two models measured (see [Numbers](#numbers)). That difference
comes from the runtime, not from the kernels.

---

## Beyond ggml: a compiler stack, not a kernel collection

llama.cpp is fast because of hand-tuned kernels, and in that design every new
fusion or model-specific optimization means writing another kernel by hand.
SonicBoom chooses an implementation for each operation from three tiers, in
this order of preference:

| Tier | What it does | Status |
|---|---|---|
| **1. MLIR compiler** | Model graphs become Boom IR, which is rewritten and lowered to MLIR. GPU kernels are generated through `linalg → gpu → nvvm` and JIT-compiled by shape signature; S-Expr graphs still run on the CPU JIT, and ONNX-imported ResNet-18 matches the ONNX reference. | **Shipped**: fused glue, attention and a bf16 dense GEMM are generated on CUDA. |
| **2. Graph planner** | Rewrites the whole model before it runs: quantize-once, norm→quantize and activation→quantize fusion, dead-computation elimination, layout selection and CUDA Graph capture of the full decode step. | **Shipped**: this tier produces the decode lead measured above. |
| **3. Hand-written kernels** | Used where the other tiers are not yet faster, for example the ported ggml MMQ/MMVQ kernels and the tiled prefill attention. | **Shipped** |

What follows from this design:

- **Planner gains carry over to every model.** A pass written once applies to
  any architecture that is lowered into the planner, with no new kernel
  required. Gemma 4 and Qwen2 both run through the same planner and backends,
  with the architecture differences expressed as generic ops and attributes
  rather than model-specific code.
- **The kernels are no longer the limit.** With identical kernels, the
  runtime is already ahead of llama.cpp. Decode is still below the GPU's memory
  bandwidth limit for this model. Speculative decoding is now one more
  `boom.program` the planner lowers: drafter, row-invariant verify and KV
  rollback, with a token stream bit-identical to greedy decoding.
- **MLIR provides new hardware targets.** NVPTX is already built into the
  pinned LLVM. MLIR's GPU, SPIR-V/Vulkan and accelerator back ends are
  available to SonicBoom through the same S-Expr graphs, without a separate
  rewrite for each target.

---

## MLIR migration: a staged plan

MLIR is linked into `libsonicboom.so` and carries the production graphs: model
plans become Boom IR, the compiler's rewrite passes run on it, and CUDA kernels
are generated from it through `linalg → gpu → nvvm` and JIT-compiled. S-Expr
documents still lower to `arith`/`func`/`linalg` and run on the CPU through the
LLVM JIT, which is how ONNX-imported ResNet-18 runs. The migration is staged so
that each step is independently verifiable, and so that a generated kernel can
always be checked against the hand-written one it replaces.

| stage | what lands | status |
|---|---|---|
| **0. Lowering** | S-Expr → `arith`/`func`/`linalg`, executed on CPU through the LLVM JIT; ONNX ResNet-18 matches the ONNX reference | **shipped** |
| **1. One GPU glue kernel** | A single fused region — norm + quantize — lowered `linalg → gpu → nvvm → cubin`, JIT-compiled at load and cached by shape signature | **shipped** |
| **2. The glue set** | The rest of the between-kernel glue: activation→quantize, residual add, RoPE | **shipped** |
| **2b. Attention and dense GEMM** | Generated FA-2 prefill attention (cp.async pipeline, padded tiles, conditional rescale), split-KV decode attention and a bf16 dense GEMM with an mma m16n8k16 core, selected per shape class by the runtime tuner | **shipped**; generated decode attention is token-identical to the hand-written pair and is the default |
| **3. GPU-independent targeting** | The same lowering for a GPU the binary was not built for (compute capability discovered at run time), and for a second vendor through SPIR-V | not started |
| **4. Beyond LLM glue** | Whole-model lowering for the research route's traced graphs | not started |

Ground rules that hold at every stage: generated code and the JIT live **behind**
Layer 2 (no MLIR, LLVM, CUDA or native-torch type crosses the C ABI); the
hand-written kernel remains the numerical and performance reference; and the
gate is the canonical benchmark plus the existing test suite, because "it
compiles" and "it is faster" are not the same claim. The pinned LLVM is 23 —
a different major is a warning, not an error (see `CMakeLists.txt`).

---

## Scheme: write it eagerly, run it compiled

The research route keeps PyTorch's eager style without Python's limits. It is
built on two Scheme features that Python lacks: hygienic macros and delimited
continuations. It currently ships as a pure-Guile research prototype in
[`bindings/guile/scheme/sonicboom/trace`](bindings/guile/scheme/sonicboom/trace).

- **Data-dependent control flow without graph breaks.** A code walker written
  entirely in `syntax-rules` rewrites `if`, `cond`, `case`, `and`, `or`, `do`
  and named `let`, so a branch on a tensor value becomes a `where` node inside
  the graph. TorchDynamo needs a CPython bytecode interpreter to approach this
  problem; in SonicBoom it is a macro.
- **Continuations where macros cannot reach.** The tracer suspends at the
  branch with a delimited continuation. It can break the graph and resume with
  a concrete value, cache a trace tree per branch, or **resume the same
  continuation twice**, once for each branch, and merge the results. A Python
  generator cannot be resumed twice.
- **A whole training step is one graph.** Reverse-mode autodiff appends the
  backward pass to the traced forward pass, so the forward pass, loss,
  gradients and optimizer update compile together. The prototype already
  traces and trains a small Transformer.
- **PyTorch-compatible at the op level.** Every emitted node carries its Core
  ATen name, so graphs map directly onto native-torch's kernels.
- **Portable by construction.** It uses only R7RS `syntax-rules` and delimited
  continuations, with no Guile compiler internals, so other Scheme
  implementations can adopt it.

Next step: connect the tracer to the core, so that traced graphs are compiled
once through the C ABI and run on the same planner and backends as the
production route.

---

## Where this is going

One core and one C ABI, with a growing set of front ends:

```
 Scheme (eager, traced)  ─┐
 ONNX / PyTorch export   ─┼─→  S-Expr graph  ─→  planner  ─→  MLIR / kernels  ─→  CPU · CUDA · …
 GGUF models (C++ core)  ─┘
```

A model written once, by tracing, by import, or as a native definition, gets
every planner pass and every back end SonicBoom adds afterwards. That is the
difference between a fast LLM runner and a framework, and it is the direction
SonicBoom is built for.

---

## Architecture

Four layers, with dependencies pointing strictly downward:

```
Layer 3    C ABI / Guile / other bindings
Layer 2    SonicBoom stable runtime interface (nt::)
Layer 1    native-torch implementation + adapter
Layer 0    CPU / CUDA
```

`libsonicboom.so` is the published core library. native-torch and MLIR are
statically linked into it. native-torch's operator sources are generated at
build time by SonicCross, a torchgen written in Guile. The stable binary-compatibility boundary is the
C ABI (`capi/`); the C++ Layer 2 interface is source-level only.

Two routes share this core:

- **Production inference:** model architectures are defined in the C++ core.
  The execution planner and the Boom IR compiler lower them into a fused plan;
  generated GPU kernels are JIT-compiled for the detected device, and the
  decode step is captured as CUDA Graphs. The host language loads weights and
  drives sessions.
- **Research and training:** `(sonicboom nn)`, `(sonicboom model)` and
  `(sonicboom train)` give a minimal PyTorch/Keras-like API with native-torch
  reverse-mode autograd. Models can be exported as an S-Expr graph plus
  weights (`model-export!`) and run by the standalone C++ loader through the
  S-Expr → MLIR path, with no Guile or Python involved.

---

## Roadmap

- **Formats and models:** Q6_K, Q8_0 and Q5_0 are native and Qwen2 is a second
  verified family; next are further families (Llama, Mistral) and the quant
  formats they ship with
- **Speed:** prefill parity with llama.cpp was reached in v0.0.3 (1.04× on
  Gemma 4, 1.09× on Qwen2.5); next are a generated tensor-core FA-2 with
  llama-class tiles (attention is the remaining prefill gap), a token-identical
  generated d64 prefill path (the opt-in arm is +10% Qwen pp512), and a
  generated f32 dense GEMM. Speculative decoding is implemented and off by
  default; making its `auto` policy fire on real traces is the next decode
  step, aiming for an average decode rate around 100 tok/s on the same RTX 3050
- **Compiler:** GPU code generation for fused elementwise and norm chains is
  shipped; next are GPU-independent targeting (compute capability discovered
  at run time, SPIR-V for a second vendor) and whole-model lowering for traced
  graphs
- **Research route:** connect the Guile tracer to the core and compile traced
  graphs through the C ABI
- **Usability:** sampling — the runtime is still greedy only — on top of the
  chat templates the serving layer now selects per model
- **Scale:** larger MoE models with expert offload
- **Beyond LLMs:** diffusion models, with a converter for existing ComfyUI
  workflows
- **Distribution:** Guix packaging, CPU first. A Vulkan backend is planned so
  that a fully free-software stack also gets GPU acceleration. The backend
  scaffold is already in the tree (`-DSONICBOOM_USE_VULKAN=ON`).

---

## Commercial support

SonicBoom is developed by [HardenedVault](https://hardenedvault.com), which
specializes in Linux, compiler and systems security.

We offer consulting services for enterprise deployment and optimization:
private or air-gapped LLM inference, performance tuning on your hardware, and
hardened, auditable inference stacks. Contact
[consulting@hardenedvault.com](mailto:consulting@hardenedvault.com).

---

## License

SonicBoom is licensed under **GPL-3.0-or-later** (see [`LICENSE`](LICENSE)).
Code derived from ggml/llama.cpp keeps its MIT notice (see [`NOTICE`](NOTICE)).
ggml is not linked as a runtime dependency.
