# SonicBoom

**A native tensor runtime. No Python. One `.so`.**

SonicBoom is an AI framework written from the ground up in C++ and CUDA,
for workloads where performance matters. Everything is exposed through a
stable C ABI, and GNU Guile is its first-class language. You load
`libsonicboom.so` from whatever language you use. There is no Python
interpreter, no pip, and no wheel tree on the inference path.

The first release targets local LLM inference on consumer GPUs. On a laptop
RTX 3050 it **decodes 14% faster than llama.cpp, using the same quantized
kernels**. The kernels are the same, so the lead comes from the layers above
them: a graph planner, and behind it an MLIR compiler tier. Those layers are
what llama.cpp's architecture does not have, and they are where SonicBoom keeps
improving.

> **Status: v0.0.1.** Two model families are verified end to end — Gemma 4
> E4B-it (Q3_K_M) and Qwen2.5-0.5B-Instruct (Q4_K_M) — on CUDA sm_86, both
> checked against llama.cpp. Read [Scope of v0.0.1](#scope-of-v001) before
> trying it.

---

## Why another runtime?

**For people who need to know what they are running.** The core is
17.7k lines of C++/CUDA. It is GPL-3.0-or-later, and the boundary is a plain
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

Gemma 4 E4B-it Q3_K_M on an NVIDIA RTX 3050 (8 GB, sm_86). Both runtimes use
the same GGUF file on the same machine.

| | SonicBoom | llama.cpp | |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | **51.65 tok/s** | 45.20 tok/s | **1.14×** |
| Decode at depth 512 | 49.96 tok/s | 45.20 tok/s | 1.11× |
| Decode at depth 2048 | 47.13 tok/s | 45.21 tok/s | 1.04× |
| Prefill, 512 tokens | 1148 tok/s | ~1949 tok/s | 0.59× |
| Prefill, 2048 tokens | 1058 tok/s | ~1993 tok/s | 0.53× |
| Prefill, 4100 tokens | 1051 tok/s | ~1944 tok/s | 0.54× |
| **Time to first token, 4000-token prefix + 100 new tokens** | **502.6 ms** with prefix reuse | | 7.8× faster than a full prefill (3903 ms) |

- **Decode is ahead.** Prefill is at roughly half of llama.cpp. It is a real
  batched path (int8 tensor-core MMQ plus cuBLAS), and closing the rest of the
  gap is on the roadmap.
- **The public API is as fast as the C++ internals.** Measured through the
  Guile binding, decode runs at 52.15 tok/s, a 4100-token prefill takes
  3906.9 ms, and reuse TTFT is 505.2 ms. All three are within 1% of the C++
  numbers.
- **Exactness.** A prefix-reuse continuation matches a full prefill token for
  token at every tested prefix length (`test_prefix_cache`, 11/11).

### Benchmark conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB, compute 8.6 |
| Driver / CUDA | 595.84 (CUDA 13.2 driver API); nvcc 12.0.140 |
| CPU | AMD Ryzen 5 5500 (6C/12T, Zen 3), PCIe 3.0 |
| RAM | 32 GB DDR4-2400, single channel |
| Host | Linux 6.8, x86-64, g++-13 |
| Model | [`unsloth/gemma-4-e4b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF) → `gemma-4-E4B-it-Q3_K_M.gguf` (4,058,137,728 bytes, sha256 `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4`) |
| llama.cpp | commit `43fe9c64281`, CUDA build, `llama-bench -ngl 99` |

The comparison is against llama.cpp at the commit above. Newer llama.cpp
versions may be faster or slower. Exact command lines, per-kernel breakdowns
and the full history are in [`design/benchmark.md`](design/benchmark.md).

### Qwen2.5-0.5B-Instruct Q4_K_M

A second architecture on the same machine, the same runtime and the same
llama.cpp commit.

| | SonicBoom | llama.cpp | |
|---|---:|---:|---:|
| **Decode**, 64-step steady state | 255.7 tok/s | 307.89 ± 1.43 tok/s | 0.83× |
| Decode at depth 512 | 224.2 tok/s | 304.10 ± 3.67 tok/s | 0.74× |
| Prefill, 512 tokens | 4036.6 tok/s (126.84 ms) | 14505.81 ± 1063.36 tok/s | 0.28× |
| **Time to first token, 512-token prefix + 64 new tokens** | **68.4 ms** with prefix reuse | | 4.4× faster than a full prefill (299.3 ms) |

Here SonicBoom is behind on both decode and prefill. The depth-0 → depth-512
decode drop (255.7 → 224.2, −12%) is attention over all 512 keys: Qwen2 has no
sliding window for the Gemma ring to wrap. The prefill gap is the batched
K-quant matmul, plus this file's Q6_K `ffn_down` adding its fp16-dequant
fallback on top. Prefix reuse transfers unchanged and is the strongest result —
with no ring buffer, a resident prefix is a plain KV slice. Reuse is against
SonicBoom's own full prefill; llama-bench has no reuse measurement.

The Qwen2 model is also where the tokenizer, the ChatML template and the
architecture-driven chat-format selection were exercised. Its greedy token
stream agrees with llama.cpp except at llama.cpp's own near-ties, where a
different prefill batch geometry flips the last-place logits (first divergent
step and top-1/top-2 margin recorded in `design/benchmark.md`).

---

## Quick start

### Requirements

- Linux x86-64, g++-13, CMake, Ninja (optional), libyaml-dev
- CUDA toolkit 12.0 or newer and an NVIDIA GPU. v0.0.1 is built and tested
  for **sm_86** only (RTX 30-series).
- GNU Guile 3.0
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
tools/mlir/build-mlir.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSONICBOOM_USE_CUDA=ON
cmake --build build -j
```

The LLVM submodule is large, so the first clone and build take a while.

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
their checksums are in [Scope of v0.0.1](#scope-of-v001)).

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
ctest --test-dir build --output-on-failure
```

Tests that need the model skip themselves when the GGUF is not present.

---

## Scope of v0.0.1

v0.0.1 is a working runtime, not a drop-in replacement for llama.cpp. Its
limits:

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
- **GPU:** CUDA on sm_86. Other architectures need `CMAKE_CUDA_ARCHITECTURES`
  changed and have not been tested. A CPU path exists but is a reference
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
| ggml/llama.cpp-derived (`core/src/quant/ggml_derived`, `core/src/vocab/ggml_derived`) | 2,820 |
| SonicBoom core total (`core/src`, `core/include`) | 19,953 |
| **Derived share** | **~14.1%** |

*Counted with `cloc` 1.98. These are code lines only, excluding blanks and
comments. The C ABI, the Guile bindings and the dependencies are not
counted.*

**Ported:** quantized matrix multiply (MMVQ for decode, MMQ for prefill),
q8_1 activation quantization, the legacy Q5_0/Q8_0 block formats, and the BPE
tokenizers (SPM-style for Gemma 4, byte-level with the qwen2 pre-tokenizer,
plus the Unicode classes and byte maps they split on).

**SonicBoom's own:** everything else. That includes the planner and graph
passes, operator fusion, CUDA Graph capture and replay, the device-resident KV
cache, split-KV flash decoding, tiled flash attention for prefill, chunked
prefill, prefix-cache reuse, and batch-invariant dense GEMM.

Decode uses the same quantized GEMV kernels as llama.cpp and is still 14%
faster. That difference comes from the runtime.

---

## Beyond ggml: a compiler stack, not a kernel collection

llama.cpp is fast because of hand-tuned kernels, and in that design every new
fusion or model-specific optimization means writing another kernel by hand.
SonicBoom chooses an implementation for each operation from three tiers, in
this order of preference:

| Tier | What it does | Status |
|---|---|---|
| **1. MLIR compiler** | S-Expr graphs are lowered to MLIR (`linalg`) and compiled to native code. ResNet-18, imported from ONNX, runs through this path and matches the ONNX reference. | **Shipped on CPU** (LLVM JIT). GPU code generation is next. |
| **2. Graph planner** | Rewrites the whole model before it runs: quantize-once, norm→quantize and activation→quantize fusion, dead-computation elimination, and CUDA Graph capture of the full decode step. | **Shipped**: this tier produces the 14% decode lead. |
| **3. Hand-written kernels** | Used only where the other tiers are not yet fast enough, for example the ported ggml MMQ/MMVQ kernels. | **Shipped** |

What follows from this design:

- **Planner gains carry over to every model.** A pass written once applies to
  any architecture that is lowered into the planner, with no new kernel
  required. Gemma 4 and Qwen2 both run through the same planner and backends,
  with the architecture differences expressed as generic ops and attributes
  rather than model-specific code.
- **The kernels are no longer the limit.** With identical kernels, the
  runtime is already ahead of llama.cpp. Decode is still below the GPU's memory
  bandwidth limit for this model. Speculative decoding fits in as one more graph
  for the planner to schedule.
- **MLIR provides new hardware targets.** NVPTX is already built into the
  pinned LLVM. MLIR's GPU, SPIR-V/Vulkan and accelerator back ends are
  available to SonicBoom through the same S-Expr graphs, without a separate
  rewrite for each target.

---

## Scheme: write it eagerly, run it compiled

The research route keeps PyTorch's eager style without Python's limits. It is
built on two Scheme features that Python lacks: hygienic macros and delimited
continuations. It currently ships as a pure-Guile research prototype in
[`bindings/guile/scheme/sonicboom/trace`](bindings/guile/scheme/sonicboom/trace),
documented in [`design/guile-tracing-prototype.md`](design/guile-tracing-prototype.md).

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
  The static execution planner lowers them into a fused plan that is captured
  as CUDA Graphs. The host language loads weights and drives sessions.
- **Research and training:** `(sonicboom nn)`, `(sonicboom model)` and
  `(sonicboom train)` give a minimal PyTorch/Keras-like API with native-torch
  reverse-mode autograd. Models can be exported as an S-Expr graph plus
  weights (`model-export!`) and run by the standalone C++ loader through the
  S-Expr → MLIR path, with no Guile or Python involved. See
  [`design/guile-ai-framework-progress.md`](design/guile-ai-framework-progress.md).

---

## Roadmap

- **Formats and models:** Q6_K, Q8_0 and Q5_0 are native and Qwen2 is a second
  verified family; next are further families (Llama, Mistral) and the quant
  formats they ship with
- **Speed:** prefill parity with llama.cpp; speculative decoding, aiming for
  an average decode rate around 100 tok/s on the same RTX 3050
- **Compiler:** MLIR GPU code generation for fused elementwise and norm
  chains
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
