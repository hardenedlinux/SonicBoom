# Architecture

Four layers, with dependencies pointing strictly downward:

```
Layer 3    C ABI / Guile / other bindings
Layer 2    SonicBoom stable runtime interface
Layer 1    native-torch implementation + adapter
Layer 0    CPU / CUDA
```

`libsonicboom.so` is the published core library. native-torch and MLIR are
statically linked into it. native-torch's operator sources are generated at
build time by SonicCross, a torchgen written in Guile. The stable
binary-compatibility boundary is the C ABI (`capi/`); the C++ Layer 2 interface
is source-level only. Layer 2 public headers expose no ATen/c10/native-torch
type, and generated code and the JIT live behind Layer 2.

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

## Why another runtime?

**For people who need to know what they are running.** The core is
hand-written C++/CUDA. It is GPL-3.0-or-later, and the boundary is a plain C
ABI that you can audit, pin and link against. It has no Python dependency chain
to vet, and nothing is downloaded at run time. Python is not needed to build it
either: PyTorch's Python code generator, torchgen, is replaced by
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

## A compiler stack, not a kernel collection

llama.cpp is fast because of hand-tuned kernels, and in that design every new
fusion or model-specific optimization means writing another kernel by hand.
SonicBoom chooses an implementation for each operation from three tiers, in
this order of preference:

| Tier | What it does | Status |
|---|---|---|
| **1. MLIR compiler** | Model graphs become Boom IR, which is rewritten and lowered to MLIR. GPU kernels are generated through `linalg → gpu → nvvm` and JIT-compiled by shape signature; S-Expr graphs still run on the CPU JIT, and ONNX-imported ResNet-18 matches the ONNX reference. | **Shipped**: fused glue, attention and a bf16 dense GEMM are generated on CUDA. |
| **2. Graph planner** | Rewrites the whole model before it runs: quantize-once, norm→quantize and activation→quantize fusion, dead-computation elimination, layout selection and CUDA Graph capture of the full decode step. | **Shipped**: this tier produces the decode lead measured in the README. |
| **3. Hand-written kernels** | Used where the other tiers are not yet faster, for example the ported ggml MMQ/MMVQ kernels and the `boom_fattn` prefill attention kernel. | **Shipped** |

What follows from this design:

- **Planner gains carry over to every model.** A pass written once applies to
  any architecture that is lowered into the planner, with no new kernel
  required. Gemma 4 and Qwen2 both run through the same planner and backends,
  with the architecture differences expressed as generic ops and attributes
  rather than model-specific code.
- **The kernels are no longer the limit.** With identical kernels, the
  runtime is ahead of llama.cpp on the d64 family. Speculative decoding is one
  more `boom.program` the planner lowers: drafter, row-invariant verify and KV
  rollback, with a token stream bit-identical to greedy decoding.
- **MLIR provides new hardware targets.** NVPTX is already built into the
  pinned LLVM. MLIR's GPU, SPIR-V/Vulkan and accelerator back ends are
  available to SonicBoom through the same S-Expr graphs, without a separate
  rewrite for each target.

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
| **2b. Attention and dense GEMM** | Generated split-KV decode attention and a bf16 dense GEMM with an mma m16n8k16 core, selected per shape class by the runtime tuner; prefill attention is served by the hand-written `boom_fattn` kernel | **shipped**; generated decode attention is token-identical to the hand-written pair and is the default |
| **3. GPU-independent targeting** | The same lowering for a GPU the binary was not built for (compute capability discovered at run time), and for a second vendor through SPIR-V | not started |
| **4. Beyond LLM glue** | Whole-model lowering for the research route's traced graphs | not started |

Ground rules that hold at every stage: generated code and the JIT live **behind**
Layer 2 (no MLIR, LLVM, CUDA or native-torch type crosses the C ABI); the
hand-written kernel remains the numerical and performance reference; and the
gate is the canonical benchmark plus the existing test suite, because "it
compiles" and "it is faster" are not the same claim. The pinned LLVM is 23 —
a different major is a warning, not an error (see `CMakeLists.txt`).

## Scheme: write it eagerly, run it compiled

The research route keeps PyTorch's eager style without Python's limits. It is
built on two Scheme features that Python lacks: hygienic macros and delimited
continuations. It currently ships as a pure-Guile research prototype in
[`bindings/guile/scheme/sonicboom/trace`](../bindings/guile/scheme/sonicboom/trace).

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

## How much of this is ggml?

SonicBoom ports a small set of kernels from ggml/llama.cpp under the MIT
license. They live in `ggml_derived/` directories, with the details in
[`NOTICE`](../NOTICE) and [`LICENSES/ggml-MIT.txt`](../LICENSES/ggml-MIT.txt).

| | Lines of code |
|---|---:|
| Verbatim upstream MMQ import (`core/src/quant/ggml_derived/mmq_upstream/`) | 8,798 |
| Other ggml/llama.cpp-derived (`core/src/quant/ggml_derived`, `core/src/vocab/ggml_derived`) | 3,899 |
| **ggml/llama.cpp-derived total** | **12,697** |
| SonicBoom core total (`core/src`, `core/include`) | 64,268 |
| **Derived share** | **~19.8%** |

*Counted with `cloc` 1.98. These are code lines only, excluding blanks and
comments. The C ABI, the Guile bindings and the dependencies are not counted.*

The verbatim import of llama.cpp's concurrent CUDA MMQ kernels
(`mmq_upstream/`, nine files) accounts for most of the derived share; it is
kept separate from the SonicBoom-written kernels so the provenance is explicit.
Without it the derived share is ~6.1%. The imported ggml flash attention and
its four-file subtree were removed in v0.0.4; prefill attention is SonicBoom's
own `boom_fattn` kernel.

**Ported:** quantized matrix multiply (MMVQ for decode, MMQ for prefill),
q8_1 activation quantization, the legacy Q5_0/Q8_0 block formats, and the BPE
tokenizers (SPM-style for Gemma 4, byte-level with the qwen2 pre-tokenizer,
plus the Unicode classes and byte maps they split on).

**SonicBoom's own:** everything else. That includes the planner and Boom IR
compiler passes, operator fusion, the generated GPU kernels (glue, attention
and bf16 dense GEMM), per-GPU runtime autotuning, CUDA Graph capture and
replay, the device-resident KV cache, split-KV flash decoding, the `boom_fattn`
tiled flash attention for prefill, chunked prefill, prefix-cache reuse,
batch-invariant dense GEMM, and speculative decoding (long-match drafter,
batched verify, KV rollback).

Decode uses the same quantized GEMV kernels as llama.cpp and is still faster
on the d64 family (see the README's [numbers](../README.md#numbers)). That
difference comes from the runtime, not from the kernels.

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
