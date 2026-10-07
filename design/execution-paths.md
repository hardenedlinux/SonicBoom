# SonicBoom execution paths

SonicBoom has **two execution paths** that share one compute/planning spine. This
document makes the two paths and their shared spine explicit, maps the current
code to each, and lists what Path 2 still needs to connect to the spine.

## The two paths

```text
Path 1 — research / learning (flexible, programmable)
-----------------------------------------------------
Guile module (user-authored model/ops)
  -> S-Expr v0.1 graph
  -> planning (adapt_graph + plan_execution + backend selection)
  -> { MLIR JIT (CpuBackend) , native-torch (NativeTorchBackend) }

Path 2 — performance / deployment (high-throughput, deployable)
---------------------------------------------------------------
GGUF loading (gguf::Reader + load_gemma4)
  -> planning (ExecutionPlan)
  -> model-specific inference framework (Gemma 4 semantics)
  -> { MLIR + native-torch } high-performance compute (incl. CUDA)
```

The two paths differ only in the **frontend** — how a model becomes a graph /
plan. They converge on the same **spine**:

```text
                      shared spine
   +------------------------------------------------------+
   |  planning  (static execution planner)                 |
   |  native-torch (Layer 1 compute kernels, CPU/CUDA)     |
   |  MLIR       (lowering + JIT)                          |
   +------------------------------------------------------+
```

"core" is this spine. Its job is to make "add a model" mean "add a graph /
plan", not "write another C++ loop".

## Path 1 — research / learning

Frontend = Guile -> S-Expr; the goal is flexibility (write a model in Scheme and
run it).

| stage | files |
|---|---|
| Guile binding | `bindings/guile/scheme/sonicboom/{ffi,nn,tensor}.scm` |
| C ABI | `capi/include/sonicboom/capi.h`, `capi/src/capi.cpp` (`sb_parse`/`sb_compile`/`sb_execute`) |
| S-Expr IR + parser + lowering | `core/src/sx/{parser,ir,sexpr,lexer,lowering}.cpp` |
| planner (graph + partition + cost) | `core/src/planner/{graph_adapter,planner,partition,cost_model,memory_planner}.cpp` |
| runtime facade | `core/include/sonicboom/runtime.h`, `core/src/planner/runtime.cpp` (`sonicboom::Model`) |
| MLIR backend | `core/src/planner/cpu_backend.cpp` -> `core/src/sx/exec.cpp` (MLIR JIT) |
| native-torch backend | `core/src/planner/native_torch_backend.cpp` -> `core/layer1/adapter/operator_handle.cpp` (c10 dispatcher) |
| executor | `core/src/planner/runtime_executor.cpp` |

Status: **end-to-end working** (ResNet-18 S-Expr exec, co-execution region
tests, autograd via C ABI).

## Path 2 — performance / deployment

Frontend = GGUF -> model framework; the goal is high-performance inference of a
concrete quantized model (currently Gemma 4 E4B).

| stage | files |
|---|---|
| GGUF loading | `core/src/gguf/reader.cpp` |
| model loader (config + weights + plan) | `core/src/model/loader.cpp` (`load_gemma4`) |
| model-specific inference framework | `core/src/model/exec.cpp` (`run_block_core`, `forward`, `forward_traced`, `embed_token`, `embed_per_layer`, `lm_head`) |
| model config / per-layer plan | `core/include/sonicboom/model/config.h` (`Gemma4Config`, `LayerConfig`) |
| weight views | `core/include/sonicboom/model/weights.h` |
| reference primitives | `core/src/quant/{dequant,quantized_matmul,quantized_tensor}.cpp`, `core/src/nn/*.cpp` |

Status today:

- **GGUF loading: done.** `gguf::Reader` + `load_gemma4` parse the model and bind
  every tensor.
- **Model framework: done, but as a scalar loop.** `run_block_core` is a
  hard-coded per-block C++ sequence that calls `nn::`/`quant::` scalar kernels
  directly. It is numerically correct (bit-exact vs. the llama.cpp oracle —
  see `design/gemma4-semantics.md` §10–§11), but it **bypasses the planner** and
  does **not** use the native-torch / MLIR compute spine.
- **Planning: not connected.** The "plan" is `Gemma4Model::plan` (`LayerConfig`
  per layer), which is unrelated to `planner::ExecutionPlan`.
- **MLIR + native-torch compute: not connected.** The heavy quantized matmuls
  and norms run as SonicBoom's scalar reimplementations, not native-torch
  kernels.

So Path 2 currently has its frontend and a correct-but-standalone compute loop;
the spine is not yet attached.

## The shared spine

Both paths should end in the same three components:

1. **planning** — `core/src/planner/`. Static execution planning: graph
   adaptation, region partition, backend selection, memory planning, cost model.
2. **native-torch** — `core/layer1/` (the Layer 1 adapter, the only SonicBoom
   code that includes ATen/c10) over `third_party/native-torch/`. Boxed operator
   registration + dispatch: `nt::define_operator` / `nt::register_kernel` /
   `nt::find_operator` / `OperatorHandle::call`.
3. **MLIR** — `core/src/sx/exec.cpp` (lowering + JIT), built from
   `third_party/llvm-project` via `tools/mlir/build-mlir.sh`.

## The operator-vocabulary gap (the concrete blocker)

The spine is *generic*, but it only knows the operators it has been taught. The
planner's operator vocabulary is `core/include/sonicboom/planner/op_kind.h`:

```text
Conv, Relu, Add, MaxPool, ReduceMean, Reshape, Gemm, Softmax, WholeGraph
```

This is the S-Expr v0.1 / ResNet-18 operator set. **None of the transformer
operators are in it.** The native-torch backend maps a single op today
(`Softmax` -> `aten::_softmax`, `native_torch_backend.cpp`). Consequently the
Gemma 4 model framework *cannot* yet be expressed against the spine — there is
no `rms_norm`, `quantized_matmul`, `rope`, or `attention` operator for it to
dispatch.

This is the root cause of the "one model = one bespoke loop" situation: the
model framework was built first, before the spine's operator vocabulary covered
it.

## Path 2 gap inventory

Mapping `run_block_core` + `lm_head` (`core/src/model/exec.cpp`) onto the
operators the spine needs:

| model op (exec.cpp call) | required spine operator | priority |
|---|---|---|
| `quant::matvec_f32_q8_K` (q/k/v/o/ffn/token_embd) | quantized matmul q*_K @ q8_K | **highest** (the dominant cost; the CUDA/Phase-6 enabler) |
| `nn::rms_norm` | rms_norm (weighted / unweighted) | high |
| `nn::gelu_fp16` | gelu (GGML_GELU_FP16 table) | high |
| `nn::rope_neox` | rope (NEOX, freq_factors) | high |
| `nn::scaled_dot_product_attention` | SDPA (GQA, causal, sliding-window) | high (needs native-torch flash-attn path for Phase 6) |
| `nn::add` / `nn::mul` / `nn::scale` | elementwise add/mul/scale | medium |
| `quant::dequantize_row_f32` | embedding lookup (dequant row) | medium |
| `embed_per_layer` bf16 matvec (inline in exec.cpp) | bf16 matvec | medium |
| lm-head softcap (scale/tanh/scale) | softcap (elementwise) | low |

Each row becomes: an `OpKind`, a Layer 2 operator schema
(`nt::define_operator`), and a native-torch kernel (`nt::register_kernel`) —
or, where native-torch has no suitable kernel, an MLIR lowering.

## What "using the spine" means for Path 2

When the vocabulary gap closes, the model-specific inference framework stops
being the compute engine and becomes the **model-aware plan emitter**: it turns
"Gemma 4's 42 blocks, shared-KV, per-layer gates, weight-tied lm-head" into a
graph/plan the planner can schedule, and the per-op compute runs on
native-torch / MLIR. The model-specific semantics (which layers share KV, the
per-layer projection, the gate order) stay in the model layer; the generic
compute does not.

```text
GGUF loading -> model framework emits plan -> planning -> native-torch / MLIR
                                              (shared spine)
```

## Boundary rules (unchanged)

- The model framework must not reach into `third_party/native-torch/` directly;
  it goes through the Layer 2 operator interface (`nt::`).
- Transformer ops added to the spine belong in `core/include/sonicboom/planner/op_kind.h`
  and the Layer 2 operator surface, not as free scalar functions in `exec.cpp`.
- llama.cpp / ggml remains a dev-time oracle only (`tools/oracle/`); it is never
  a runtime dependency of `libsonicboom.so`.
