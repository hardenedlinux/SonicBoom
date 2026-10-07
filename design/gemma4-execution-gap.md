# Gemma 4 E2B — execution capability audit

> **Native `nn` path (2026-10-06):** a second, pure-C++23 reference path now
> exists at `sonicboom::nn` (`core/include/sonicboom/nn/`, `core/src/nn/`),
> independent of the S-Expr → MLIR path audited below. It provides the
> transformer-block primitives — RMSNorm, RoPE (NeoX), SiLU/GELU, elementwise
> mul/scale/add/axpy, softmax (+causal), grouped-query attention with
> sliding-window + logit softcapping, logit softcap + argmax, and token
> embedding — plus `quant::dequantize_row_f32` for single-row dequant. All are
> tested in `tests/core/test_nn.cpp` / `test_dequant.cpp` and land in
> `libsonicboom.so`. The matrix below describes the S-Expr/MLIR path only; see
> `design/gemma4-model-map.md` for the target model's architecture.

Scope: code-backed assessment of whether SonicBoom can execute a Gemma 4 E2B
model today. No inference engine is implemented here; this documents what
exists and what is missing, distinguishing five levels:

1. GGUF parsing,
2. quantized tensor representation / data access,
3. kernel-level numerical execution,
4. complete model-architecture support,
5. end-to-end inference.

Classification key: `IMPLEMENTED_AND_TESTED`, `IMPLEMENTED_NOT_TESTED`,
`PARTIALLY_SUPPORTED`, `MISSING`, `UNKNOWN`. Every entry cites source.

## Current execution surface (ground truth)

SonicBoom has three independent compute paths, each with its own op vocabulary:

- **S-Expr → MLIR (CPU JIT)** — the deployment engine. 7 ops lowered
  (`core/src/sx/lowering.cpp:637-643`): `relu`, `add`, `reshape`, `conv`,
  `max_pool`, `reduce_mean`, `gemm`. Dtypes: all ten in `sx::DType`
  (`core/include/sonicboom/sx/ir.h:39-52`; `lower_elem_type`,
  `lowering.cpp:71-85`).
- **native-torch boxed dispatch** — the S-Expr path routes **only** `softmax`
  here (`aten::_softmax`, `core/src/planner/native_torch_backend.cpp:123`), with
  a dtype bridge limited to f32/f64/i64/i32/bool
  (`sx_to_nt`, `native_torch_backend.cpp:43-52`). Separately, the Layer 1
  adapter proves these `aten::` operators dispatch and compute
  (`tests/core/test_native_dispatch.cpp`, `tests/core/test_native_ops.cpp`):
  `add`, `mul`, `relu`, `mm`, `_softmax`, `max`, `index_select`, `conv2d`,
  `to.dtype`.
- **autograd (training)** — `linear` (mm + transpose + bias), `relu`,
  `mse` (sub/mul/mean), SGD, VJP replay (`core/src/autograd/autograd.cpp`,
  `core/include/sonicboom/autograd.h`). Not a deployment path.

The S-Expr operator vocabulary is the 8-name set in
`core/include/sonicboom/planner/op_kind.h:24-33` (`conv relu add max_pool
reduce_mean reshape gemm softmax`); a name outside it is rejected
(`tests/core/test_planner_graph.cpp:135` sets `op_name = "gelu"` and asserts
rejection).

## Capability matrix

| Area | Status | Evidence |
|---|---|---|
| GGUF parsing (header/KV/tensor dir/offsets) | IMPLEMENTED_AND_TESTED | `core/src/gguf/reader.cpp`; `tests/core/test_gguf.cpp`; real `hift.gguf`/`llm.gguf` parsed |
| Quantized tensor **data access** (raw bytes, size) | IMPLEMENTED_NOT_TESTED | `ggml_type_size` + `tensor_data()` (`reader.cpp`); table ids 0–30 incl. `q4_K`/`q5_K`; byte-size tested, real quantized model not available |
| Quantized tensor **representation** (typed value) | IMPLEMENTED_AND_TESTED | `sonicboom::quant::QuantizedTensor` + `quantized_tensor_from_gguf` (`core/include/sonicboom/quant/quantized_tensor.h`); exercised by `tests/core/test_dequant.cpp` |
| Quantized **dequantization** (Q3_K, Q4_K, Q5_K) | IMPLEMENTED_AND_TESTED | `core/src/quant/dequant.cpp`; bit-exact vs ggml baseline (random + real-model differential, `tests/core/test_dequant.cpp`) |
| Quantized GEMM / quantized matmul | PARTIALLY_SUPPORTED | CPU reference `quant::matvec_f32` (quantized weight @ f32 vector, on-the-fly dequant) in `core/src/quant/quantized_matmul.cpp`; `tests/core/test_quantized_matmul.cpp`. Matmul (batched) + CUDA + runtime selection still open |
| RMSNorm / LayerNorm | MISSING | no norm op in `op_kind.h`; `reduce_mean` present but no `pow`/`sqrt`/`rsqrt`/`mul` graph ops to compose it |
| RoPE | MISSING | no `rope`/`rotary` symbol anywhere in `core/ capi/ bindings/` |
| GELU / SiLU / SwiGLU | MISSING | only `relu`; `gelu` explicitly rejected (`test_planner_graph.cpp:135`) |
| Embedding / gather | PARTIALLY_SUPPORTED | no graph op; primitive `aten::index_select` dispatches (`test_native_ops.cpp:142`) but is not exposed through S-Expr/MLIR |
| Transpose / elementwise mul | PARTIALLY_SUPPORTED | `mul` dispatches (`aten::mul`, `test_native_dispatch.cpp:44`); `transpose` exists only as 2-D internal helper for gemm/linear (`lowering.cpp:294`, `autograd.cpp:55`); neither is a graph op |
| Attention (scaled-dot, sliding-window, shared-KV) | MISSING | only `gemm`+`softmax` primitives; no scale `mul`, no mask, no QK-norm, no windowing |
| KV-cache management | MISSING | no cache structure / incremental-decode symbol |
| MoE routing / experts | MISSING | no `moe`/`expert`/router symbol |
| Logit softcapping / output scaling / decode | MISSING | no softcap/scaling/`top_k`/`top_p`/argmax-token-loop symbol |
| gemma4 tensor naming / shape / weight loading | PARTIALLY_SUPPORTED | generic GGUF name/shape/offset parsing works; no `gemma4` architecture binding, no weight→graph loader |

### Notes on the UNKNOWN boundary

Whether the migrated native-torch subset contains `aten::gelu`, `aten::silu`,
`aten::native_layer_norm`, `aten::embedding`, or a fused
`aten::scaled_dot_product_attention` is **not verified** here — that would
require inspecting the native-torch tree, which is out of scope for this audit
and is a controlled migration area. Even if those kernels are present, they are
not reachable through SonicBoom's S-Expr/MLIR graph vocabulary today, so the
architecture-level status above is unchanged.

## End-to-end status

A real model is now present: `models/gemma-4-E4B-it-Q3_K_M.gguf` (Gemma 4
E4B-It, 4.06 GB, GGUF v3, 720 tensors). The reader parses it completely and all
720 tensor data spans resolve (`bad=0`). This confirms the model's weights are
**mixed-quantized** — `q3_K`×169, `q4_K`×123, `q5_K`×4, `f32`×423, `bf16`×1 —
so dequantization of `q3_K`/`q4_K`/`q5_K` is a hard prerequisite, not optional.
No end-to-end claim is made and none is possible yet: the graph op set lacks
every transformer-block primitive, so execution would fail at the first missing
op. Current state: **GGUF bytes parse; nothing above `gemm`/`softmax`/`add`/
`reshape`/`reduce_mean` is assembled into a transformer block.**

## Smallest steps to a first-token test (prefill only; no KV-cache needed)

Ordered by dependency, smallest first:

1. **Obtain the target Gemma 4 GGUF** — done: `models/gemma-4-E4B-it-Q3_K_M.gguf`
   (E4B-It, 720 tensors). Its dtypes are recorded above (`q3_K`/`q4_K`/`q5_K`/
   `f32`/`bf16`), which resolves the step-8 gate: the weights are quantized, so
   dequant is required.
2. **Add elementwise `mul` (+ `rsqrt`/`sqrt`)** to the S-Expr set and MLIR
   lowering (`arith.mulf`, `math.rsqrt`). Unblocks RMSNorm, attention scaling,
   and SiLU composition. `aten::mul` already dispatches.
3. **RMSNorm** — compose from `reduce_mean` + `mul` + `rsqrt` (or route a native
   norm kernel if present).
4. **Embedding** — expose `aten::index_select` as a graph op (already proven at
   the boxed-dispatch level).
5. **RoPE** — compose (needs `sin`/`cos` or a precomputed table + `mul`/`add`).
6. **SiLU/SwiGLU activation** — compose (`x·sigmoid(x)`, needs `sigmoid`/`exp`)
   or route a native activation if present.
7. **Assemble attention** — QKV `gemm` + `reshape`/split + `softmax` + scale
   `mul` + `gemm`, with causal/sliding mask and shared-KV head handling.
8. **Dequantize Q3_K/Q4_K/Q5_K** — done: `sonicboom::quant` (Phase 1–2), bit-exact
   vs the ggml baseline including the negative-subnormal scale edge case.
9. **First-token decode** — output `softmax` (have) + `argmax`/greedy token.

Steps 2–7 are the operator/assembly work; 1 and 8 are gated on the model file.
None are implemented in this task.
