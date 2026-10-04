# SonicBoom S-Expr v0.1 → MLIR Lowering

This document records the design of the S-Expr → MLIR lowering: it turns a
validated `sx::Document` (see `s-expr-v0-spec.md`) into a **verified MLIR
module** and returns its textual form. It is the link in the pipeline

```text
ONNX / Scheme → S-Expr → MLIR → Runtime
```

and it **stops at the verified module** — no MLIR optimization passes, no code
generation, no CPU/GPU backend selection, no scheduling, no memory planning,
no JIT, no Native-Torch execution. Everything below the verified module is
out of scope here.

## Files

- `core/include/sonicboom/sx/lowering.h` — public facade (MLIR-free).
- `core/src/sx/lowering.cpp` — implementation (MLIR types stay behind this
  boundary and `core/src/sx/exec.cpp`).
- `core/src/sx/lowering_internal.h` — internal `lower_into_module` +
  `WeightMap`, shared with the execution path (see `cpu-execution-mvp.md`).

## Goal and boundary

Input: a validated `sx::Document` (already parsed + semantically checked).

Output: `std::expected<std::string, LoweringError>` — on success, the textual
form of an MLIR module that has passed full MLIR verification; on failure, a
structured error. A successful return therefore implies **both** lowering and
verification succeeded.

Deliberately **not** done here: anything that consumes the module
(optimization, codegen, execution), any runtime weight loading, and any
dynamic/symbolic dimension support.

## Dialects

`builtin` + `func` + `arith` + `tensor` + `linalg`. This is the smallest set
needed for the frozen v0 operator subset. **No custom SonicBoom dialect** is
introduced — the namespace is not owned by a dialect; module-level metadata is
carried as an ordinary attribute.

## Type mapping

Every value is a tensor with a static, non-negative shape. The mapping is
direct: `sx::TensorType { DType, Shape }` → `mlir::RankedTensorType`.

| S-Expr dtype | MLIR element type |
|---|---|
| `float32` | `f32` (`Float32Type`) |
| `float16` | `f16` (`Float16Type`) |
| `bfloat16` | `bf16` (`BFloat16Type`) |
| `float64` | `f64` (`Float64Type`) |
| `int8`  | `i8` (signless `IntegerType(8)`) |
| `uint8` | `i8` (signless) |
| `int16` | `i16` (signless `IntegerType(16)`) |
| `int32` | `i32` (signless `IntegerType(32)`) |
| `int64` | `i64` (signless `IntegerType(64)`) |
| `bool`  | `i1` (`IntegerType(1)`) |

Notes:

- **Signedness is not materialized.** The frozen dtype set distinguishes
  signed (`int8/16/32/64`) from unsigned (`uint8`), but no v0 operator depends
  on the distinction, so all integer dtypes lower to signless MLIR integers.
  This is a documented limitation, not an oversight.
- **Static shapes only.** There are no dynamic or symbolic dimensions in v0;
  a shape carries concrete `int64_t` extents. `RankedTensorType` is always
  fully static.

## Graph structure

One S-Expr graph lowers to **one `func.func`** (named after the graph):

- **graph `inputs`** → function arguments, in declaration order.
- **parameters** → materialized in the entry block (see below).
- **node `outputs`** → intermediate SSA values, in topological order.
- **graph `outputs`** → `func.return` values, in declaration order.

Value mapping is a flat SSA namespace: every name defined by an input, a
parameter, or a node output is looked up in one map. Types come from the
definition site (the S-Expr carries an explicit type on every value); the
lowering **never infers a missing type** and errors if a reference is
undefined.

### Parameters

- **`:values`** → `arith.constant` with a dense literal; the element type comes
  from the parameter's dtype. The literal list must be homogeneous (int64 or
  float64).
- **`:external`** → `tensor.empty` placeholder of the parameter's type. The
  tensor's identity/dtype/shape are preserved on the value; the sidecar
  reference is recorded on the module (see External data). There is **no
  runtime weight loader** — the values are intentionally not read.

## Operator mapping

Lowering is **per-operator** (`lower_relu`, `lower_add`, `lower_reshape`,
`lower_conv`, `lower_max_pool`, `lower_reduce_mean`, `lower_gemm`), dispatched
through a lookup table — not a monolithic switch — so adding an operator means
adding one function plus one table entry. Each lowerer returns `""` on success
or an error string, and writes its output value into the SSA map.

| S-Expr op | MLIR lowering |
|---|---|
| `relu` | `arith.maximumf(x, 0.0)` (zero is a splat constant) |
| `add` | `arith.addf(a, b)` |
| `reshape` | `tensor.reshape` (target shape is the output type's static shape, as a `tensor<Nxi64>` shape operand) |
| `conv` | `tensor.pad` → `linalg.conv_2d_nchw_fchw` → `linalg.generic` bias broadcast-add |
| `max_pool` | `tensor.pad` → `linalg.pooling_nchw_max` |
| `reduce_mean` | `linalg.generic` (sum reduction) → `arith.divf` |
| `gemm` | `linalg.matmul` (+ `linalg.transpose`, `arith.mulf`, `linalg.generic` bias) |

### Convolution (`conv`)

`conv` takes three inputs `(x, weight, bias)`; the weight is `[F, C, KH, KW]`
(NCHW input, FCHW filter). `linalg.conv_2d_nchw_fchw` is a *valid* convolution
(no padding, no bias), so:

1. **Padding** is done explicitly with `tensor.pad`, using the **exact ONNX
   pads** `[top, left, bottom, right]` (spatial dims only; `0` on N and C).
2. The valid convolution then reproduces ONNX's `floor` output semantics on
   its own — no output-size recomputation is needed, because both ONNX and
   linalg compute `floor((padded - kernel) / stride) + 1`.
3. **Bias** (rank-1, `[F]`) is added by a `linalg.generic` broadcast along the
   channel axis (dim 1), since `conv_2d_nchw_fchw` has no bias operand.

Restricted to `group == 1` and `dilations == 1` (explicit error otherwise).
`strides` and `dilations` must each be exactly 2 entries and `pads` exactly 4;
the sizes are validated **before** any indexing, so a malformed attribute
produces a structured `Operator` error rather than an out-of-bounds read.

### Max pooling (`max_pool`)

`max_pool` takes one input; `linalg.pooling_nchw_max` is also a *valid* pooling,
so padding uses the same "exact ONNX pads" convention as conv — but with the
**`-∞` pad value**, not zero. ONNX pads max-pool windows with NaN (treated as
`-∞`); padding with `0.0` instead would drag negative inputs toward zero, so
the pad constant is `-∞`, matching the accumulator identity. The pooling kernel
is carried as a dummy `[KH, KW]` operand (`tensor.empty`): its **shape** drives
the window size, and its values are never read by the max reduction (the region
uses only the input and the accumulator). Restricted to `dilations == 1` and
`ceil_mode == 0` (explicit error otherwise); `kernel_shape` must be 2 entries,
`pads` 4, and `strides`/`dilations` 2, validated before indexing.

### Reduce mean (`reduce_mean`)

`reduce_mean(x, axes)` requires an explicit `axes` input that references a
`:values` int64 parameter (negative axes are normalized). **Duplicate axes are
rejected** (a repeated axis would otherwise double-count into the divisor). The
reduction is a `linalg.generic` with `reduction` iterators over the reduced
dims and `parallel` iterators elsewhere, summing into an accumulator; the
result is then divided by the element count with `arith.divf`. `keepdims` is
honored implicitly: the output type keeps the reduced dims at size `1`.

### GEMM (`gemm`)

`gemm(A, B[, C])` lowers to `linalg.matmul`, with:

- `linalg.transpose` on A/B when `transA`/`transB` is set,
- `arith.mulf` for a non-unit `alpha` / `beta`,
- a trailing-axis `linalg.generic` broadcast-add for the bias C.

v0 gemm is strictly 2-D: A and B must both be rank-2 (the bias broadcasts along
the trailing axis). The rank is validated **before** `transpose_2d`/`matmul`
touch the dimensions, so a rank-mismatched input is an explicit error.

## External data

`:external` parameters are not loaded. Their sidecar reference is preserved as
a module-level attribute:

```text
#sonicboom.external_data = { <name> = { file = "...", offset = <i64>, length = <i64> } }
```

one entry per external parameter, keyed by parameter name. This keeps the
identity/dtype/shape on the `tensor.empty` value and the `{file, offset,
length}` metadata on the module, so no information is silently dropped, while
still avoiding a runtime weight loader.

## Public API

```cpp
namespace sonicboom::sx {
enum class LoweringErrorKind : uint8_t { Type, Operator, Verification };
struct LoweringError { LoweringErrorKind kind; std::string message; };
std::expected<std::string, LoweringError> lower_to_mlir(const Document& doc);
}
```

- `lower_to_mlir` internally creates an `MLIRContext`, loads the four dialects,
  lowers the graph into a `func.func`, runs full MLIR verification, and prints
  the module.
- The header is **MLIR-free**: `lowering.h` includes only `ir.h`, so no MLIR
  type leaks through this API to the C API, Guile, or any other consumer.
- Failure kinds: `Type` (a dtype/shape could not be mapped), `Operator` (an
  operator is unsupported or could not be lowered), `Verification` (the
  produced module failed MLIR verification).

## Error handling

All failures are explicit strings routed through `std::expected`. Any operator
that cannot yet be lowered cleanly fails explicitly rather than inventing
semantics (e.g. `conv group != 1`, `max_pool ceil_mode != 0`, unknown op name,
missing `reduce_mean` axes). Malformed attributes and unsupported shapes are
also validated up front and rejected with a structured error before any MLIR
builder API is reached: `strides`/`dilations` must be exactly 2 entries, `pads`
4, `kernel_shape` 2, `gemm` A/B must be rank-2, and `reduce_mean` axes must be
distinct and in range. MLIR verification failures are captured as
`Verification` errors with the diagnostic text.

## Known limitations

- Integer signedness is not materialized (signless MLIR integers).
- Static shapes only; no dynamic or symbolic dimensions.
- `conv`: `group == 1`, `dilations == 1` only.
- `max_pool`: `dilations == 1`, `ceil_mode == 0` only.
- `gemm`: rank-2 A/B only (no batched matmul in v0).
- `reduce_mean`: requires an explicit `:values` int64 `axes` parameter; duplicate
  axes are rejected.
- `:external` weights are placeholders (`tensor.empty`); values are not loaded
  (the execution path bakes them as dense constants — see
  `cpu-execution-mvp.md`).
- The pooling kernel operand is a dummy — correct for max (its values are
  unused), but not a general pooling representation.

## Tests

- `tests/core/test_s_expr_lowering.cpp` — 16 focused cases (type mapping, input
  argument, constant, add, relu, reshape, reduce_mean, conv, max_pool, gemm,
  return wiring, unsupported-operator error, plus four validation cases:
  empty conv strides, empty conv dilations, gemm rank mismatch, duplicate
  reduce_mean axes); structural checks, not byte-for-byte.
- `tests/core/test_s_expr_resnet18_mlir.cpp` — integration: parse the frozen
  ResNet-18 example → lower → verify → structural checks (op census, input /
  output types, external-data attribute).
