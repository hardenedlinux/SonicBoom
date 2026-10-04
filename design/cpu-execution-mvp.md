# SonicBoom End-to-End CPU Execution MVP

This document records the design and results of the **End-to-End CPU Execution
MVP**: a complete, working CPU execution path from the frozen S-Expr v0.1
format through MLIR compilation and JIT execution to correct ResNet-18
inference, plus the stable C ABI that publishes it.

```text
S-Expr v0.1 → Parser/Validator → SonicBoom IR → MLIR lowering
  → MLIR optimization / LLVM lowering → CPU JIT
  → SonicBoom Runtime (Executable, Buffer, Weight Loader, CPU execution)
  → Correct ResNet-18 inference
```

This extends the earlier `s-expr-to-mlir-lowering.md` (which stops at a verified
MLIR module) all the way down to executed, numerically-verified native code.

## Files

| File | Role |
|---|---|
| `core/include/sonicboom/sx/ir.h` | S-Expr IR data model (`DType`, `TensorType`, `Graph`, `Document`, …) |
| `core/src/sx/parser.cpp` | parser + validator (`parse_document`) |
| `core/include/sonicboom/sx/lowering.h` | MLIR-free lowering facade (`lower_to_mlir`) |
| `core/src/sx/lowering.cpp` | S-Expr → MLIR lowering (operator-level) |
| `core/src/sx/lowering_internal.h` | `lower_into_module` + `WeightMap` (internal) |
| `core/include/sonicboom/sx/exec.h` | MLIR-free execution facade (`Executable`, `load_external_weights`) |
| `core/src/sx/exec.cpp` | MLIR pass pipeline + LLVM JIT (`Executable::compile`, `run`) |
| `core/src/sx/runner_utils.cpp` | JIT runtime support symbol `memrefCopy` |
| `capi/include/sonicboom/capi.h` | stable C ABI (Layer 3) |
| `capi/src/capi.cpp` | C ABI implementation (facade over `sx::`) |
| `tests/core/test_s_expr_exec.cpp` | Stage A: Add+Relu through the real JIT |
| `tests/core/test_s_expr_resnet18_exec.cpp` | Stage C: ResNet-18 vs ONNX reference |
| `tests/core/test_capi.c` | Stage D: C ABI exercised from real C |
| `tools/reference/gen_resnet18_reference.py` | ONNX `ReferenceEvaluator` oracle |

## Stage A — CPU JIT

`Executable::compile` lowers a validated `Document` into an MLIR module, runs a
CPU lowering pipeline, and JITs it with MLIR's `ExecutionEngine` into a callable
native function. The pipeline is:

```text
convert-elementwise-to-linalg
→ one-shot-bufferize (function boundaries, identity layout, copy-before-write)
→ convert-bufferization-to-memref
→ convert-linalg-to-loops
→ convert-scf-to-cf
→ expand-strided-metadata
→ lower-affine
→ convert-arith-to-llvm
→ convert-scf-to-cf (residual)
→ finalize-memref-to-llvm
→ convert-func-to-llvm (bare-ptr call conv)
→ convert-cf-to-llvm
→ reconcile-unrealized-casts
```

The function is invoked through the **bare-pointer calling convention**: the
single `func.func @name(tensor<…xf32>) -> tensor<…xf32>` becomes
`float *name(float *)`; the returned pointer is malloc'd by the JIT runtime and
freed by the caller. The entry symbol is named after the graph (not `main`).

Three bufferization options are load-bearing and were tuned empirically:

- `functionBoundaryTypeConversion = IdentityLayoutMap` keeps the signature a
  plain `memref<…xf32>` (no dynamic offset/stride), which is what bare-ptr
  needs to produce `float*(float*)`.
- `copyBeforeWrite = true` forces out-of-place bufferization so the input
  buffer is never written and the result is a fresh buffer the caller can free.
- `bufferAlignment = 1` keeps `memref.alloc`'s allocated pointer identical to
  its aligned (data) pointer; a larger alignment would return the raw malloc
  base while the data sits at the aligned offset.

Bufferization external models (arith/tensor/linalg/func) are registered via
`DialectRegistry` + `registerBufferizableOpInterfaceExternalModels`; without
them one-shot-bufferize cannot bufferize any op ("op was not bufferized").

### `memrefCopy` runtime symbol

`tensor.pad` (conv/max-pool padding) bufferizes to a `memref.copy` on
non-contiguous memrefs, which lowers to a call to the `memrefCopy` runtime
helper. MLIR ships that helper only as a standalone shared object when the LLVM
build has PIC enabled; the SonicBoom component build does not, so SonicBoom
provides it directly in `core/src/sx/runner_utils.cpp`. The descriptor types
come from MLIR's `CRunnerUtils.h`, so the ABI matches exactly; the symbol is
compiled into `libsonicboom.so` with `extern "C"` linkage and resolved by the
`ExecutionEngine`'s `DynamicLibrarySearchGenerator::GetForCurrentProcess`.

## Stage B — runtime + weight loading

`load_external_weights(doc, base_dir)` resolves every `:external` parameter into
raw little-endian bytes, reading `(file, offset, length)` from sidecars (a
relative `file` is resolved against `base_dir`) and validating each buffer's
byte count against the parameter's declared shape and dtype. `Executable`
then bakes those buffers as dense constants during lowering.

`Executable::run` validates the input byte count, invokes the entry point, and
copies the returned malloc'd buffer into the caller's output. The v0 execution
scope is **exactly one float32 input and one float32 output**; anything else is
an explicit compile-time error.

## Stage C — ResNet-18 and the reference oracle

The frozen fixture (`design/s-expr-v0-resnet18.example.sx`) has 49 nodes
(20 conv, 17 relu, 8 add, 1 max_pool, 1 reduce_mean, 1 reshape, 1 gemm), 42
float32 weights in `resnet18.weights.bin`, and 2 inlined int64 constants.

The independent reference is `tools/reference/gen_resnet18_reference.py`, which
runs `onnx.reference.ReferenceEvaluator` over the same ONNX model with input
`np.random.RandomState(1234).randn(1,3,224,224).astype(np.float32)` and writes
the `[1,1000]` logits to `resnet18.reference.bin`. It self-checks that the
weight layout matches the fixture's offsets/lengths.

**Results** (float32 logits, 1000 elements):

| Metric | Value |
|---|---|
| max absolute error | `3.34e-6` |
| max relative error | `1.15e-4` |
| argmax (got vs ref) | `107` = `107` ✓ |

The error is pure float32 accumulation-order noise across ~50 operators.

## Stage D — C ABI

The stable C ABI (`capi/include/sonicboom/capi.h`) exposes the pipeline as
opaque-handle + explicit-status calls, with no C++/MLIR/LLVM/ATen/c10 leakage:

```text
sb_parse → sb_compile → sb_input_info/sb_output_info (bind)
        → sb_bind_input → sb_execute → sb_retrieve_output
        → sb_document_free / sb_executable_free (release)
```

plus `sb_error_message` / `sb_error_get_kind` / `sb_error_free`. The header is
compiled by a C compiler (proven by `tests/core/test_capi.c`). The
implementation lives behind `capi/src/capi.cpp`, compiled as an object library
and folded into `libsonicboom.so`, so all `sb_*` symbols (and `memrefCopy`) are
exported from the single published core library.

## Accumulator initialization (correctness fix)

`linalg.conv_2d_nchw_fchw`, `linalg.pooling_nchw_max`, `linalg.matmul`, and the
`reduce_mean` sum reduction all **accumulate into their `outs` operand** (e.g.
conv is `O += I ⊛ K`, pooling is `O = max(O, I)`). Using a bare `tensor.empty`
as that operand folds uninitialized memory — typically NaN — into every result.
The lowering therefore initializes each accumulator to its identity with
`linalg.fill`: `0.0` for conv/matmul/sum, `-∞` for max pooling.

## Build and dependencies

- `libsonicboom.so` is the final published core library (SHARED); native-torch
  and MLIR/LLVM are statically linked into it (PIC via
  `CMAKE_POSITION_INDEPENDENT_CODE ON`).
- `ldd build/core/libsonicboom.so` shows **no libMLIR/libLLVM** shared
  dependency — only libstdc++, libm, libgcc_s, libc, and the dynamic linker.
- MLIR/LLVM are pinned at `llvmorg-23.1.2` (`third_party/llvm-project`,
  unmodified), built once by `tools/mlir/build-mlir.sh` into a gitignored
  prefix.

## Tests

All 13 executables pass: `test_schema`, `test_value`, `test_operator`,
`test_backend`, `test_tensor`, `test_mlir`, `test_s_expr`,
`test_s_expr_lowering`, `test_s_expr_exec`, `test_s_expr_resnet18`,
`test_s_expr_resnet18_mlir`, `test_s_expr_resnet18_exec`, `test_capi`.

## Limitations

- v0 execution scope is exactly one float32 input and one float32 output.
- `conv`: `group == 1`, `dilations == 1` only; `max_pool`: `ceil_mode == 0`,
  `dilations == 1` only.
- Static shapes only; integer signedness is not materialized.
- No autograd, training, SymInt, quantized dtypes, or code generation (all
  deferred by the v0 scope).
- The pooling kernel operand is a dummy shape carrier (correct for max).
- No GPU execution, dynamic roofline, device placement, or adaptive scheduling.
