# ONNX → S-Expr v0.1 Importer (Milestone 2)

This document records the design of the **ONNX-to-S-Expr importer**: an offline
tool that reads an ONNX model and emits (a) a SonicBoom S-Expr v0.1 document and
(b) a flat little-endian external-weight sidecar, both consumable by the
existing frozen `sx::` pipeline (parse → compile → JIT → execute). The
end-to-end result is verified against an independent ONNX reference.

```text
ONNX model (models/resnet18.onnx)
        ↓ tools/onnx2sx (offline, Python tool — not a runtime dependency)
S-Expr v0.1 document + weight sidecar
        ↓ existing parser/validator → lowering → MLIR → JIT (unchanged)
Correct ResNet-18 inference
```

## Position in the pipeline

S-Expr v0.1 remains the canonical interchange format. The importer is a
**front-end to the frozen format**, not a parallel path:

- Its output is ordinary S-Expr v0.1 text — it does **not** bypass the parser or
  the validator, and it introduces **no new syntax**.
- The generated document is parsed/validated/lowered by exactly the same code
  that consumes the hand-authored fixture (`design/s-expr-v0-resnet18.example.sx`).
- Nothing in this tool is linked into `libsonicboom.so`; the C ABI and Layer 2
  interfaces are unchanged, and no ONNX/protobuf/MLIR/LLVM/ATen/c10 type appears
  in them.

### Why a Python tool

The importer is a **Python offline development tool** under `tools/onnx2sx/`.
It reuses the `onnx` + `numpy` already present in the `models/.venv` toolchain
for reading ONNX protobufs, and is deliberately kept separate from the runtime:

- Python is **not** a runtime dependency of SonicBoom v0. Imported models are
  executed entirely by `libsonicboom.so`; the only artifacts that cross the
  boundary are the `.sx` text and the `.weights.bin` sidecar.
- The importer is run once at authoring time (ahead-of-time conversion), never
  at inference time.

This is the smallest practical implementation: no new runtime library, no new
C++ dependency, no coupling between the ONNX reader and the core.

## Files

| File | Role |
|---|---|
| `tools/onnx2sx/importer.py` | conversion + CLI (`convert`, `write_outputs`, `main`) |
| `tools/onnx2sx/__init__.py` / `__main__.py` | `python -m onnx2sx` entry point |
| `tools/onnx2sx/test_importer.py` | 16 unit tests over programmatic ONNX fixtures |
| `tests/core/test_onnx_import_resnet18_exec.cpp` | end-to-end: imported ResNet-18 vs reference |
| `design/onnx-to-s-expr-importer.md` | this document |

## Supported subset

Exactly the seven ResNet-18 operators, default ONNX domain (`""`/`ai.onnx`):

| ONNX op | S-Expr symbol |
|---|---|
| `Conv` | `conv` |
| `Relu` | `relu` |
| `Add` | `add` |
| `MaxPool` | `max_pool` |
| `ReduceMean` | `reduce_mean` |
| `Reshape` | `reshape` |
| `Gemm` | `gemm` |

Anything else — operator, opset domain, data type, or dynamic shape — is
rejected with an actionable `onnx2sx: error: …` diagnostic and a nonzero exit.

## Emitted document shape

The output matches the frozen spec's grammar exactly:

```scheme
(sonicboom-s-expr
  (version 0 1)
  (graph
    (name "main")
    (opset "default" 20)
    (inputs …)
    (outputs …)
    (parameters …)
    (nodes …)))
```

- **SSA + topological order** are preserved: nodes are emitted in graph order,
  each value defined exactly once before use (enforced by the importer and
  re-enforced by the validator).
- **Names** are carried through verbatim as quoted string literals.
- **Attributes** are translated into typed forms and emitted **sorted by name**
  for a stable canonical document: `(attrs (auto_pad (string "NOTSET")) …)`.
- **Node outputs** carry a fully inferred `(tensor <dtype> (shape …))` type.

### Initializer handling

Initializers are classified by element type and by overlap with graph inputs:

| Initializer | Emission |
|---|---|
| float (`float32/16`, `bfloat16`, `float64`) | external sidecar `(data :external <file> <offset> <len>)`, appended in ONNX initializer order, little-endian |
| integer (`int8/16/32/64`, `uint8`) | inlined `(data :values …)` (reshape shapes, reduction axes) |
| also a graph input | graph input (v0 has no input-default mechanism; the default value is dropped) |

The sidecar's byte ordering is `np.ascontiguousarray(arr).tobytes()`
(little-endian on all supported hosts); offsets/lengths are recomputed from the
accumulated blob so they are always self-consistent. The importer's ResNet-18
sidecar is byte-identical to the reference generator's
`design/resnet18.weights.bin`, confirming exact ordering.

## Shape inference

Shapes are computed **authoritatively** from inputs/parameters through each
operator — never trusted from ONNX `value_info`. This makes the importer robust
to models whose `value_info` is missing or stale (the unit-test fixtures omit it
entirely), and it guarantees the emitted `(shape …)` types are internally
consistent with the lowering that consumes them.

- **Conv / MaxPool** — ONNX floor rule:
  `out = floor((in + pad_begin + pad_end − dilation·(kernel−1) − 1) / stride) + 1`.
- **Add** — `numpy.broadcast_shapes`.
- **ReduceMean** — resolves int64 axes (negative → rank-normalized), applies
  `keepdims`.
- **Reshape** — resolves `0`/`-1` with `allowzero`, checks element count.
- **Gemm** — `transA`/`transB`, rank-2 A/B, bias shape check.

The same inference is used both to emit the output type and (for
reduce/reshape) to resolve constant-input values at import time, matching the
lowering's expectations (`reduce_mean` reads axes from a `:values` param;
`reshape` uses the output type's static shape).

## Output writing (failure-safe)

`write_outputs` writes both files to temporary names in their final directories
and `os.replace`s them into place only after both succeed:

- On any failure, the temporaries are removed and **no partial artifact** is
  left behind (a renamed sidecar is rolled back if the S-Expr rename fails).
- Files are `chmod 0o644` so the runtime/test can read them.
- Write failures surface as `onnx2sx: error: write failed: …` with exit 1.

## CLI

```text
models/.venv/bin/python -m onnx2sx <input.onnx>
    -o/--output      output S-Expr file      (default: <input-stem>.sx)
    --weights-dir    sidecar directory       (default: dirname of --output)
    --weights-file   sidecar filename        (default: <output-stem>.weights.bin)
```

Nonzero exit on any load/convert/write failure, with a single actionable line.

## Rejection table (unit tests)

| Construct | Behavior |
|---|---|
| unsupported operator (`Gelu`, …) | `unsupported operator 'X'` |
| unsupported data type (`UINT16`, …) | `unsupported ONNX data type N` |
| dynamic dimension | `dynamic dimension` |
| missing `kernel_shape` / malformed `pads` | attribute-name error |
| invalid tensor data (short raw buffer) | `invalid tensor data` |
| duplicate value definition | `duplicate definition of value 'y'` |
| output references unknown value | `unknown value 'missing'` |
| custom opset domain | `custom opset domain 'X' is not supported` |
| multi-output node | `must produce exactly one output` |
| initializer also graph input | emitted as graph input (not a parameter) |

## End-to-end verification

`tests/core/test_onnx_import_resnet18_exec.cpp` re-runs the full ResNet-18
pipeline over **generated** artifacts (not the hand-authored fixture): it parses
`build/import/resnet18.sx`, loads `build/import/resnet18.weights.bin`, feeds the
deterministic seed-1234 input, and compares the `[1,1000]` logits against the
independent `onnx.reference.ReferenceEvaluator` oracle.

| Metric | Value |
|---|---|
| max absolute error | `3.34e-6` |
| max relative error | `1.15e-4` |
| argmax (got vs ref) | `107` = `107` ✓ |

The comparison is **semantic** (numeric agreement), not textual identity; the
generated document is a fresh artifact, never the hand-authored fixture.

## Boundaries preserved

- S-Expr v0.1 spec, C ABI surface, `third_party/llvm-project`, and
  `third_party/native-torch` are unmodified.
- The operator set is not expanded; SonicCross remains the torchgen replacement
  (ONNX import does not move into it).
- No ONNX/protobuf type crosses the C ABI or Layer 2 boundary.
- `libsonicboom.so` contains **no Python** — the importer is a separate offline
  tool, run ahead of time.

## Limitations

- Default ONNX domain only; `conv` group≠1 / dilations≠1, `max_pool`
  ceil_mode≠0 / dilations≠1, rank-2 `gemm` only (mirroring the lowering).
- Static shapes only; scalar/`bool` graphs are out of v0 execution scope.
- An initializer that is also a graph input loses its default value (v0 has no
  input-default mechanism).
- The importer itself is Python-only and not distributed inside
  `libsonicboom.so`.
