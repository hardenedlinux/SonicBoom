# SonicBoom

SonicBoom is a **native tensor runtime** — a generic interpreter (v0) that
executes neural-network graphs through native C++ backends, with a first-class
**Guile** binding.

## Architecture

Four layers with strictly downward dependencies:

```
Layer 3    C ABI / Guile / other bindings
Layer 2    SonicBoom stable runtime interface (nt::)
Layer 1    native-torch implementation + adapter
Layer 0    CPU / CUDA
```

`libsonicboom.so` is the published core library (built SHARED); native-torch and
MLIR are statically linked into it (position-independent code). The stable
binary-compatibility boundary is the C ABI (`capi/`); the C++ Layer 2 interface
is source-level only.

## Guile AI framework

The `(sonicboom …)` modules provide a minimal PyTorch/Keras-like model API:

1. **Define / compose** — `(sonicboom nn)`: `linear`, `relu`, `sequential`.
2. **Train** — `(sonicboom model)`: `make-model` + `model-train!` drive
   native-torch reverse-mode autograd (`linear`/`relu`/`mse`/SGD) through the
   C ABI; parameters live as native tensors, not Scheme copies.
3. **Export** — `model-export!` writes a self-contained `model.sx` +
   `weights.bin` artifact via `sb_export_model`.
4. **Independent inference** — standalone C++ `sonicboom::Model::load` runs the
   S-Expr → MLIR path with no Guile or Python.

Training (native-torch autograd) and deployment (S-Expr → MLIR) are two engines
that agree on structure, parameter naming, shapes, and math semantics. See
`design/guile-ai-framework-progress.md` for the step-by-step record.

## Build

C++23 with g++-13. MLIR/LLVM are pinned and installed to `build/llvm-mlir-install`
via `tools/mlir/build-mlir.sh`.

```sh
cmake -S . -B build
cmake --build build
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

runs the full suite — 40 tests (36 C++ suites + 4 Guile suites) — in one pass.
