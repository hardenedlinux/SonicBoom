# Getting started

This page builds SonicBoom, then runs a small model forward pass from Guile.
The result of the example is shown at the end, so it is easy to check that the
build and the load path are right.

## Build

SonicBoom needs g++-13, CMake, Ninja, GNU Guile 3.0 and an LLVM/MLIR install;
[Building SonicBoom](build.html) lists the requirements and the ways to supply
MLIR. For a functional build without a GPU:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSONICBOOM_USE_CUDA=OFF
cmake --build build
```

The artifact is `build/core/libsonicboom.so`.

## Start Guile with the binding on the load path

The Scheme sources are in `bindings/guile/scheme`; the compiled library is
loaded by the binding through the C ABI, so its directory goes on
`LD_LIBRARY_PATH`:

```sh
LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme
```

## A two-layer network

`(sonicboom tensor)` supplies the tensor and parameter types, and
`(sonicboom nn)` supplies layers, a model and its forward pass. The example
below builds a linear layer with two inputs and four outputs, applies `relu`,
and maps the four values to one output; every weight is one and there is no
bias, so the result is the sum of the input, squared:

```scheme
(use-modules (sonicboom tensor)
             (sonicboom nn))

(define net
  (sequential (linear 2 4 #:bias #f #:init 'ones)
              (relu)
              (linear 4 1 #:bias #f #:init 'ones)))

(define x (make-tensor '(1.0 2.0) '(1 2)))
(tensor->list (model-forward net x))
=> (12.0)
```

`model-forward` compiles the layer structure into the runtime's graph format
and executes it; the arithmetic itself happens in the native library, not in
Scheme.

## Run a model

For inference from a GGUF file, `(sonicboom inference)` loads a model and a
session and generates text. The model files, their checksums and the commands
are in [Building SonicBoom](build.html#run-guile).
