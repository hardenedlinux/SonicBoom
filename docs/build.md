# Building SonicBoom

SonicBoom builds out of tree with CMake. The published artifact is
`libsonicboom.so`: native-torch and MLIR are statically linked into it, and
nothing else is needed at run time.

## Requirements

- Linux x86-64, g++-13, CMake, Ninja (optional), libyaml-dev. The CPU
  quantized-matmul kernels are AVX2 (with a scalar fallback); an aarch64 build
  is not wired up yet — the AVX2 translation unit has no architecture guard, so
  it does not compile there as it stands. The port is a guard plus a NEON
  sibling of the same fixed-point kernel.
- CUDA toolkit 12.0 or newer and an NVIDIA GPU, compute capability 6.1 or
  newer (`__dp4a`) — 7.5 or newer for the tensor-core MMQ path. The release is
  built and tested for **sm_86** only (RTX 30-series).
- GNU Guile 3.0 and
  [guile-json](https://github.com/aconchillo/guile-json) (the `(json)` module,
  used by the Guile binding for build info and by the serve front end).
  `apt install guile-3.0 guile-json` on Debian/Ubuntu.
- [SonicCross](https://github.com/hardenedlinux/SonicCross), a required
  build-time dependency (see below)
- LLVM/MLIR 23, either built by the in-tree script or installed elsewhere

## Build

SonicBoom's operator library, native-torch, is generated at build time from
PyTorch's `native_functions.yaml`, as PyTorch itself does with torchgen.
SonicBoom uses **SonicCross**, a torchgen written in Guile, which turns the YAML
into native-torch's C++ declaration, registration and dispatch sources (more
than 7,000 per-operator headers). SonicCross runs only at build time and is
not linked into `libsonicboom.so`. Install SonicCross **v0.0.2** first, the
version pinned in [`tools/generation/SONICCROSS.lock`](../tools/generation/SONICCROSS.lock):

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

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONICBOOM_USE_CUDA=ON
cmake --build build
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

### Build information

Whatever was configured is compiled into the library and can be read back —
from C with `sb_build_info()` ([`capi/include/sonicboom/capi.h`](../capi/include/sonicboom/capi.h)),
from Guile with `(build-info)` ([`bindings/guile/README.md`](../bindings/guile/README.md)),
and from the command line with
[`tools/sb-report.sh`](../tools/sb-report.sh), which prints it beside the host,
the GPU and the toolchain:

```sh
tools/sb-report.sh --model your-model.gguf --no-bench --no-golden --json
```

## Run (Guile)

Download a model and verify it:

```sh
mkdir -p models
curl -L -o models/gemma-4-E4B-it-Q3_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-E4B-it-Q3_K_M.gguf
echo "0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4  models/gemma-4-E4B-it-Q3_K_M.gguf" | sha256sum -c
```

Any GGUF the runtime supports runs the same way — for instance the other
benchmarked files (`qwen2.5-0.5b-instruct-q4_k_m.gguf`,
`DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` and
`qwen2.5-7b-instruct-1m-q4_k_m.gguf`); every file and checksum is in
[Scope](scope.md).

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
[`capi/include/sonicboom/capi.h`](../capi/include/sonicboom/capi.h) (`sb_llm_*`).
To serve an OpenAI-compatible API, see
[Serve an OpenAI-compatible API](serve-openai.md).

## Test

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

## Benchmark

```sh
LLAMA_BIN=/path/to/llama.cpp/build/bin tools/bench.sh your-model.gguf
LLAMA_BIN=/path/to/llama.cpp/build/bin tools/bench.sh your-model.gguf --long
LLAMA_BENCH=/path/to/llama-bench tools/sb-report.sh --gemma gemma.gguf --qwen qwen.gguf
LLAMA_BENCH=/path/to/llama-bench tools/sb-report.sh --model your-model.gguf
```

`bench.sh` runs the canonical bench (3 process runs) for any GGUF the runtime
loads; `--long` adds the 4K–32K decode and prefill tiers. `sb-report.sh` runs
the canonical bench and prints the hardware and the build behind the numbers;
`--gemma` / `--qwen` / `--deepseek` also run the golden checks, while `--model`
is for a benchmark-only file with no golden tag.

Every metric is the median of 3 process runs and both runtimes are measured in
the same session; `tools/bench.sh` is the only benchmark SonicBoom publishes
numbers from. The protocol is described in [Benchmark](benchmark.md).

## Warm-up tables and canonical numerics

Kernel and tuner choices are measured on the GPU when a model loads and kept in
a per-device cache: `$SONICBOOM_TUNE_DIR`, else
`$XDG_CACHE_HOME/sonicboom/tune`. A run can carry that table to another machine
with the same GPU:

```sh
sb-tune export --out FILE
sb-tune import --in FILE [--replace]
```

An imported table is refused when the library's build fingerprint differs, so
a stale export cannot be applied to a different build. The table holds every
record family the cache writers produce: each tuner `<op>.json` family, the
device and hosted reduction verdicts, and the JIT candidate cache
(`autotune/sm_<NN>.bin`); append-only logs and lock files are not packed.

`SONICBOOM_TUNE=canonical` fixes every tuner whose pick changes arithmetic to
its documented reference default and makes it read and write no such tune
record, so the same model and prompt produce the same tokens on an empty and a
populated tune directory. Numerics-neutral picks — the MMQ tile configuration
and grouping, the verify GEMV work partition, the speculative length — are
still measured; canonical mode re-measures them rather than reading them from
an imported table. Use canonical mode when comparing token streams across
machines, and the default mode with a full imported table when a fixed plan is
what has to be compared. The table of canonical values is in
[Decision registry](decision-registry.md#canonical-numerics).
