# Native-torch generation (SonicCross)

SonicCross is a standalone Guile/Scheme build-time replacement for PyTorch
`torchgen`. It compiles `native_functions.yaml` into the C++ operator
registration / dispatch / declaration sources that back the `native-torch`
(Layer 1) implementation.

It is a **build-time code generator only**:

- SonicBoom does **not** link against or embed SonicCross or Guile.
- SonicCross is **not** a runtime dependency and **not** the SonicBoom Guile
  binding (`bindings/guile/` is a separate Layer 3 concern).

## Layout

```text
tools/generation/
├── native/native_functions.yaml   # frozen input snapshot (SHA256-pinned)
├── SONICCROSS.lock                # pinned PyTorch/SonicCross revisions + checksum
├── README.md                      # this file
└── GenerateNativeTorch.cmake      # the `generate-native-torch` CMake target

<build>/third_party/native-torch/generated/ATen/   # SonicCross output (build dir, gitignored)
├── Functions.h, NativeFunctions.h, …             # aggregate declaration headers
├── ops/<op>.h, <op>_native.h, …                  # per-operator split headers
├── core/                                         # TensorBody.h, TensorMethods.cpp, …
└── Register{Key}*.cpp, Operators*.cpp            # registration + dispatch sources
```

`tags.yaml` is intentionally absent: SonicCross embeds the frozen valid-tags
constant internally rather than reading a file.

## Generated-source policy

- Generation is **build-time** (matching upstream PyTorch's build-time torchgen):
  `cmake` runs SonicCross during configure and writes the generated C++ into the
  build directory (`<build>/third_party/native-torch/generated/ATen/`), which is
  gitignored and never committed.
- SonicCross is a **required build-time dependency** (like torchgen for PyTorch):
  a `git clone → cmake → build` must have `guild` installed.

## Generation prerequisites (required)

SonicCross must be installed (`./bootstrap && ./configure && make && make
install` in the SonicCross tree, so `guild torchgen` runs from compiled
`.go` modules), or point CMake at an explicit `guild` executable:

```sh
cmake -B build -DSONICCROSS_GUILD=/path/to/guild
```

Configuration fails if `guild` is not found.

## Reproducibility

`SONICCROSS.lock` pins the PyTorch reference commit, the SonicCross release
(now `e3acee5`, which adds `--per-operator-headers`), and the
`native_functions.yaml` SHA256. Regenerating with those exact inputs is
byte-identical to the committed baseline. SonicCross v0 is frozen; do not
regenerate against a different PyTorch revision.

## Deferred (not decided here)

Several torchgen outputs (`core/TensorBody.h`, `core/aten_interned_strings.h`,
`core/enum_tag.h`, `core/ATenOpList.*`) were also hand-staged under
`third_party/native-torch/aten/src/ATen/core/` during migration. Reconciling
those staged copies with the generated `core/` files is a native-torch
migration-manifest decision, deferred until `design/native-torch-migration.md`
is established.
