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

third_party/native-torch/generated/   # SonicCross output root (checked in)
├── TensorBody.h … Register{Key}.cpp …   # headers + registration/dispatch sources
└── core/                                # TensorMethods.cpp, ATenOpList.cpp, …
```

`tags.yaml` is intentionally absent: SonicCross embeds the frozen valid-tags
constant internally rather than reading a file.

## Generated-source policy

- `third_party/native-torch/generated/` is a **checked-in baseline**. A normal
  `git clone → cmake → build` never needs SonicCross installed; it compiles the
  committed generated sources.
- Regeneration is an **explicit developer action**, never part of a normal
  build:

  ```sh
  cmake -B build
  cmake --build build --target generate-native-torch
  ```

  This overwrites the checked-in baseline in place. The regenerated files are
  then committed by hand (the generator does not commit).

## Regeneration prerequisites

SonicCross must be installed (`./bootstrap && ./configure && make && make
install` in the SonicCross tree, so `guild torchgen` runs from compiled
`.go` modules). Alternatively point CMake at a source tree:

```sh
cmake -B build -DSONICCROSS_SOURCE_DIR=/path/to/SonicCross
```

The `generate-native-torch` target only exists when `guild` is found; otherwise
it is omitted and the build falls back to the checked-in sources.

## Reproducibility

`SONICCROSS.lock` pins the PyTorch reference commit, the SonicCross release,
and the `native_functions.yaml` SHA256. Regenerating with those exact inputs is
byte-identical to the committed baseline. SonicCross v0 is frozen; do not
regenerate against a different PyTorch revision.

## Deferred (not decided here)

Several torchgen outputs (`core/TensorBody.h`, `core/aten_interned_strings.h`,
`core/enum_tag.h`, `core/ATenOpList.*`) were also hand-staged under
`third_party/native-torch/aten/src/ATen/core/` during migration. Reconciling
those staged copies with the generated `core/` files is a native-torch
migration-manifest decision, deferred until `tools/native-torch-migration.md`
is established.
