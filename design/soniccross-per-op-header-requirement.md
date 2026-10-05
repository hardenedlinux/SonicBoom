# SonicCross — Per-op Header Generation Requirement

> **Status: SATISFIED** (SonicCross, 2026-10-04). New `--per-operator-headers`
> flag-gated mode in `per-operator-headers.scm` generates the per-op split
> headers + per-op `Register{key}.cpp`, **byte-identical** to the frozen
> torchgen oracle (7082 headers + 47 Register files, all diffs green). The four
> header kinds below are covered, plus `{name}_{dispatch}_dispatch.h`
> (DispatchKeyFunction). Aggregate headers (Functions/NativeFunctions/Meta/etc.)
> degrade to include-only shims, matching upstream. G1–G20 default output is
> unchanged.

> **Independent verification (2026-10-04):** regenerated with the installed
> `guild torchgen --per-operator-headers` (commit `e3acee5`) and cross-checked
> the 237 `native/*.cpp` files' `#include <ATen/ops/*.h>` set against the
> output: 1595/1597 distinct targets generated; the 2 non-generated
> (`from_blob.h`, `tensor.h`) are hand-authored facade headers already staged
> under `aten/src/ATen/ops/` (upstream `git ls-files` confirms they are
> committed, not torchgen output). Full coverage.

## Purpose

SonicCross currently generates the **aggregate** operator declaration headers
(`Functions.h`, `NativeFunctions.h`, `MetaFunctions.h`, `core/TensorBody.h`,
`TensorMethods.cpp`) and the **full registration** (`RegisterCPU_*.cpp`,
`RegisterCUDA_*.cpp`, `Operators_*.cpp`, …). This is sufficient for the
dispatcher core and the registration mechanism.

What is missing for the full native-kernel build is the **per-op split headers**
under `ATen/ops/` that the migrated top-level native dispatcher files
(`aten/src/ATen/native/*.cpp`, 237 files) `#include` directly. SonicBoom cannot
compile those files until these headers exist.

## Input

- `native_functions.yaml` (631 KB, 16238 lines) + `tags.yaml` — frozen baseline
  pytorch release/2.14 @ `41ffbc4a994e058af9fe00ed5caba73fc1033359`
  (v2.14.1-rc1). Located in SonicBoom at
  `third_party/native-torch/aten/src/ATen/native/{native_functions.yaml,tags.yaml}`.

## Required output (measured from the copied `native/*.cpp`)

3024 `#include <ATen/ops/...>` occurrences across the 237 files, in four kinds:

| Header kind | Occurrences | Declares |
|---|---|---|
| `ATen/ops/<op>.h` | 1320 | `at::<op>(...)` |
| `ATen/ops/<op>_native.h` | 1653 | `at::native::<op>(...)` |
| `ATen/ops/<op>_meta.h` | 37 | `at::meta::<op>(...)` |
| `ATen/ops/<op>_ops.h` | 14 | operator object (`at::op()`) |

~1765 distinct op names are referenced (a subset of the full YAML op set —
composite/functional ops are registered in the generated `Register*.cpp`, not in
`native/*.cpp`).

No `_structured.h` / `_decomposed.h` / `_native_Tensor.h` / `_meta_Tensor.h`
headers are referenced by the migrated files.

## Relationship to existing output

The per-op headers are a **per-operator split** of what SonicCross already emits:

- `Functions.h`       ≡ union of `ATen/ops/<op>.h`
- `NativeFunctions.h` ≡ union of `ATen/ops/<op>_native.h`
- `MetaFunctions.h`   ≡ union of `ATen/ops/<op>_meta.h`

So this is a split/partition of already-generated declarations, not a new
codegen domain.

## Semantics requirement

The generated `ops/*.h` must match upstream `torchgen`'s output for these
headers **semantically** — same function signatures, same namespace layout
(`at::`, `at::native::`, `at::meta::`), and (for structured ops) the same
`structured_*` / `*_meta` struct layout — so that the migrated `native/*.cpp`
files compile **unmodified** (SonicBoom must not rewrite those files'
`#include` lines).

Reference: upstream `torchgen/gen.py` / `torchgen/api/` (the `ops` destination,
`RegisterDispatchKey` and `structured` dispatch paths).

## Non-goal

SonicCross must NOT generate: `Declarations.yaml`, `VmapGeneratedPlumbing.h`,
AOTI `c_shim_*.cpp` (already excluded per the SonicCross README).
