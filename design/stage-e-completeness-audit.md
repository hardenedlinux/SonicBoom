# Stage E — Native-Torch Full-Inheritance Completeness Audit

Audit date: 2026-10-05. Frozen upstream: PyTorch release/2.14
(`41ffbc4a994e058af9fe00ed5caba73fc1033359`, v2.14.1a0). SonicCross: v0.0.1
(`e3acee5afede1b63074f1aa6f9622c90eb3b0bd4`). This audit supersedes the earlier
§22.E "in progress" wording and records the state of the frozen-version source
after Stage C (copy) + Stage D (CPU build/verify).

## 1. Method

Every claim below is checked against the live tree + build directory, not
memory. Numbers were re-derived on 2026-10-05:

- **Migrated** — source files physically present under `third_party/native-torch/`.
- **Generated** — files emitted by SonicCross into the gitignored build dir
  (`build/third_party/native-torch/generated/`).
- **Compiled** — translation units turned into `.o` inside `libaten_core.a`
  (or `libc10.a`).
- **Registered** — `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` static initializers
  that actually run in a linked binary.
- **Executed-verified** — a real kernel invoked through the real dispatcher and
  numerically asserted.

These five are **distinct**. A file can be migrated but not compiled (CUDA
kernels), compiled but not registered (objects dropped from a static link), or
registered but not executed (every op beyond the 7 in `numeric_proof`).

## 2. State matrix

| Dimension | Migrated | Generated | Compiled | Registered | Executed-verified |
|---|---|---|---|---|---|
| c10 core + util | ✅ 45 + 43 `.cpp` | — | ✅ 89 obj | ✅ (dispatch keys, schema) | ✅ (via dispatcher) |
| ATen core (`ATen/core`) | ✅ 57 `.cpp` | — | ✅ | ✅ | ✅ (dispatch_proof) |
| ATen facade (`ATen/*.cpp` + `detail/` + `cpu/`) | ✅ 51 + 12 + 2 | — | ✅ | ✅ | ✅ (numeric_proof) |
| native CPU kernels (`native/*.cpp`, `native/cpu/`) | ✅ 479 `.cpp` (64 cpu) | — | ✅ 273 obj | ✅ (CPU split registers) | ⚠️ 7 ops only |
| native CUDA kernels (`native/cuda/`) | ✅ 221 `.cu` + 27 `.cpp` | — | ❌ (not wired) | ❌ | ❌ |
| ATen CUDA facade (`ATen/cuda/`) | ✅ 102 files | — | ❌ | ❌ | ❌ |
| cudnn (`ATen/cudnn/`) | ✅ 4 `.cpp` | — | ❌ (cudnn.h absent) | ❌ | ❌ |
| quantized boundary (`native/quantized/`) | ✅ 70 `.cpp` (fbgemm-free subset) | — | ✅ 2 obj | ✅ (stubs) | ⚠️ deferred by design |
| op registration (`Register*.cpp`) | n/a | ✅ 54 files (26 split) | ✅ ~20 split | ✅ (whole-archive only) | ⚠️ 7 ops |
| per-op headers (`ops/*.h`) | n/a | ✅ 7063 | n/a | n/a | n/a |
| build integration (`libsonicboom.so`) | n/a | n/a | ✅ links | ❌ **registrations dropped** | ❌ |

Legend: ✅ done / ⚠️ partial / ❌ gap. Details in §4.

## 3. Concrete counts (re-derived 2026-10-05)

- Total source files in tree (`.h/.hpp/.cpp/.cc/.cu`): **2711**.
- c10: 45 `core/` + 43 `util/` `.cpp` (2 excluded: `signal_handler`, `tempfile`).
- ATen core: 57 `.cpp`; facade 51 + `detail/` 12 + `cpu/` 2; native 479 (64 cpu).
- CUDA: `native/cuda/` 221 `.cu` + 27 `.cpp`; `ATen/cuda/` 102 files; `cudnn/` 4.
- Generated (SonicCross, gitignored): **7063** `ops/*.h`; **69** `.cpp`
  (54 `Register*` — 26 split + 28 `*Everything*` — plus `Operators_0..4`,
  `Functions`, `TensorMethods`, `ATenOpList`, `CompositeViewCopyKernels`,
  `ViewMetaClasses`, `UfuncCPU_add`, `UfuncCPUKernel_add`, and `core/` copies).
- Compiled: `libaten_core.a` **439** objects; `libc10.a` **89** objects.
- Register files **compiled** into `aten_core` (CPU-only split variants):
  `RegisterSchema`, `RegisterBackendSelect`, `RegisterCPU_0..3`,
  `RegisterCompositeExplicitAutograd*`/`NonFunctional`,
  `RegisterCompositeImplicitAutograd*`/`NestedTensor`,
  `RegisterFunctionalization_0..3`, `RegisterMeta_0`, `RegisterNestedTensorCPU_0`,
  `RegisterNestedTensorMeta_0`, `RegisterSparseCPU_0`, `RegisterSparseCsrCPU_0`,
  `RegisterSparseCsrMeta_0`, `RegisterSparseMeta_0`, `RegisterZeroTensor_0`.
  **Not compiled**: `RegisterCUDA`, `RegisterHPU`, `RegisterMkldnnCPU`,
  `RegisterQuantizedCPU/CUDA/Meta`, `RegisterNestedTensorCUDA/HPU`,
  `RegisterSparseCUDA`, `RegisterSparseCsrCUDA`, and every `*Everything*`.

## 4. Confirmed gaps (with resolution paths)

### G-1 — `libsonicboom.so` drops op registration (critical)

`core/CMakeLists.txt:54` links `sonicboom PRIVATE aten_core` **without**
`--whole-archive`. `aten_core` is a static library whose `Register*.cpp`
translation units register kernels via `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`
static initializers; with no referenced symbol, the linker drops those objects.

**Evidence:** `numeric_proof` (linked with `-Wl,--whole-archive aten_core
-Wl,--no-whole-archive`) contains 4 `TorchLibraryInit` symbols +
`_GLOBAL__sub_I_RegisterCPU_0.cpp`; `libsonicboom.so` contains **0**
`TorchLibraryInit` symbols. Consequence: the Layer 1 adapter's
`c10::Dispatcher::singleton().findSchemaOrThrow(...)` throws at runtime, so
`libsonicboom.so` — the published core — cannot dispatch any native op.

**Resolution (executable):** in `core/CMakeLists.txt`, change line 54 to force
the archive in, mirroring `numeric_proof`:

```cmake
target_link_libraries(sonicboom PRIVATE
  "-Wl,--whole-archive" aten_core "-Wl,--no-whole-archive")
```

After the fix, re-verify with `nm -C build/core/libsonicboom.so | grep -c
TorchLibraryInit` (expect > 0) and add a `tests/core` dispatch test (G-3).

### G-2 — CUDA not wired into the build (blocking, but not SDK-blocked)

CUDA source is fully copied (§2 matrix) but no `.cu` is compiled and
`RegisterCUDA*` is excluded. **The blocker changed:** the CUDA SDK is now
present — `nvcc` 12.0 (`cuda_12.0.r12.0/compiler.32267302_0`) + a GeForce
RTX 3050 (8 GB) + `libcudart/cublas/cusparse/cusolver/cufft` — so the prior
"BLOCKED on SDK/toolkit" note in §22.E is **stale**. The real blockers are:

1. Build not wired: no `.cu` glob, no `CUDA` language, no `RegisterCUDA*`.
2. **cuDNN absent** (`cudnn.h` not found) — blocks the `ATen/cudnn/` subset
   only (cudnn convolution descriptors); the rest of CUDA does not need it.
3. MAGMA absent — blocks `native/cuda/` `linalg` (magma) subset only.

**Resolution (executable):** gate CUDA behind `SONICBOOM_USE_CUDA` (already an
option in the root `CMakeLists.txt`). Under it: `enable_language(CUDA)`, add the
`native/cuda/*.cu` + `ATen/cuda/*.cu` globs, include `RegisterCUDA*` /
`RegisterSparseCUDA*` / `RegisterSparseCsrCUDA*` split files, and set
`AT_CUDNN_ENABLED=0` (drop `cudnn/`) until cuDNN is installed. This is a
follow-up task (see §6), not a manifest-scope change.

### G-3 — No libsonicboom.so-level native-dispatch test

`tests/core/` exercises Layer 2 / S-Expr / MLIR but has no test that loads
`libsonicboom.so` and runs a real native op through the adapter. Blocked on G-1
(until the archive is force-linked, the adapter's `findSchemaOrThrow` throws).
Add a `tests/core/test_native_dispatch.cpp` after G-1.

### G-4 — Op coverage is 7 verified vs ~1595 registered

`numeric_proof` asserts only `ones`/`full`/`add`/`mul`/`relu`/`sum`. The
generated registration covers ~1595 op names (§22.C) but the vast majority are
never invoked. Correctness beyond this 7-op smoke set is **unverified**, not
broken. Extend `numeric_proof` (or add an op-set smoke test) over a
representative op matrix (matmul, softmax, conv, reductions, indexing, casts).

### G-5 — BLAS / LAPACK / MKL / oneDNN / FFT guarded out

`ATen/Config.h` sets `AT_BUILD_WITH_BLAS=0`, `AT_BUILD_WITH_LAPACK=0`,
`AT_MKL_ENABLED=0`, `AT_POCKETFFT_ENABLED=0`, `AT_MKLDNN_ENABLED=0`. BLAS/LAPACK
accelerated CPU ops (`addmm`, `bmm`, `matmul`, linear-algebra) and FFT compile
their non-accelerated or stubbed paths (e.g. `mkl/SpectralOps.cpp` registers
`REGISTER_NO_CPU_DISPATCH` + throwing `_fft_*` stubs). Their numeric behavior is
therefore **unverified / intentionally stubbed**, not silently broken. This is
consistent with the deferred scope; record it so it is not mistaken for a
complete backend.

### G-6 — Autograd, quantized compute, SymInt, Generator (deferred by design)

These are deferred (§4.1.5 / CLAUDE.md §4.1.5). They are **not** gaps to fix;
they are documented exclusions. The fallthrough adaptation (G-7) exists so the
dispatcher degrades cleanly in their absence.

## 5. Adaptations recorded (native-torch source owned by SonicBoom)

These are the source-level changes SonicBoom made to make the frozen tree build
and dispatch in v0. They are recorded in the manifest §4.2 MODIFY table as
M6–M10.

| # | File | What it does |
|---|---|---|
| M6 | `aten/src/ATen/record_function.cpp` | No-op profiler hooks + `get/set_record_function_tls_` (upstream impl lives in `torch/csrc/profiler`, outside scope). |
| M7 | `c10/core/Scalar.h` | Inline `operator<<(ostream, Scalar)` (upstream impl is fmt-based `Formatting.cpp`, deferred). |
| M8 | `aten/src/ATen/core/DispatchFallthroughStubs.cpp` (new) | `makeFallthrough()` for `BackendSelect` + `ADInplaceOrView` + all `Autograd*` keys — replaces the excluded `BackendSelectFallbackKernel.cpp` / `VariableFallbackKernel.cpp`. |
| M9 | `aten/src/ATen/native/quantized/QuantizedStubs.cpp` (new) | `REGISTER_NO_CPU_DISPATCH` for 15 quantized stubs + throwing fbgemm prepack stubs + bare `register_linear_params()`. |
| M10 | `aten/src/ATen/Version.cpp` | Drop `caffe2::GetBuildOptions()` reference. |

## 6. Next steps (suggested follow-up tasks)

1. **Fix G-1** (whole-archive link) — prerequisite for everything downstream.
2. **Add G-3** native-dispatch test against `libsonicboom.so`.
3. **Broaden numeric coverage** (G-4) — op-matrix smoke test.
4. **Re-open Task #74 (CUDA)** with corrected blockers: SDK present
   (nvcc 12.0 + RTX 3050); wire `SONICBOOM_USE_CUDA`, defer `cudnn/` until
   cuDNN is installed; leave MAGMA-dependent linalg guarded.
5. **Record G-5** (BLAS/LAPACK/oneDNN/FFT) as an explicit future amendment if
   full float32/float64 matmul acceleration becomes a requirement.
