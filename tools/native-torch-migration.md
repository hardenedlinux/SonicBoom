# SonicBoom — native-torch Migration Manifest

**Status:** AUTHORITATIVE — this document is the source of truth for the migration.

**Phase:** Source discovery + minimal staging (v0). No architecture changes are
proposed here; the SonicBoom architecture and directory layout are fixed and
were decided before this phase.

**Rule for future coding agents:** do **not** rediscover the migration scope.
Do **not** re-search the native-torch tree, re-analyze PyTorch architecture, or
recursively re-inspect dependencies. If compilation fails on an unlisted
native-torch dependency, STOP and report (missing file / missing symbol /
required dependency / where it was referenced). The human amends this manifest;
agents do not expand it.

**Source reference:** `~/Project/pytorch` (read-only reference; not the project
being implemented). Native code is staged under `third_party/native-torch/`
(the term `native-torch` is fixed project terminology — never `pytorch`).

**Pinned source version:** `2.14.1` (`release/2.14`, tag `v2.14.1-rc1`,
commit `41ffbc4a994`, `version.txt` = `2.14.1a0`). The staged tree was
re-synced from this pinned version; re-sync whenever the lock is moved and
update this line.

---

## 1. What this manifest covers

The v0 SonicBoom runtime needs the following native-torch machinery:

| SonicBoom concept | native-torch provider |
|---|---|
| Tensor, Scalar, DType, Device, Layout, MemoryFormat, Backend, Allocator | `c10/core` + `aten/src/ATen/core` |
| Value, ArgumentList, ResultList (boxed invocation) | `c10::IValue`, `c10::Stack` |
| OperatorHandle, OperatorSchema, dispatch() | `c10::OperatorHandle`, `c10::FunctionSchema`, `c10::Dispatcher` |

All of the above is the **Layer 1** implementation. SonicBoom Layer 2
(`core/include/sonicboom/`) must remain independent of `ATen`/`c10`/`torch`
headers; it talks to Layer 1 only through the adapters in `core/layer1/`.

---

## A. Source tree map

Each entry: `~/Project/pytorch/<path>` → `SonicBoom/<destination>` → category → reason.

### A.1 COPY — staged now

| PyTorch path | SonicBoom destination | Category | Reason |
|---|---|---|---|
| `c10/core/` | `third_party/native-torch/c10/core/` | COPY | c10 core types: `Scalar`, `ScalarType`, `Device`, `DeviceType`, `Layout`, `MemoryFormat`, `Backend`, `Allocator`, `TensorImpl`, `StorageImpl`, `DispatchKey(Set)`, `Stream`, `GeneratorImpl`, `TensorOptions`, `SymInt` (+ `core/impl/` guards/TLS/SizesAndStrides). |
| `c10/util/` | `third_party/native-torch/c10/util/` | COPY | utilities: `ArrayRef`, `SmallVector`, `intrusive_ptr`, `TypeMeta`, `int128`, `string_view`, `Exception`, `Logging`, hash maps, dtype scalar types (`Half`, `BFloat16`, `Float8_*`, `qint*`). |
| `c10/macros/` | `third_party/native-torch/c10/macros/` | COPY | `Macros.h`, `Export.h`, `cmake_macros.h` (visibility, C++ std, platform macros). |
| `aten/src/ATen/core/` | `third_party/native-torch/aten/src/ATen/core/` | COPY | `Tensor`/`TensorBase`, `Scalar` (alias), `IValue`, `Stack`, `FunctionSchema`, `OperatorName`, `Dispatcher`, `OperatorHandle` (in `dispatch/Dispatcher.h`), `OperatorEntry`, `KernelFunction` (`boxing/`), `op_registration/` (`library.h`), `Generator`, `List`, `Dict`, `dynamic_type`. |

**Staged contents:** `c10/core`, `c10/util`, `c10/macros`, `aten/src/ATen/core`.
Test files (`*_test.cpp`, `*_test.h`, `test_helpers.h`) and Buck build metadata
(`*.bzl`, `BUILD_MODE.bzl`) were excluded — they are not native implementation.

### A.2 COPY — part of the closure, not yet staged (minimal pass)

| PyTorch path | SonicBoom destination | Category | Reason |
|---|---|---|---|
| `aten/src/ATen/*.h` (non-backend type/runtime headers: `Tensor.h`, `TensorOperators.h`, `Context.h`, `Dispatch.h`, `TensorOptions.h`, `DeviceGuard.h`, `Parallel.h`, `ATen.h`, …) | `third_party/native-torch/aten/src/ATen/` | COPY | `aten/core` headers include these top-level headers; the full `at::Tensor` API is assembled here. Deferred because they transitively require the generated op headers (below). |
| `aten/src/ATen/ops/*.h` | (generated at build) | COPY | Per-op generated headers (`Functions.h`, `NativeFunctions.h`, `ops/*.h`) are produced by `torchgen`, not checked in. |

### A.3 BLACK BOX — required at link time, MUST NOT inspect

| PyTorch path | Category | Reason |
|---|---|---|
| `aten/src/ATen/native/**` | BLACK BOX | Native operator kernels (CPU/CUDA/…). 28 MB, thousands of files. Needed only so boxed dispatch reaches a real kernel; the coding phase must not read these. |
| `aten/src/ATen/{cpu,cuda,mkl,mkldnn,cudnn,miopen,mps,metal,vulkan,quantized,sparse,nested,transformers}/**` | BLACK BOX | Backend-specific kernels and vectorized math. |
| `c10/{cuda,hip,xpu,metal}/**` | BLACK BOX | Backend device guard/stream/allocator impls. |
| `c10/core/impl/{PyInterpreter*,PythonDispatcherTLS*}.cpp/h` + `c10/core/{SafePyObject,PyHandleCache,PyObjectSlot}*` | BLACK BOX | Python-interpreter hooks are part of `TensorImpl`'s compile closure but are inert without a Python interpreter. Staged (must compile) but must not be adapted or studied. |

### A.4 EXCLUDE — outside v0

| PyTorch path | Category | Reason |
|---|---|---|
| `torch/**` (incl. `torch/csrc/`, Python bindings) | EXCLUDE | Python frontend/bindings. |
| `functorch/**`, `caffe2/**`, `torchgen/**` | EXCLUDE | vmap frontend / legacy / codegen tool. (`torchgen` is a **build-time** tool dependency, not migrated source — see §C.) |
| `android/`, `benchmarks/`, `test/`, `docs/`, `.ci/`, `.github/` | EXCLUDE | Not native runtime implementation. |
| autograd, distributed training, quantization, `torch/csrc/autograd` | EXCLUDE | Deferred subsystems. |

---

## B. Layer 1 adapter map

SonicBoom abstraction → native-torch type → adapter source.

| SonicBoom abstraction | native-torch type | adapter source (`core/layer1/adapter/`) |
|---|---|---|
| `Tensor` | `at::Tensor` (`aten/core/Tensor.h`) | `tensor.cpp` |
| `Scalar` | `c10::Scalar` (`c10/core/Scalar.h`) | `scalar.cpp` |
| `DType` | `c10::ScalarType` (`c10/core/ScalarType.h`) | `dtype.cpp` |
| `Device` | `c10::Device` (`c10/core/Device.h`) | `device.cpp` |
| `Layout` | `c10::Layout` (`c10/core/Layout.h`) | `layout.cpp` |
| `MemoryFormat` | `c10::MemoryFormat` (`c10/core/MemoryFormat.h`) | `memory_format.cpp` |
| `Backend` | `c10::Backend` (`c10/core/Backend.h`) | `backend.cpp` |
| `Allocator` | `c10::Allocator` (`c10/core/Allocator.h`) | `allocator.cpp` |
| `Value` | `c10::IValue` (`aten/core/ivalue.h`) | `value.cpp` |
| `ArgumentList` | `c10::Stack` (`aten/core/stack.h`) | `argument_list.cpp` |
| `ResultList` | `c10::Stack` (`aten/core/stack.h`) | `result_list.cpp` |
| `OperatorHandle` | `c10::OperatorHandle` (`aten/core/dispatch/Dispatcher.h`) | `operator_handle.cpp` |
| `OperatorSchema` | `c10::FunctionSchema` (`aten/core/function_schema.h`) | `operator_schema.cpp` |
| `dispatch()` | `c10::Dispatcher::callBoxed()` (`aten/core/dispatch/Dispatcher.h`) | `dispatch.cpp` |

The 14 Layer 2 abstractions above are **Category C — REIMPLEMENT**: SonicBoom
provides its own opaque representations; it does **not** re-export the native
types. The native types stay below the Layer 1 boundary.

Registration of operators into the dispatcher during initialization lives in
`core/layer1/registration/` (e.g. `registration.cpp`), not in the adapter dir.

---

## C. Dependency closure (v0)

Only the closure required by the fixed architecture:

```
SonicBoom Layer 2 (core/include/sonicboom/)      [REIMPLEMENT, owned by SonicBoom]
        │  adapter functions only
        ▼
SonicBoom Layer 1 adapters (core/layer1/)         [ADAPTER]
        │
        ▼
native-torch core (third_party/native-torch/)     [COPY, staged]
        ├─ c10/core, c10/util, c10/macros
        └─ aten/src/ATen/core
        │
        ▼
native-torch kernels (aten/src/ATen/native/**)    [BLACK BOX, link-time only]
```

External / build-time dependencies (not migrated source):

1. **`torchgen`** + `aten/src/ATen/native/native_functions.yaml` +
   `aten/src/ATen/native/tags.yaml` — generates `Functions.h`,
   `NativeFunctions.h`, `RegistrationDeclarations.h`, `RegisterSchema.cpp`,
   `RegisterCodegenUnboxedKernels.cpp`, and the per-op `ATen/ops/*.h` headers.
   These generated files are required for operator registration and boxed
   dispatch to link. **Unresolved dependency** — recorded here; do not expand.
2. **Native kernels** (`aten/src/ATen/native/**`) — required so
   `callBoxed()` has a kernel to hit. **BLACK BOX**, deferred until a
   compile/link error names a specific symbol.

---

## D. Explicit black-box areas (coding agents MUST NOT inspect)

- `third_party/native-torch/aten/src/ATen/native/**` and all backend kernel dirs
  listed in §A.3.
- `c10/core/impl/PyInterpreter*`, `PythonDispatcherTLS*`, `SafePyObject*`,
  `PyHandleCache*`, `PyObjectSlot*` — Python hooks in the compile closure.
- Low-level allocator internals (`c10/core/impl/alloc_cpu.*`, `CPUAllocator.cpp`,
  `CachingDeviceAllocator*`, `aten/core/CachingHostAllocator*`).
- Anything not enumerated in §A as COPY/ADAPTER/REIMPLEMENT.

Inspection of a black-box area is only permitted if a concrete compile/link
error names a specific symbol — and then only to the minimum extent to report
that symbol, per §17 of the discovery brief.

---

## E. Explicit exclusions (outside v0)

- Python bindings / Python frontend (`torch/**`, `torch/csrc/**`).
- `functorch/**`, `caffe2/**`, `torchgen/**` (codegen is a build tool only).
- `android/`, `benchmarks/`, `test/`, `docs/`, CI and GitHub config.
- Autograd, training-only machinery, distributed training, quantization.
- **Deferred dtype systems:** `Float8_*`, `UInt16/UInt32/UInt64`, quantized
  `qint*` (present in `c10/util` only because `ScalarType.h`/`ScalarType.cpp`
  reference them — see §F.4 dependency exceptions).
- **Deferred:** `SymInt`/`SymIntList`, `Generator`, higher-order control flow,
  generic code generation.

---

## F. Migration statistics

1. **Files staged now:** 413 (implementation files only; tests/build metadata
   excluded). Breakdown: `c10/core` 118, `c10/util` 157, `c10/macros` 3,
   `aten/src/ATen/core` 135.
2. **Directories staged now:** 4 (`c10/core`, `c10/util`, `c10/macros`,
   `aten/src/ATen/core`), plus their subdirs (`core/impl`, `core/dispatch`,
   `core/boxing`, `core/boxing/impl`, `core/op_registration`).
3. **Adapter files (planned, not yet written):** 15
   (14 adapters in §B + `core/layer1/registration/registration.cpp`).
4. **Excluded subsystems:** 7 top-level (`torch`, `functorch`, `caffe2`,
   `torchgen`, `android`, `benchmarks`, `test`) + backend dirs listed in §A.3.
5. **Unresolved dependencies:** 2
   (a) generated op headers/registration via `torchgen` + `native_functions.yaml`;
   (b) native kernels (`aten/src/ATen/native/**`) at link time.

### F.4 Dependency exceptions (deferred items forced into the compile closure)

These are outside v0 semantics but cannot be omitted from the staged source
because the c10/aten core will not compile without them. They are staged, and
are BLACK BOX for the coding phase:

- **SymInt** (`c10/core/SymInt.*`, `SymIntArrayRef.*`, `SymNodeImpl.*`,
  `ConstantSymNodeImpl.*`): tensor sizes/strides are `SymInt` in modern c10.
- **GeneratorImpl** (`c10/core/GeneratorImpl.*`, `aten/core/Generator.*`):
  referenced by `TensorImpl`.
- **Float8 / quantized scalar types** (`c10/util/Float8_*`, `qint*`):
  referenced by `ScalarType.h`/`ScalarType.cpp`.

---

## Provenance & licensing

Staged code is unmodified PyTorch source (BSD-3-Clause) copied from
`~/Project/pytorch`. It retains its original headers. The SonicBoom repository
`LICENSE` is GPL-3.0; the interaction of GPL-3.0 with the BSD-3-Clause
native-torch sources is a licensing decision for the project owner and is not
resolved by this manifest.
