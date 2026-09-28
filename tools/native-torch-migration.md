# SonicBoom Native-Torch Migration Manifest

**Status:** AUTHORITATIVE — this document is the source of truth for the
native-torch migration. Future coding agents must work from this manifest, not
rediscover PyTorch.

**Companion rules:** `CLAUDE.md` / `AGENTS.md` (architecture, boundaries,
migration + token-efficiency rules). This manifest is the concrete source set
that those rules anticipate.

---

## 1. Frozen Upstream Revision

- **Repository:** `~/Project/pytorch` (source reference only — not the project).
- **Commit:** `41ffbc4a994e058af9fe00ed5caba73fc1033359`
- **Tag/branch:** `v2.14.1-rc1` / `release/2.14`
- **version.txt:** `2.14.1a0`
- **Date recorded:** 2026-09-28

The manifest refers to exactly this revision. Do not change the upstream
revision, and do not update the PyTorch repository.

---

## 2. Migration Principles

1. **Requirement-driven, not directory-driven.** Every entry below exists
   because a SonicBoom v0 concept requires it, not because it lives near a
   required file.
2. **SonicBoom owns its interfaces.** Layer 2 (`core/include/sonicboom/`) never
   exposes native-torch types. `third_party/native-torch/` is a modifiable
   implementation area, but modifications are controlled and must be recorded
   here.
3. **Minimum closure.** Migrate the smallest source set that compiles the
   required Layer 1 capability; everything else is BLACK BOX or EXCLUDE.
4. **Python is not a runtime dependency.** Python tooling may participate in
   *building* generated sources (see §12) but is never linked or required at
   runtime.
5. **Token boundary.** BLACK BOX / EXCLUDE areas are marked so agents do not
   inspect them. If a compile/runtime failure proves one is needed, STOP and
   report (§13 / `CLAUDE.md` missing-dependency rule).

---

## 3. SonicBoom Layer Mapping

Requirement-driven map of v0 concepts → native-torch provider:

| SonicBoom concept | native-torch provider | Home |
|---|---|---|
| Tensor | `at::Tensor` / `at::TensorBase` (`aten/core/Tensor.h`, `TensorBase.h`) | `c10/core/TensorImpl.*` below it |
| Scalar | `c10::Scalar` | `c10/core/Scalar.*` |
| DType | `c10::ScalarType` | `c10/core/ScalarType.*` |
| Device | `c10::Device` / `c10::DeviceType` | `c10/core/Device.*`, `DeviceType.*` |
| Layout | `c10::Layout` | `c10/core/Layout.h` |
| MemoryFormat | `c10::MemoryFormat` | `c10/core/MemoryFormat.h` |
| Backend | `c10::Backend` | `c10/core/Backend.h` |
| Allocator | `c10::Allocator` + `c10::CPUAllocator` + `c10::impl::alloc_cpu` | `c10/core/Allocator.*`, `CPUAllocator.*`, `core/impl/alloc_cpu.*` |
| Value | `c10::IValue` | `aten/core/ivalue.*` |
| ArgumentList / ResultList | `c10::Stack` (`std::vector<c10::IValue>`) | `aten/core/stack.h` |
| OperatorHandle | `c10::OperatorHandle` | `aten/core/dispatch/Dispatcher.h` |
| OperatorSchema | `c10::FunctionSchema` | `aten/core/function_schema.*` |
| dispatch() | `c10::Dispatcher::callBoxed()` | `aten/core/dispatch/Dispatcher.*` |
| operator registration | `c10::Library` (`TORCH_LIBRARY`) | `aten/core/library.cpp`, `aten/core/op_registration/*` |

The middle of the boxed-invocation path (`IValue`, `Stack`, `Dispatcher`,
`KernelFunction`) is Layer 1 implementation detail.

---

## 4. Required Native Source

Classification legend: **COPY** (as-is) · **MODIFY** (must change) · **ADAPT**
(SonicBoom adapter wraps it) · **BLACK BOX** (below boundary, don't inspect) ·
**EXCLUDE** (out of v0).

Destination root is `third_party/native-torch/` (mirrors the upstream relative
path). The already-staged tree under that root corresponds to §4.1 (413 files).

### 4.1 COPY

Required implementation migrated substantially as-is. Grouped by subsystem;
the critical files are listed at file level, and the remainder of each named
directory shares the same treatment (the entire directory is in the upstream
library glob and is self-contained).

#### 4.1.1 c10 core — types, storage, allocator, dispatch keys

`c10/core/` → `third_party/native-torch/c10/core/` (COPY, whole directory).

Critical files:

| Upstream | Reason |
|---|---|
| `Scalar.h/.cpp`, `ScalarType.h/.cpp`, `ScalarTypeToTypeMeta.h` | Scalar / DType |
| `Device.h/.cpp`, `DeviceType.h/.cpp` | Device |
| `Layout.h`, `MemoryFormat.h`, `Backend.h` | Layout / MemoryFormat / Backend |
| `Allocator.h/.cpp`, `AllocatorConfig.h/.cpp`, `CPUAllocator.h/.cpp` | Allocator |
| `TensorImpl.h/.cpp`, `StorageImpl.h/.cpp`, `Storage.h/.cpp` | tensor/storage impl |
| `TensorOptions.h/.cpp`, `DefaultTensorOptions.h`, `DefaultDtype.h/.cpp` | tensor construction options |
| `DispatchKey.h/.cpp`, `DispatchKeySet.h/.cpp` | dispatch keys (Layer 1 only) |
| `Stream.h/.cpp`, `StreamGuard.h`, `Event.h` | stream/event |
| `GeneratorImpl.h/.cpp` | generator (deferred semantics — see exception §4.1.5) |
| `GradMode.h/.cpp`, `InferenceMode.h/.cpp`, `AutogradState.*` | mode guards (inert without autograd) |
| `SymInt.h/.cpp`, `SymIntArrayRef.*`, `SymNodeImpl.*`, `ConstantSymNodeImpl.*`, `SymBool.*`, `SymFloat.*`, `SymbolicShapeMeta.*` | SymInt (deferred semantics — see §4.1.5) |
| `Contiguity.h`, `CopyBytes.h/.cpp`, `WrapDimMinimal.*`, `DeviceGuard.h`, `OptionalRef.h`, `RefcountedDeleter.*`, `DynamicCast.h`, `alignment.h`, `CompileTimeFunctionPointer.h`, `RingBuffer.h`, `thread_pool.*`, `QScheme.h`, `QEngine.h`, `DeviceArray.h`, `DeviceCapability.h`, `CachingDeviceAllocator.h/.cpp`, `StorageMaterializer.h` | supporting primitives |

`c10/core/impl/` → `third_party/native-torch/c10/core/impl/` (COPY):

| Upstream | Reason |
|---|---|
| `SizesAndStrides.h/.cpp` | tensor sizes/strides storage |
| `alloc_cpu.h/.cpp` | CPU allocation |
| `DeviceGuardImplInterface.h/.cpp`, `InlineDeviceGuard.h`, `InlineStreamGuard.h`, `InlineEvent.h`, `VirtualGuardImpl.h`, `FakeGuardImpl.h` | device guards |
| `LocalDispatchKeySet.h/.cpp` | thread-local dispatch keys |
| `COW.h/.cpp`, `COWDeleter.h/.cpp` | storage copy-on-write |
| `FakeTensorModeTLS.*`, `TorchDispatchModeTLS.*`, `GPUTrace.*` | mode TLS (inert) |
| `PyInterpreter.h/.cpp`, `PyInterpreterHooks.h/.cpp`, `PyObjectSlot.h`, `PythonDispatcherTLS.*` | Python bridge — **inert, black-box** (see §4.4) |

#### 4.1.2 c10 util — containers, refcounting, scalar types

`c10/util/` → `third_party/native-torch/c10/util/` (COPY, whole directory).

Critical files: `ArrayRef.h`, `SmallVector.h/.cpp`, `intrusive_ptr.h/.cpp`,
`typeid.h/.cpp`, `Type.h`, `TypeIndex.h`, `int128.h/.cpp`, `string_view.h`,
`string_utils.h`, `StringUtil.h/.cpp`, `Exception.h/.cpp`, `Logging.h/.cpp`,
`Registry.h`, `UniqueVoidPtr.h/.cpp`, `Optional.h/.cpp`, `MaybeOwned.h`,
`FunctionRef.h`, `Metaprogramming.h`, `TypeTraits.h`, `hash.h`,
`flat_hash_map.h`, `order_preserving_flat_hash_map.h`, `LeftRight.h/.cpp`,
`Synchronized.h`, `ThreadLocal.h`, `CallOnce.h`, `ScopeExit.h`, `Lazy.h`,
`overloaded.h`, `IdWrapper.h`, `strong_type.h`, `irange.h`, `Enumerate.h`,
`accumulate.h`, `SmallBuffer.h`, `Bitset.h`, `bit_cast.h`, `bits.h`,
`Unroll.h`, `OverflowUtils` (`overflows.h`), `safe_numerics.h`, `safe_conv.*`,
`strides.h`, `DimVector.h`, `SmallVector.h`, `complex*.h/.cpp`, `MathConstants.*`,
`generic_math.h`, and dtype scalar types `Half.*`, `BFloat16*`, `Float8_*`,
`qint8/16/32.h`, `quint*` (see §4.1.5), plus platform glue (`C++17.h`,
`Deprecated.h`, `Macros`-adjacent, `win32-headers.h`, etc.).

#### 4.1.3 c10 macros

`c10/macros/` → `third_party/native-torch/c10/macros/` (COPY):
`Macros.h`, `Export.h`, `cmake_macros.h`.

#### 4.1.4 ATen core — Tensor, IValue, dispatcher, boxing, registration

`aten/src/ATen/core/` → `third_party/native-torch/aten/src/ATen/core/` (COPY,
whole directory). Critical files:

| Upstream | Reason |
|---|---|
| `Tensor.h`, `TensorBase.h`, `Tensor.cpp`, `TensorAccessor.h`, `ATen_fwd.h` | Tensor |
| `Scalar.h`, `ScalarType.h` (aliases into c10) | Scalar / DType re-export |
| `ivalue.h/.cpp`, `ivalue_inl.h`, `ivalue_to.h` | Value |
| `stack.h` | ArgumentList / ResultList |
| `function_schema.h/.cpp`, `function_schema_inl.h`, `operator_name.h/.cpp`, `alias_info.h` | OperatorSchema |
| `dispatch/Dispatcher.h/.cpp`, `dispatch/OperatorEntry.h/.cpp`, `dispatch/DispatchKeyExtractor.h/.cpp`, `dispatch/OperatorOptions.h`, `dispatch/ObservedOperators.h/.cpp`, `dispatch/RegistrationHandleRAII.h`, `dispatch/CppSignature.h` | dispatch() / OperatorHandle |
| `boxing/KernelFunction.h/.cpp`, `boxing/KernelFunction_impl.h`, `boxing/BoxedKernel.h`, `boxing/BoxedKernel_impl.h`, `boxing/OperatorKernel.h`, `boxing/impl/make_boxed_from_unboxed_functor.h`, `boxing/impl/WrapFunctionIntoFunctor.h`, `boxing/impl/WrapFunctionIntoRuntimeFunctor.h` | boxed kernel invocation |
| `op_registration/op_registration.h/.cpp`, `op_registration/infer_schema.h/.cpp`, `op_registration/adaption.h`, `op_registration/op_allowlist.h` | operator registration |
| `library.cpp` (top-level), `ATenOpList.h` | `c10::Library` implementation |
| `List.h/.cpp`, `List_inl.h`, `Dict.h/.cpp`, `Dict_inl.h`, `IListRef.h`, `IListRef_inl.h` | container values in IValue |
| `Generator.h/.cpp`, `GeneratorForPrivateuseone.*` | generator (deferred semantics) |
| `type.h/.cpp`, `jit_type.h`, `jit_type_base.h`, `dynamic_type.h/.cpp`, `enum_type.h`, `union_type.cpp`, `class_type.h/.cpp`, `tensor_type.cpp`, `type_factory.*`, `type_ptr.h`, `typeid.h`, `qualified_name.h`, `symbol.h`, `interned_strings.*`, `register_symbols.cpp`, `builtin_function.h`, `function.h`, `functional.h`, `Range.*` | JIT type/symbol machinery required by IValue/FunctionSchema (compile closure; inert at runtime) |
| `Reduction.h`, `Array.h`, `DistributionsHelper.h`, `TensorAccessor.h`, `TransformationHelper.h`, `CheckMemoryFormat.h`, `Variadic.h`, `enum_tag.h` | supporting primitives |
| `DeprecatedTypeProperties.*`, `DeprecatedTypePropertiesRegistry.*`, `LegacyTypeDispatch.h`, `UnsafeFromTH.h`, `UndefinedTensorImpl.h`, `OpaqueTensorImpl.h` (in `ATen/`), `MetaFallbackKernel.cpp`, `BackendSelectFallbackKernel.cpp`, `VariableFallbackKernel.cpp`, `PythonFallbackKernel.*`, `PythonOpRegistrationTrampoline.*`, `VariableHooksInterface.*`, `GradMode` (`grad_mode.h`), `CachingHostAllocator.*`, `Formatting.*`, `blob.*`, `adaption.cpp`, `custom_class.*`, `rref_interface.h`, `GraphImplInterface.*`, `TorchDispatchUtils.*` | fallback kernels / hooks / utilities (inert without their subsystems) |

#### 4.1.5 COPY with dependency-exception annotation

These are **deferred semantically** by v0 scope but are **unavoidable in the
compile closure** — they are staged as COPY and are BLACK BOX for coding:

| Upstream | Why it is in the closure despite deferral |
|---|---|
| `SymInt`/`SymIntArrayRef`/`SymNodeImpl`/`ConstantSymNodeImpl`/`SymBool`/`SymFloat`/`SymbolicShapeMeta` | `TensorImpl` sizes/strides are `SymInt` in 2.14. Cannot omit. |
| `GeneratorImpl` / `aten/core/Generator.*` | `TensorImpl` holds a generator pointer. Cannot omit. |
| `Float8_*`, `qint*`, `quint*`, `Half`, `BFloat16` scalar types | `ScalarType.h/.cpp` enumerates and references all scalar types. Cannot omit. |

Do **not** expand the migration to implement SymInt/generator/quantized
semantics. They are present only to satisfy the compiler.

#### 4.1.6 Additional COPY required by the ATen core closure (not yet staged)

`aten/core` includes these **top-level ATen headers**, which are part of the
closure and must be migrated (COPY) but are not yet staged (they transitively
require generated headers, see §4.2 and §10):

| Upstream | Reason |
|---|---|
| `aten/src/ATen/Tensor.h`, `TensorOperators.h`, `TensorUtils.h`, `TensorGeometry.h`, `TensorIndexing.h` | full `at::Tensor` API assembled over `core/Tensor.h` |
| `aten/src/ATen/Context.h`, `ATen.h`, `Dispatch.h`, `Dispatch_v2.h` | runtime context / dispatch access |
| `aten/src/ATen/Scalar.h`, `ScalarOps.h`, `ScalarType.h` (re-export), `Layout.h`, `Device.h`, `Backend.h`, `Generator.h` (re-exports) | type re-exports |
| `aten/src/ATen/NumericUtils.h`, `StorageUtils.h`, `SequenceNumber.h`, `record_function.h`, `MethodOperators.h`, `TensorAccessor.h`, `ArrayRef.h`, `SmallVector.h`, `DimVector.h`, `Formatting.h` (re-export), `InitialTensorOptions.h`, `EmptyTensor.h`, `ExpandUtils.h`, `InferSize.h`, `MemoryOverlap.h`, `Parallel.h`, `DeviceGuard.h` | supporting headers referenced by core |

The exact set is the transitive `#include` closure of `aten/core` — resolved by
compilation; do not migrate the entire `aten/src/ATen/` tree.

### 4.2 MODIFY

Required source that must be changed to work inside SonicBoom. All changes are
below the Layer 2 boundary and do **not** affect the public SonicBoom API.

| # | Source | Destination | Change | Why | Removes PyTorch assumption? | Affects public boundary? |
|---|---|---|---|---|---|---|
| M1 | generated operator registration (`RegisterSchema.cpp`, `RegisterCodegenUnboxedKernels*.cpp`, `Functions.h`, `NativeFunctions.h`, `UnboxingFunctions.h`, `ops/*.h`, `selected_mobile_ops.h`) | build-generated into the SonicBoom build dir (not checked into `third_party/native-torch/`) | Generate/register **only the v0 operator subset**, not the full op set | The full generated set is produced by `torchgen` over all `native_functions.yaml` and drags in the entire `aten/src/ATen/native/**` kernel library at link time. v0 must register a minimal op set. | Yes — drops the full-op-set assumption. | No |
| M2 | `aten/src/ATen/Config.h.in` | replaced by a SonicBoom-supplied `ATen/Config.h` | Provide config macros (e.g. `AT_CUDNN_ENABLED=0`, `AT_MKL_ENABLED=0`) via SonicBoom CMake | `Config.h` is CMake-configured upstream; SonicBoom owns its own build. | Yes — backend flags are SonicBoom's decision. | No |
| M3 | (none at c10/aten source level) | — | — | The c10 Python-bridge files (`PyInterpreter`, `PyObjectSlot`, `SafePyObject`, `PyHandleCache`, `PythonDispatcherTLS`) compile **without** Python because `PyObject*` is forward-declared opaque. No source modification required; do not "fix" them. | n/a | n/a |
| M4 | `c10/util/env.cpp` | `third_party/native-torch/c10/util/env.cpp` | Guard `#include <fmt/format.h>` behind `#ifdef _MSC_VER` | `fmt::format` is used only in the Windows `_MSC_VER` `set_env`/`unset_env` paths; POSIX uses `setenv`/`unsetenv` directly. Removes the fmt dependency for the CPU-only Linux build. See §15.C. | Yes — drops a Linux-dead fmt dependency. | No |
| M5 | `c10/core/CPUAllocator.cpp` | `third_party/native-torch/c10/core/CPUAllocator.cpp` | Guard the mobile `DefaultMobileCPUAllocator` (and its `c10/mobile/*` includes + `g_mobile_cpu_allocator` global) behind `#ifdef C10_MOBILE` | The mobile allocator only adds QNNPACK/XNNPACK guard bytes + thread-local caching/profiling (out of v0 scope); the non-mobile default `DefaultCPUAllocator` already calls `alloc_cpu`/`free_cpu` directly. Removes the `c10/mobile` dependency. See §15.D. | Yes — drops the mobile caching/profiling layer. | No |

The MODIFY work is therefore concentrated in the **build/generated-code
strategy** (M1, M2), not in rewriting native source.

### 4.3 ADAPT

Functionality that stays native-torch but is reached through a SonicBoom
Layer 1 adapter. See §5 for the full map. Adapter sources are SonicBoom-owned
(`core/layer1/adapter/*`) and are **not** part of `third_party/native-torch/`.

### 4.4 BLACK BOX

Required at link/runtime time but below the SonicBoom boundary. Coding agents
must not inspect or modify these unless the manifest (or a concrete failure)
authorizes it.

| Area | Why black-box |
|---|---|
| `aten/src/ATen/native/**` (all 19 subdirs: `cpu`, `cuda`, `mkl`, `mkldnn`, `cudnn`, `miopen`, `mps`, `metal`, `vulkan`, `quantized`, `sparse`, `nested`, `transformers`, `ufunc`, `utils`, `xnnpack`, `ao_sparse`, `kleidiai`, `hip`) | native operator kernels — needed only so `callBoxed()` reaches a real kernel. 28 MB. |
| `aten/src/ATen/{cpu,cuda,mkl,mkldnn,cudnn,miopen,mps,metal,vulkan,hip,xpu}/**` | backend vectorized math + kernel dispatchers |
| `c10/{cuda,hip,xpu,metal}/**` | backend device guard/stream/allocator impls |
| c10 Python bridge: `c10/core/impl/PyInterpreter.*`, `PyInterpreterHooks.*`, `PyObjectSlot.h`, `PythonDispatcherTLS.*`, `c10/core/SafePyObject.*`, `PyHandleCache.h` | inert without a Python interpreter; part of `TensorImpl`'s compile closure |
| allocator internals: `c10/core/impl/alloc_cpu.*`, `CPUAllocator.cpp`, `CachingDeviceAllocator.*`, `aten/core/CachingHostAllocator.*` | low-level allocation; not needed to understand Layer 1 |

### 4.5 EXCLUDE

Not part of SonicBoom v0.

| Area | Reason |
|---|---|
| `torch/**` (incl. `torch/csrc/**`, `torch/_*/**` Python) | Python frontend + bindings |
| `functorch/**` | vmap / batching frontend |
| `caffe2/**` | legacy |
| `torchgen/**` | Python codegen tool — used only at build time if M1 chooses torchgen (see §12); never migrated into the runtime |
| `android/**`, `benchmarks/**`, `test/**`, `docs/**`, `.ci/**`, `.github/**`, `scripts/**`, `binaries/**`, `mypy_plugins/**` | not native runtime implementation |
| autograd (`torch/csrc/autograd`, `aten` autograd hooks beyond the inert compile-closure stubs) | deferred |
| distributed / quantization / training machinery | deferred |
| unsupported dtype systems semantics (float8, UInt16/32/64) | deferred (symbols only, per §4.1.5) |
| `torch/compiler` / higher-order control flow / `torch/export` compiler stack | out of scope (see §13) |

---

## 5. Layer 1 Adapter Map

SonicBoom abstraction → native-torch type → adapter source (all SonicBoom-owned
under `core/layer1/adapter/`):

| SonicBoom abstraction | native-torch type | adapter source |
|---|---|---|
| `Tensor` | `at::Tensor` | `core/layer1/adapter/tensor.cpp` |
| `Scalar` | `c10::Scalar` | `core/layer1/adapter/scalar.cpp` |
| `DType` | `c10::ScalarType` | `core/layer1/adapter/dtype.cpp` |
| `Device` | `c10::Device` | `core/layer1/adapter/device.cpp` |
| `Layout` | `c10::Layout` | `core/layer1/adapter/layout.cpp` |
| `MemoryFormat` | `c10::MemoryFormat` | `core/layer1/adapter/memory_format.cpp` |
| `Backend` | `c10::Backend` | `core/layer1/adapter/backend.cpp` |
| `Allocator` | `c10::Allocator` | `core/layer1/adapter/allocator.cpp` |
| `Value` | `c10::IValue` | `core/layer1/adapter/value.cpp` |
| `ArgumentList` | `c10::Stack` | `core/layer1/adapter/argument_list.cpp` |
| `ResultList` | `c10::Stack` | `core/layer1/adapter/result_list.cpp` |
| `OperatorHandle` | `c10::OperatorHandle` | `core/layer1/adapter/operator_handle.cpp` |
| `OperatorSchema` | `c10::FunctionSchema` | `core/layer1/adapter/operator_schema.cpp` |
| `dispatch()` | `c10::Dispatcher::callBoxed()` | `core/layer1/adapter/dispatch.cpp` |
| operator registration | `c10::Library` (`TORCH_LIBRARY`) | `core/layer1/registration/registration.cpp` |

Public API firewall: **none** of the native-torch types in the middle column may
appear in `core/include/sonicboom/`. The adapter is the only place where the
conversion is visible.

---

## 6. Dependency Closure

Minimum native-torch closure per required subsystem (all within §4.1/§4.2):

- **Tensor** → `core/Tensor.h` + `TensorBase.h` → `c10/core/TensorImpl.*` →
  `StorageImpl.*` → `c10/core/impl/SizesAndStrides.*` → `c10/core/Allocator.*`
  → `c10/util/intrusive_ptr.h` + `typeid.h` (TypeMeta). Plus `ATen/Tensor.h`
  closure (§4.1.6).
- **Operator** → `core/dispatch/Dispatcher.*` → `OperatorEntry.*` →
  `DispatchKeyExtractor.*` → `c10/core/DispatchKey(Set).*` → `boxing/KernelFunction.*`.
- **Value** → `core/ivalue.*` → `core/List.*`/`Dict.*` → `core/stack.h` →
  `c10/util/intrusive_ptr.h` + `typeid.h` + `Optional.h`.
- **DType/Device/Layout/MemoryFormat** → `c10/core/ScalarType.*`,
  `Device.*`/`DeviceType.*`, `Layout.h`, `MemoryFormat.h`.
- **Backend** → `c10/core/Backend.h` + `DispatchKey.*` (backend registration is
  implicit in dispatch keys; no separate backend registry needed in v0).
- **Schema** → `core/function_schema.*` + `operator_name.*` + `alias_info.h` +
  `jit_type.h` (argument/return type representation).

---

## 7. Dispatcher / Boxed Invocation Closure

The v0 runtime path and its exact minimum source:

```
SonicBoom OperatorHandle + ArgumentList
   → adapter dispatch.cpp
   → c10::Dispatcher::callBoxed(schema, Stack)     [dispatch/Dispatcher.cpp]
   → Dispatcher → OperatorEntry lookup             [dispatch/OperatorEntry.cpp]
   → DispatchKeyExtractor::computeDispatchKeySet   [dispatch/DispatchKeyExtractor.cpp]
   → KernelFunction::callBoxed(stack)              [boxing/KernelFunction.cpp]
   → boxed → unboxed functor adapter               [boxing/impl/WrapFunctionIntoRuntimeFunctor.h]
   → generated RegisterCodegenUnboxedKernels / RegisterSchema   [M1, generated]
   → native kernel (aten/src/ATen/native/**)       [BLACK BOX]
   → results written back into the Stack
   → adapter result_list.cpp → SonicBoom ResultList
```

Minimum native-torch source for this path: `dispatch/Dispatcher.*`,
`dispatch/OperatorEntry.*`, `dispatch/DispatchKeyExtractor.*`,
`dispatch/OperatorOptions.h`, `dispatch/RegistrationHandleRAII.h`,
`boxing/KernelFunction.*`, `boxing/BoxedKernel*.h`, `boxing/OperatorKernel.h`,
`boxing/impl/WrapFunctionInto{Functor,RuntimeFunctor}.h`,
`boxing/impl/make_boxed_from_unboxed_functor.h`, `function_schema.*`,
`operator_name.*`, `ivalue.*`, `stack.h`, `op_registration/op_registration.*`,
`op_registration/infer_schema.*`, `library.cpp`, and the generated registration
files (M1). **Do not** migrate the rest of the dispatcher subsystems
(e.g. `torch/csrc` dispatcher tooling, autograd dispatch keys) — they are EXCLUDE.

---

## 8. Tensor / Storage / Allocator Closure

```
at::Tensor → c10::TensorImpl (intrusive_ptr) → c10::StorageImpl → c10::DataPtr
   → c10::Allocator (interface) → c10::CPUAllocator → c10::impl::alloc_cpu
```

Minimum source: `c10/core/TensorImpl.*`, `StorageImpl.*`, `Storage.*`,
`Allocator.*`, `AllocatorConfig.*`, `CPUAllocator.*`, `DefaultDtype.*`,
`TensorOptions.*`, `RefcountedDeleter.*`, `core/impl/SizesAndStrides.*`,
`core/impl/alloc_cpu.*`, `core/impl/COW*`, `c10/util/intrusive_ptr.*`,
`c10/util/typeid.*` (TypeMeta), `c10/util/UniqueVoidPtr.*`,
`aten/core/Tensor.*`/`TensorBase.h`.

---

## 9. Backend Closure

- **CPU — required for v0.** Compile `c10/core` + `aten/core` with CPU
  dispatch keys; link `aten/src/ATen/native/cpu/**` + `aten/src/ATen/native/**`
  CPU kernels (BLACK BOX).
- **CUDA — deferred / optional.** Not required for the v0 minimum. The c10 core
  carries CUDA dispatch-key *types* (compile-time), but a CPU-only build must
  not link CUDA (`USE_CUDA=OFF`). Treat `aten/src/ATen/cuda/**`,
  `native/cuda/**`, `c10/cuda/**` as EXCLUDE for v0 (revisit only if a
  CUDA-backed Device becomes a requirement).
- **Other backends (MPS/Metal/Vulkan/XLA/HPU/XPU/MKL/MKLDNN/cuDNN/MIOpen)** —
  EXCLUDE for v0.

---

## 10. Build-System Requirements

SonicBoom owns its build; do **not** copy PyTorch's `CMakeLists.txt`/`cmake/`
tree. What SonicBoom must provide to compile the migrated source:

**Source sets (from §4.1):** `c10/core`, `c10/util`, `c10/macros`,
`aten/src/ATen/core`, plus the §4.1.6 top-level ATen header closure.

**Include directories:** `third_party/native-torch/c10`,
`third_party/native-torch/aten/src`, and the generated-header output dir.

**Compile definitions (subset of upstream's, decided by SonicBoom):**
`C10_BUILD_MAIN_LIB` (private), `AT_PER_OPERATOR_HEADERS` (if per-op headers are
generated), `AT_CUDNN_ENABLED=0`, `AT_MKL_ENABLED=0`, `USE_CUDA=OFF`, plus
platform/visibility macros from `c10/macros/Export.h`.

**Generated files (M1):** `ATen/Config.h` (from `Config.h.in`), `ATen/Functions.h`,
`ATen/NativeFunctions.h`, `ATen/UnboxingFunctions.h`, `ATen/ops/*.h`,
`RegisterSchema.cpp`, `RegisterCodegenUnboxedKernels*.cpp`,
`CompositeRegistrations.cpp` (only if composite ops are used). Produced by
`torchgen` (build-time) over a v0-scoped schema list, or pre-generated and
committed to a SonicBoom-owned generated dir.

**Libraries:** none beyond the C++ standard library for the core; the native
kernels pull in their own backend deps (BLACK BOX, link-time).

---

## 11. CUDA / CPU Requirements

| Backend | Status | Notes |
|---|---|---|
| CPU | **required** | `c10/core` CPU allocator + `native/cpu` kernels |
| CUDA | **deferred** | not in v0; keep `USE_CUDA=OFF` |
| MPS / Metal / Vulkan / ROCm / XPU / HPU | **excluded** | not in v0 |
| MKL / MKLDNN / cuDNN / MIOpen | **excluded** | not in v0 |

---

## 12. Python / Non-runtime Dependencies

- **Python is not a runtime dependency.** The staged runtime links no Python
  and requires no Python interpreter.
- **Build-time Python is possible (M1):** `torchgen` (a Python tool) generates
  the operator schema/registration C++ from `native_functions.yaml` +
  `tags.yaml`. If SonicBoom chooses to run `torchgen`, that is a **build-time**
  dependency only. Alternatives: pre-generate and commit, or hand-write a
  minimal registration — a decision to make before migration (§13).
- **c10 Python-bridge files** compile without Python (opaque `PyObject*`
  forward decls) and are inert; see §4.1.5/M3.
- Generated metadata or build artifacts produced by Python tooling must be
  documented at migration time, not migrated as Python source.

---

## 13. Unresolved Questions

1. **Generated-code strategy (M1):** run `torchgen` at build, pre-generate and
   commit, or hand-write minimal registration for the v0 op subset? This is the
   largest open decision — it determines whether `torchgen` + full
   `native_functions.yaml` are build dependencies.
2. **v0 operator list:** the exact operator set for the ExportedProgram-style
   interpreter is not yet enumerated. It bounds which native kernels (BLACK BOX)
   must actually link, and what M1 must generate. ExportedProgram is treated as
   an **external input format**, not a native-torch runtime dependency — the
   runtime only needs operator lookup + boxed execution (§3).
3. **CUDA:** confirmed deferred; if a later v0.x requires CUDA, the backend
   closure (§9/§11) must be re-opened as an explicit manifest amendment.
4. **SymInt handling:** SymInt is in the compile closure (§4.1.5) but deferred
   semantically. Confirm v0 can run with constant (non-symbolic) sizes only, or
   whether a minimal SymInt unboxing path must be enabled.

---

## 14. Migration Statistics

| Metric | Count |
|---|---|
| COPY entries | 4 directories staged (413 files) + §4.1.6 top-level ATen header closure (transitive; resolved at compile) + generated headers (M1) |
| MODIFY entries | 3 (M1 generated registration subset, M2 Config.h, M3 "no source change — do not modify Python bridge") |
| ADAPT entries | 15 (14 adapters + 1 registration) |
| BLACK BOX areas | 4 (native kernels, backend dirs, c10 backend dirs, Python bridge + allocator internals) |
| EXCLUDE areas | 6 (torch, functorch, caffe2, torchgen, non-runtime dirs, autograd/distributed/quantization/dtype-systems) |
| Unresolved dependencies | 4 (§13) |
| Dependency exceptions (§4.1.5) | SymInt family, GeneratorImpl, Float8/quantized scalar symbols |

---

## Provenance & Licensing

Staged code is unmodified PyTorch source (BSD-3-Clause) from the frozen
revision (§1), retaining its original headers. The SonicBoom repository
`LICENSE` is GPL-3.0; reconciling GPL-3.0 with the BSD-3-Clause native-torch
sources is a licensing decision for the project owner and is not resolved here.

---

## Migration Freeze Audit

Audited against frozen revision `41ffbc4a994e058af9fe00ed5caba73fc1033359`.

### SymInt Boundary — PASS

**Evidence.** `c10/core/TensorImpl.h` includes `SymInt.h`/`SymIntArrayRef.h`
and exposes **both** concrete and symbolic accessors: `IntArrayRef sizes()`
and `IntArrayRef strides()` (concrete `int64_t`), alongside
`SymIntArrayRef sym_sizes()` / `sym_strides()` and `c10::SymInt sym_numel()`.
`c10/core/impl/SizesAndStrides.h` stores the concrete (non-symbolic) sizes and
strides — the fast path that v0 static shapes use.

**Conclusion.** The intended boundary is sufficient:

- native-torch internal: SymInt/SymNode exist (compile closure only).
- Layer 2: SymInt is **not** exposed.
- v0 semantics: concrete/static sizes only — Layer 1 reads `sizes()`/`strides()`.
- Layer 1: no SymInt unboxing required for the v0 static-shape path.

**Minimum internal SymInt closure** (already in §4.1.5 COPY, black-box for
coding): `c10/core/SymInt.*`, `SymIntArrayRef.*`, `SymNodeImpl.*`,
`ConstantSymNodeImpl.*`, `SymBool.*`, `SymFloat.*`, `SymbolicShapeMeta.*`, and
`aten/core/NestedIntSymNodeImpl.*`. No new Layer 2 API is introduced.

### COPY Closure Verification — PASS

**Evidence.** The COPY closure (`c10/core`, `c10/util`, `c10/macros`,
`aten/src/ATen/core`) contains **no** `#include <ATen/native/...>` and no
backend-kernel includes. The single apparent cross-boundary reference is
`c10/util/generic_math.h → c10/cuda/CUDAMathCompat.h`, and it is **guarded**:

```cpp
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  #include <c10/cuda/CUDAMathCompat.h>   // device-compile only
#else
  #include <c10/util/copysign.h>         // CPU path
#endif
```

`CUDAMathCompat.h` itself is `#if defined(__CUDACC__) || defined(__HIPCC__)`
guarded and self-contained (only `c10/macros/Macros.h` + `c10/util/Exception.h`,
both in COPY). **A CPU-only build requires no `c10/cuda` file.**

The genuine compile boundary is **generated headers, not native kernels**:
`aten/core/Tensor.cpp` includes `ATen/ops/{contiguous,fill,to,zero}_ops.h`
(declarations generated by M1); the definitions live in `aten/src/ATen/native/**`
(BLACK BOX) and are only needed at link time. Therefore the COPY closure
compiles with a v0-scoped generated-header set and links only the v0 kernel
subset — it does **not** drag in the native tree.

### v0 Operator Inventory — PASS (mechanism established)

The inventory mechanism is established; the list itself is not frozen here
because the final op set depends on the later ExportedProgram workload.

- **Authoritative source of the op list:** a v0-scoped subset of
  `aten/src/ATen/native/native_functions.yaml` (2585 `- func:` entries upstream,
  with `tags.yaml`) — or an equivalent hand-written list of `FunctionSchema`
  definitions. Upstream `native_functions.yaml` is torchgen **input metadata**,
  not a runtime dependency.
- **Per-operator entry must contain:** operator name (namespace + overload),
  the `FunctionSchema` (signature + returns), and the dispatch-key kernel(s) it
  is registered under.
- **torchgen/generated metadata actually required (M1):** schema registration
  (`RegisterSchema.cpp` → `m.def(schema)`), the boxed→unboxed wrapper
  (`RegisterCodegenUnboxedKernels*.cpp`), and the per-op declaration headers
  (`ops/*.h`, `Functions.h`) needed to compile `aten/core/Tensor.cpp`.
- **Native implementation required to include an op:** a kernel registered for
  that op under CPU (and `Composite*` where applicable) dispatch keys, reachable
  via `c10::Dispatcher::findSchema()` + `callBoxed()`.

**Minimal bootstrap set** (to prove the runtime; not the final v0 list):
tensor creation (`empty`/`zeros`/`ones`/`full`), tensor metadata
(`sizes`/`strides`/`dtype`/`device`/`dim`/`numel`), basic arithmetic
(`add`/`sub`/`mul`), basic shape ops (`reshape`/`view`), operator lookup
(`findSchema`), and boxed invocation (`callBoxed`).

### CUDA Boundary — PASS

**Evidence.** CUDA references inside the COPY closure are enum/comment-level
(`DispatchKey::CUDA`, `DeviceType::CUDA`, `Backend::CUDA`) or guarded device
code; none is a link-time CUDA dependency. The only file-level cross-reference
(`generic_math.h → CUDAMathCompat.h`) is device-guarded (see above).

**Conclusion.** The closure is CPU-first with CUDA explicitly deferred and
`USE_CUDA=OFF`. **Reopen condition:** CPU migration/build is frozen first; CUDA
becomes a separate manifest amendment if and only if a CUDA-backed `Device`
becomes a v0 requirement. Do not expand the source closure now for hypothetical
CUDA support.

### Freeze Status — READY TO FREEZE

No concrete blocker found. This audit resolves prior §13 open items:
- §13 Q3 (CUDA) — resolved: CPU-first, CUDA deferred with a defined reopen
  condition.
- §13 Q4 (SymInt) — resolved: concrete-size fast path suffices; no new Layer 2
  API.

Remaining decisions that do **not** block freezing the migration scope but must
be made before/at implementation: the generated-code strategy (M1: torchgen vs.
pre-generate vs. hand-write) and the exact v0 operator list (bounded by the
ExportedProgram workload).

---

## 15. Dependency-Closure Amendment

**Status:** AUTHORITATIVE amendment to §4, produced by the first real CPU compile
of the c10 closure. Each entry is a direct response to a concrete compiler
finding; this does not reopen PyTorch archaeology.

### A. `torch/headeronly/**` — ADDED (COPY, header-only subset)

**Why the original closure was incomplete.** At the frozen revision
`41ffbc4a994e058af9fe00ed5caba73fc1033359`, PyTorch migrated many small
definitions out of c10/aten-core into `torch/headeronly/`, leaving the
c10/aten-core headers as thin forwarding shims (`#include <torch/headeronly/...>`).
The freeze audit's closure check verified only the absence of `ATen/native` and
`c10/cuda` includes, so it did not detect this shim layer. The staged c10
closure therefore could not compile without the `torch/headeronly` target
headers.

**Why these are non-Python implementation headers.** `torch/headeronly/` is a
header-only C++ tree (no `.py`, no `.cpp`, no runtime). The staged subset is
macros, scalar types, and metaprogramming/utility headers — implementation
detail, not the Python frontend (`torch/csrc`, `torch/_*`).

**Why the full `torch/` tree remains EXCLUDED.** `torch/**` (§4.5) still means
the Python frontend + bindings + `torch/csrc` runtime. This amendment adds only
the specific header-only files below, mirroring the staged include path.

**Exact files added** (staged under `third_party/native-torch/torch/headeronly/`;
38 files: 37 headers + 1 configure template):

```text
macros/Macros.h   macros/Export.h   macros/cmake_macros.h.in
core/DeviceType.h   core/Dispatch.h   core/Dispatch_v2.h   core/Layout.h
core/MemoryFormat.h   core/ScalarType.h   core/TensorAccessor.h
util/BFloat16.h   util/bit_cast.h   util/bits.h   util/complex.h   util/complex_utils.h
util/Deprecated.h   util/Exception.h   util/Float4_e2m1fn_x2.h   util/Float8_e4m3fn.h
util/Float8_e4m3fnuz.h   util/Float8_e5m2.h   util/Float8_e5m2fnuz.h
util/Float8_e8m0fnu.h   util/Float8_fnuz_cvt.h   util/floating_point_utils.h
util/Half.h   util/HeaderOnlyArrayRef.h   util/Metaprogramming.h   util/NumericUtils.h
util/qint32.h   util/qint8.h   util/quint2x4.h   util/quint4x2.h   util/quint8.h
util/TypeList.h   util/TypeSafeSignMath.h   util/TypeTraits.h   util/win32-headers.h
```

**Deliberately NOT staged** (still EXCLUDE / BLACK BOX):

- `torch/headeronly/cuda/**` (`Atomic.h`, `KernelUtils.h`, `detail/ROCmMacros.h`)
  — CUDA; a self-referential island not reached by the CPU closure (deferred).
- `torch/headeronly/cpu/vec/**` (`intrinsics.h`, `vec_half.h`) — guarded by
  `CPU_CAPABILITY_AVX2/AVX512`; not in the minimal non-AVX v0 build.
- `torch/headeronly/core/enum_tag.h` and `macros/cmake_macros.h` — GENERATED:
  `enum_tag.h` is torchgen output (M1, aten/core only); `cmake_macros.h` is
  configured from `cmake_macros.h.in` by SonicBoom CMake (all flags OFF).
- `shim_utils.h`, `*.bzl`, `BUCK.oss`, `CMakeLists.txt`, `README.md` — unused /
  build metadata.

### B. `cpuinfo` — EXCLUDED

`c10/core/thread_pool.cpp` has an unconditional `#include <cpuinfo.h>`, but
**nothing in the staged c10 core closure includes `c10/core/thread_pool.h`**:
the thread pool is a parallelism primitive consumed by native kernels (BLACK
BOX), not by the c10 types/dispatch/IValue path. `thread_pool.cpp` is therefore
excluded from the c10 build (SonicBoom CMake). When the native kernels that need
intra-op parallelism are linked (BLACK BOX), `thread_pool.cpp` + `cpuinfo` can be
re-introduced as an explicit amendment.

### C. `fmt` — EXCLUDED

The unconditional fmt users are resolved without adding the fmt library:

- `c10/util/signal_handler.cpp` — orphaned (no references); excluded from build.
- `c10/util/tempfile.cpp` — orphaned (no references); excluded from build.
- `c10/util/env.cpp` — REQUIRED (`get_env` is used by `AllocatorConfig.cpp` and
  `Logging.cpp`), but its only `fmt::format` calls are inside `#ifdef _MSC_VER`
  (Windows `set_env`/`unset_env`); POSIX uses `setenv`/`unsetenv` directly.
  Guarded the `#include <fmt/format.h>` behind `_MSC_VER` (new MODIFY entry M4).
- `c10/util/strong_type.h` — already guarded (`STRONG_HAS_FMT_FORMAT` defaults to
  0); no change.

Result: no fmt dependency for the CPU-only Linux v0 build.

### D. `c10/mobile` — EXCLUDED (M5, CPUAllocator mobile-layer removal)

`c10/core/CPUAllocator.cpp` unconditionally included
`c10/mobile/CPUCachingAllocator.h` and `c10/mobile/CPUProfilingAllocator.h` and
compiled a `DefaultMobileCPUAllocator` (guard-byte safety for QNNPACK/XNNPACK),
on the rationale that it "must always be present even on non-mobile builds."

**Inspection result.** The non-mobile default `DefaultCPUAllocator` — selected
whenever `C10_MOBILE` is undefined, i.e. in every SonicBoom v0 build — already
calls `c10::alloc_cpu`/`c10::free_cpu` directly and is independent of c10/mobile.
The `DefaultMobileCPUAllocator` adds only:

- guard bytes (8 pre / 16 post) for QNNPACK/XNNPACK out-of-bounds SIMD access;
- a thread-local caching allocator (memory reuse; upstream-commented
  "experimental", "only used in StaticRuntime");
- a thread-local profiling allocator;
- an allocation planner.

None of these is a v0 requirement. QNNPACK/XNNPACK are mobile/quantized-inference
(EXCLUDE); caching/profiling/planning are mobile performance/profiling concerns,
not correctness.

**Modification (M5).** Guarded `DefaultMobileCPUAllocator`, its `c10/mobile/*`
includes, and `g_mobile_cpu_allocator`/`GetDefaultMobileCPUAllocator` behind
`#ifdef C10_MOBILE`. `C10_MOBILE` is never defined in v0, so the v0 CPU allocator
is `DefaultCPUAllocator` → `alloc_cpu`/`free_cpu`.

**Preserved** (all in `alloc_cpu.cpp`, staged): allocation (`posix_memalign`,
`gAlignment`), deallocation (`free`), zero-size handling (`nullptr`), negative-size
enforcement, allocation-failure reporting (`CAFFE_ENFORCE` → `c10::Error`),
NUMA placement, zero/junk-fill flags, and the `ReportAndDelete` deleter
(profiler hook + `free_cpu`).

**Intentionally omitted** (mobile-only): guard bytes, caching memory reuse,
profiling allocation tracking, allocation planning.

**Contract unchanged.** The SonicBoom v0 allocator contract (allocation /
deallocation / alignment / size handling / failure / deleter) is fully preserved;
only the mobile caching/profiling layer is removed.

## 16. M1 — Minimal v0 Operator Generation (implemented)

**Status:** M1 boundary complete. Establishes the minimal ATen/core build, the
hand-authored generated headers, and a dispatch proof. This implements MODIFY M1
(minimal op subset) by **hand-authoring** the generated headers rather than
running torchgen over the full operator universe — per the M1 directive ("not
full torchgen, not all ops").

### A. Minimal aten/core compile set (build target `aten_core`)

`aten_core` (static) compiles the dispatcher core — schema, type, IValue,
dispatcher, boxing — as the v0 compile closure. Sources are globbed from
`aten/src/ATen/core/**` with these exclusions (deferred, outside the M1
boundary):

```text
library.cpp                        torch/csrc library frontend + fmt
Formatting.cpp                     tensor pretty-print (fmt + op methods)
adaption.cpp                       ATen/Tensor.h top-level op facade (needs ops/*)
Tensor.cpp                         needs generated ATen/ops/*.h (added with ops)
op_registration/op_registration.cpp   function_schema_parser.h (JIT) + fmt
op_registration/infer_schema.cpp      fmt (findSchemaDifferences split out, §16.D)
Generator*.cpp, NestedIntSymNodeImpl.cpp   deferred (§4.1.5)
PythonFallbackKernel.cpp, VariableFallbackKernel.cpp, VariableHooksInterface.cpp
BackendSelectFallbackKernel.cpp, MetaFallbackKernel.cpp
custom_class.cpp, TorchDispatchUtils.cpp
```

### B. Hand-authored generated headers (MODIFY M1, implemented by hand)

Three torchgen outputs are required by the aten/core closure. Each is
hand-authored to the minimal v0 subset:

| File | Role | Minimal subset |
|---|---|---|
| `torch/headeronly/core/enum_tag.h` | dispatcher schema tag enum (`at::Tag`) | full 20 tags from `tags.yaml`, in order |
| `aten/src/ATen/core/aten_interned_strings.h` | `FORALL_ATEN_BASE_SYMBOLS` / `FORALL_ATTR_BASE_SYMBOLS` | proof set + compile-closure ops (10); attrs empty |
| `aten/src/ATen/core/TensorBody.h` | `at::Tensor` type surface + op-method declarations | type surface + 11 op-method declarations |

The dispatcher core addresses operators by string `OperatorName`, not interned
`c10::Symbol`, so the aten symbol table is not load-bearing for dispatch — a
minimal hand-authored list is sufficient.

### C. Minimal generated operator-method definitions (`ops/minimal_ops.cpp`)

`ivalue.cpp`'s comparison / hash / deepcopy paths call 12 Tensor/TensorBase
methods (`is_nonzero`, `eq`, `lt`, `clone`, `indices`, `coalesce`, `values`,
`_values`, `_indices`, `crow_indices`, `col_indices`, `to`). These are normally
torchgen output (`ATen/ops/<op>.cpp`). SonicBoom hand-authors them as thin
dispatcher calls (`findSchemaOrThrow` + `callBoxed`); `TensorBase::to`
(TensorOptions overload) is a defined-but-throwing placeholder since it is
outside the proof set.

### D. New / modified sources

| Source | Category | Change |
|---|---|---|
| `aten/src/ATen/SequenceNumber.cpp` | COPY | migrated as-is (sequence counter) |
| `aten/src/ATen/core/PythonOpRegistrationTrampoline.cpp` | COPY | un-excluded (self-contained `PyInterpreter*` holder; no Python deps) |
| `aten/src/ATen/record_function.cpp` | ADAPT | no-op profiler hooks (`getStepCallbacksUnlessEmpty` → nullopt) |
| `aten/src/ATen/core/op_registration/find_schema_differences.cpp` | ADAPT | fmt-free reimplementation (was in `infer_schema.cpp`) |
| `aten/src/ATen/core/Formatting.h` | MODIFY | inline `operator<<(Tensor)` prints placeholder instead of `at::print` (fmt) |

### E. Dispatch proof (`tests/dispatch_proof.cpp`)

A smoke test proving the M1 mechanism end-to-end:

```text
FunctionSchema -> Dispatcher::registerDef -> registerImpl (boxed kernel)
  -> findSchema / findOp -> OperatorHandle::callBoxedForDispatchKey
```

Runs `aten::relu` boxed on the CPU key; passes. This proves registration +
dispatcher lookup + boxed invocation without the Tensor op facade or any Layer 1
adapter.

### F. Operator inventory (proof set vs. link-forced set)

The v0 operator subset is the union of:

- **Intended proof set** (from `Tensor.cpp` op-dispatch methods): `contiguous`,
  `fill_`, `to`, `zero_` — to be registered when `Tensor.cpp`/`adaption.cpp` are
  enabled (deferred past M1).
- **Link-forced set** (`ivalue.cpp` closure): the 12 methods in §16.C — defined
  as dispatcher calls to satisfy the linker.

M1 registers neither set as real kernels; the proof registers `aten::relu`
directly. Layer 1 adapters are out of scope (next phase).

## 17. M2 — Layer 2 Stable Interface Boundary (implemented)

**Status:** M2 boundary complete. Establishes the first real Layer 2 / Layer 1
boundary: a stable, compilable, testable **Native Torch** public API (`nt::`)
under `core/include/sonicboom/` that exposes **zero** native-torch types, plus a
Layer 1 adapter (`core/layer1/`) that bridges it to the M1 dispatcher. This is
architectural proof, not new functionality. `dispatch_proof` continues to pass.

### A. Public API (Layer 2, `core/include/sonicboom/`, namespace `nt`)

No `c10::`/`at::`/`ATen` type is reachable from any public header (verified by
grep; only comment mentions remain). New types, all SonicBoom-owned:

| Header | Type(s) |
|---|---|
| `backend.h` | `BackendId { uint16_t value; int32_t priority; }` + `BackendId::cpu()`; `enum class Functionality { Dense, Sparse, Quantized, Autograd }` |
| `scalar_type.h` | `enum class ScalarType` (v0 dtypes) |
| `device.h` | `enum class DeviceType { CPU }`; `struct Device { type; index; }` + `Device::cpu()` |
| `layout.h` | `enum class Layout { Strided, Sparse }` |
| `memory_format.h` | `enum class MemoryFormat { Contiguous, Preserve, ChannelsLast }` |
| `scalar.h` | `Scalar` = `std::variant<int64_t, double, bool>` + `is*`/`to*` |
| `value.h` | `Value` = `std::variant` over Native Torch-owned types + `ValueKind`; `is*`/`to*` accessors |
| `argument.h` | `enum class ArgKind` (15 kinds) + `Argument { name; kind; }` |
| `schema.h` | `OperatorSchema { name; overload_name; arguments; returns; }` + `num_args()`/`num_returns()` |
| `argument_list.h` | `ArgumentList` / `ResultList` (Layer-2-owned `std::vector<Value>` wrapper; not `c10::Stack`) |
| `operator_handle.h` | `OperatorName` + `OperatorHandle` (pimpl) + `find_operator()` |
| `registration.h` | `KernelFn`, `RegistrationHandle` (RAII), `define_operator()`, `register_kernel()` |
| `tensor.h` | `Tensor` (pimpl) + `empty(sizes, dtype)` factory |
| `allocator.h` | `Allocator` interface (C-compatible deleter `void(*)(void*,void*)`, no `std::function`) |
| `sonicboom.h` | umbrella re-export |

**Key decisions:**
- **Namespace** `nt` (per M2 directive); directory stays `core/include/sonicboom/`
  with include prefix `<sonicboom/...>`. Dir-vs-namespace mismatch is intentional
  and cheap to rename later.
- **`Value` is a variant, not `c10::IValue`.** Alternative order in `data_`
  matches `ValueKind` index 0 == None.
- **`Tensor` is an opaque pimpl** (`std::shared_ptr<detail::TensorImpl>`), with
  `detail::Adapter` as friend. `detail::TensorImpl` is SonicBoom-owned (holds an
  `at::Tensor` inside the adapter); the c10 `TensorImpl`/`StorageImpl`/
  `intrusive_ptr` never appear in Layer 2.
- **`Scalar`/`Value` carry an `int` overload** in addition to `int64_t`/`double`/
  `bool`, because an `int` literal is otherwise ambiguous between the three
  (verified: `S(42)` is ill-formed without it).

### B. Layer 1 adapter (`core/layer1/adapter/` + `core/layer1/registration/`)

The only SonicBoom-owned code that uses c10/ATen headers (PRIVATE include dirs in
`core/CMakeLists.txt`). Internal header `adapter/adapter.h` defines
`detail::TensorImpl`, `detail::OperatorHandleImpl`, `detail::RegistrationState`,
the `detail::Adapter` friend, and all conversion decls.

| Source | Responsibility |
|---|---|
| `scalar_type.cpp`, `device.cpp`, `layout.cpp`, `memory_format.cpp`, `scalar.cpp` | value-type conversions ↔ c10 |
| `value.cpp` | `nt::Value ↔ c10::IValue` |
| `schema.cpp` | `nt::Argument/OperatorSchema ↔ c10::Argument/c10::FunctionSchema` |
| `argument_list.cpp` | `nt::ArgumentList ↔ c10::Stack` |
| `operator_handle.cpp` | lookup + boxed `call` |
| `tensor.cpp` | `Tensor` accessors + `empty` factory via `make_cpu_tensor` |
| `backend.cpp` | `(BackendId, Functionality) → c10::DispatchKey` |
| `registration/registration.cpp` | `define_operator` (→ `registerDef`), `register_kernel` (→ `registerImpl`) |

**Known c10 semantics recorded during implementation:**

1. **`c10::OperatorHandle` has no `defined()`** — validity is `hasSchema()`.
2. **CPU device index normalization:** `c10` reports a CPU tensor's device as
   index `-1` (meaning "current/default"), so `from_aten(Device)` maps CPU →
   index `0` to match `Device::cpu()`.
3. **Stateful kernel registration** uses
   `BoxedKernel::makeFromFunctor<NtKernel>(...)` +
   `KernelFunction::makeFromBoxedKernel(...)` (an `NtKernel : c10::OperatorKernel`
   functor bridging `nt::KernelFn` onto the boxed stack), because
   `makeFromBoxedFunction<&fn>()` cannot capture the `nt::KernelFn` pointer.

### C. Dispatch mapping

v0 is single-backend (CPU), so "highest-priority applicable backend wins" is
trivially satisfied. The adapter maps:

```text
(BackendId::cpu(), Functionality::Dense)     -> DispatchKey::CPU
(BackendId::cpu(), Functionality::Sparse)    -> DispatchKey::Sparse
(BackendId::cpu(), Functionality::Quantized) -> DispatchKey::QuantizedCPU
(BackendId::cpu(), Functionality::Autograd)  -> DispatchKey::Autograd
```

`OperatorHandle::call` dispatches to `DispatchKey::CPU` explicitly (same as the
M1 proof); generic backend selection from a tensor's key set is deferred.

### D. Build wiring

- `core/CMakeLists.txt`: static lib `sonicboom` (globbed adapter `.cpp`), PUBLIC
  include `core/include`, PRIVATE native-torch include roots, links `aten_core`
  PRIVATE (so native-torch does not leak to Layer 2 consumers). SonicBoom-owned
  code compiles at C++23 (global default); native-torch keeps its own C++17.
- Root `CMakeLists.txt`: `add_subdirectory(core)` + `add_subdirectory(tests/core)`.
- `tests/core/CMakeLists.txt`: 5 executables linking `sonicboom`.

### E. Tests (5 focused executables, `assert`-based, no framework)

| Test | Proves |
|---|---|
| `test_schema.cpp` (A) | `OperatorSchema` build → register → lookup → round-trip |
| `test_value.cpp` (B) | `Value` kind/`is*`/`to*` accessors (all variant arms) |
| `test_operator.cpp` (C) | `define_operator` + `register_kernel` + `call` boxed end-to-end (double→double) |
| `test_backend.cpp` (D) | `BackendId` value/priority/equality + highest-priority selection |
| `test_tensor.cpp` (E) | `empty` factory + `Tensor` accessors; no `TensorImpl` in scope |

All tests include only `<sonicboom/...>`; the Value↔IValue and BackendId→key
mappings are exercised end-to-end via tests C/E rather than exposed.

### F. Leakage checklist

- Layer 2 PyTorch internal leakage: **NO** (no `c10::`/`at::`/`ATen` token outside
  comments in `core/include/`; no native `#include`).
- New native-torch dependency beyond the M1 closure: **NO** (only existing
  `c10`/`aten_core`; no new migrated source).
- Python / fmt / CUDA / mobile: **NO** each.
- `dispatch_proof` (M1): **still passes**.

## 18. M2 audit (frozen)

- M2 semantic API audit completed.
- All public Layer 2 headers were inspected with their transitive include
  closure.
- Layer 2 compiler-level firewall verified.
- All 6 binaries pass: M1 `dispatch_proof` + 5 M2 tests.
- No code changes were required by the audit.
- No new dependency beyond the existing M1 `c10`/`aten_core` closure.
- No Python, fmt, CUDA, or mobile dependency.
- Layer 2 semantics are backend-independent.
- PyTorch-specific translation remains confined to Layer 1.
- `nt::OperatorHandle` was explicitly reviewed and confirmed as a genuine
  Native Torch abstraction rather than a leaked c10 handle.
- CPU device index `-1 → 0` normalization was explicitly reviewed and confirmed
  as intentional Native Torch semantics.

Two DESIGN GAPs recorded, not fixed (both non-blocking for v0):

1. `ScalarType`/`Layout`/`MemoryFormat` round-trip through `c10::IValue` is
   lossy in the adapter (c10 stores them as an `Int` IValue, so `from_aten`
   degrades them to `Value(Int)`).
2. `OperatorHandle` lifetime is coupled to registration in the current v0
   invariant (the c10 handle is a non-owning view into the dispatcher list;
   v0 never deregisters, so it is unexercised).

Final verdict: **M2 FREEZE READY**.
