# SonicBoom — Claude Code Project Guide

SonicBoom is a **native tensor runtime**.

## Project identity

- **Layer 1 implementation:** a selected native implementation derived from
  PyTorch, referred to inside the project as **`native-torch`**. Never call it
  `pytorch`.
- **`third_party/native-torch/` is modifiable.** It is a migrated and adapted
  implementation tree, not a read-only vendor snapshot. Modification is
  permitted where required, but must remain controlled (see "Native-torch
  migration rules").
- **First-class language binding:** **Guile** (**3.0.9+**).
- **v0 shape:** an inference/runtime system with a generic interpreter
  (no code-generation architecture unless explicitly requested).

## Language and toolchain

SonicBoom's own C++ implementation targets **C++23** with **g++-13** as the
primary compiler/toolchain. Agent work should assume C++23 and g++-13 unless
explicitly instructed otherwise.

- Do not downgrade SonicBoom's language standard merely for convenience.
- Do not introduce compatibility constraints with older C++ standards unless
  explicitly required.
- The C++23 requirement applies to SonicBoom's own code only. It does **not**
  imply that the native-torch tree should be rewritten or modernized to C++23;
  native-torch remains an implementation dependency with its own upstream/build
  constraints.

## Repository structure

```text
SonicBoom/
├── bindings/guile/      official Guile binding
├── capi/                stable C ABI
├── core/                SonicBoom Layer 2 runtime + Layer 1 adapter
├── design/              design documents (S-Expr spec, native-torch migration manifest)
├── tests/               tests for SonicBoom
├── tools/               project tooling
└── third_party/native-torch/   selected native implementation (Layer 1)
```

This structure is fixed. Do not redesign it or add new top-level
architectural directories.

## Architecture (four layers)

```text
Layer 3    C ABI / Guile / AgentOS / other language bindings
Layer 2    SonicBoom stable runtime interface
Layer 1    native-torch implementation + adapter
Layer 0    CPU / CUDA / hardware
```

Dependency direction is strictly downward:

```text
Guile → C API → SonicBoom Core → native-torch → CPU / CUDA
```

Dependencies must not point upward. In particular:

- Guile **must not** depend directly on native-torch.
- The C API **must not** depend directly on native-torch.
- Layer 2 public headers **must not** expose native-torch types.

## Layer boundaries

### Layer 2 (core/include/sonicboom/)

The Layer 2 API belongs to SonicBoom and is the stable runtime interface.
Public headers under `core/include/sonicboom/` must **not** expose:

```text
at::Tensor, at::Scalar
c10::ScalarType, c10::Device, c10::Layout, c10::MemoryFormat
c10::IValue, c10::OperatorHandle, c10::KernelFunction
c10::Stack, c10::FunctionSchema, c10::DispatchKey, c10::DispatchKeySet
TensorImpl, StorageImpl
c10::intrusive_ptr, c10::SymInt, c10::SymNodeImpl
```

or any equivalent implementation-specific type. All conversion between
SonicBoom types and native-torch types lives **behind** the Layer 2 boundary.

### Layer 1 adapter (core/layer1/)

Connects SonicBoom Layer 2 abstractions to the native-torch implementation:

```text
SonicBoom Tensor        → native-torch Tensor
SonicBoom Value         → native-torch IValue
SonicBoom OperatorHandle → native-torch operator representation
```

Native-torch implementation details must not leak into Layer 2.

### C API (capi/)

The long-term stable ABI. It uses C-compatible representations, opaque handles,
and explicit status/error mechanisms. It must **not** expose C++ classes, C++
templates, STL containers, or any native-torch / ATen / c10 types. The C API is
the boundary used by language bindings such as Guile.

C++ ABI compatibility across compilers or standard libraries is **not** a
project requirement. The stable ABI target is the C API; C++ Layer 2
interfaces are source-level only.

### Guile binding (bindings/guile/)

The official first-class binding, targeting **Guile 3.0.9+**. Dependency path:

```text
Guile → SonicBoom C API → SonicBoom Core
```

The Guile binding must not reach into `core/layer1/` or
`third_party/native-torch/` directly.

### AgentOS

AgentOS is a potential downstream consumer of SonicBoom. How AgentOS will
integrate with SonicBoom is **not currently defined**; do not invent or freeze
an AgentOS integration architecture. Possible future integration may use the C
API or other interfaces, but this is intentionally undecided.

The architectural rule:

> AgentOS integration must not cause SonicBoom's Layer 2 or C API boundaries
> to depend on AgentOS internals.

Treat AgentOS as a future downstream integration target, not a current
implementation dependency.

## Layer 2 core concepts (v0)

Types:

```text
Tensor, Scalar, DType, Device, Layout, MemoryFormat, Backend, Allocator
```

Runtime concepts:

```text
Value, ArgumentList, ResultList, OperatorHandle, OperatorSchema, dispatch()
```

Runtime graph execution uses **boxed invocation**:

```text
SonicBoom OperatorHandle + SonicBoom ArgumentList
        ↓ Layer 1 adapter
        ↓ native-torch boxed invocation → dispatcher → native kernel
        ↓ Layer 1 result conversion
SonicBoom ResultList
```

The exact native-torch implementation is **not** part of the Layer 2 contract.

## Ownership and lifetime

- **Tensor ownership:** `Tensor` is an opaque Layer 2 abstraction. The Layer 2
  interface owns the abstraction and lifetime contract. The underlying
  native-torch tensor may be retained/released through Layer 1 mechanisms.
  Do not expose `TensorImpl`, `StorageImpl`, or `intrusive_ptr` through Layer 2.
- **Operator lifetime:** v0 registers operators during initialization and
  performs no normal deregistration. Operator handles may be treated as valid
  for the lifetime of the registered runtime state. Do not introduce a new
  ownership model without architectural justification.

## v0 scope

Included:

```text
Tensor, Scalar, DType, Device, Layout, MemoryFormat, Backend, Allocator
Value, ArgumentList, ResultList
OperatorHandle, OperatorSchema, dispatch()
boxed operator invocation
ExportedProgram-style runtime interpretation
```

Explicitly deferred (unless later evidence proves they are unavoidable
dependencies):

```text
Autograd, Training, SymInt / SymIntList, Generator
quantized dtypes, float8, UInt16, UInt32, UInt64
generic code generation
```

v0 is a generic interpreter. Do not introduce a code-generation architecture
unless explicitly requested.

## Error handling

- The public SonicBoom interfaces must not depend on implementation-specific
  exception semantics.
- The C API uses explicit status/error mechanisms.
- C++ exceptions may be used internally where appropriate, but exception
  behavior must not become an implicit cross-layer ABI contract.
- Do not impose unnecessary restrictions on internal C++ exception handling.

## RTTI and ABI

- Do not make RTTI part of the public Layer 2 contract.
- Do not rely on C++ ABI stability across compilers or standard libraries.
- The stable binary compatibility target is the C API.

## Build architecture (libsonicboom.so)

- `libsonicboom.so` is SonicBoom's **final published core library**. Layer 2 is
  a stable native runtime/library consumed through FFI (first Guile, possibly
  other languages later).
- The top-level `sonicboom` CMake target **must be SHARED**, producing
  `libsonicboom.so`; `libsonicboom.a` is never the final artifact.
- Internal implementation components (native-torch, MLIR, LLVM) are built as
  **static libraries** and statically linked into `libsonicboom.so`.
- Any library statically linked into `libsonicboom.so` **must be built with
  position-independent code (PIC)**.
- Do **not** use a "STATIC core + SHARED wrapper" two-layer arrangement.
- Tests link against and load `libsonicboom.so` to exercise the shared library.

## Native-torch migration rules

### native-torch is a selective, modifiable subset

`third_party/native-torch/` is **not** an immutable vendor snapshot and **not**
a wholesale copy of the PyTorch repository. It is a migrated and adapted
implementation tree. SonicBoom may modify native-torch source where required to
make the selected native implementation work as part of SonicBoom.

The intended process is:

```text
upstream PyTorch native implementation
        ↓ selective migration
native-torch
        ↓ adaptation / modification
SonicBoom
```

More concretely:

```text
~/Project/pytorch
        ↓ selective source analysis
Migration Manifest
        ├── COPY
        ├── MODIFY
        ├── ADAPT
        ├── BLACK BOX
        └── EXCLUDE
        ↓
third_party/native-torch/
```

The goal is to extract the **minimum** native implementation required by v0, so
the native-torch tree may differ substantially from upstream PyTorch.

These four things must remain conceptually distinct:

```text
Upstream PyTorch source   (source reference)
        ↓
Migration Manifest        (approved subset + modifications)
        ↓
third_party/native-torch/ (migrated + adapted implementation)
        ↓ Layer 1 adaptation
SonicBoom Core / Layer 2
```

Do not collapse these into a single "PyTorch dependency". SonicBoom owns the
resulting runtime architecture.

### Migration categories

The migration manifest will classify each source item into exactly one of:

- **COPY** — required and migrated substantially as-is.
- **MODIFY** — required but must be changed to work within SonicBoom.
- **ADAPT** — functionality required, but SonicBoom provides an explicit
  Layer 1 adapter rather than exposing the native implementation directly.
- **BLACK BOX** — remains below the implementation boundary; agents should not
  inspect or modify it unless an approved task requires doing so. "Black box"
  does **not** mean "forbidden forever".
- **EXCLUDE** — not part of SonicBoom v0.

### Modifications are controlled

`third_party/native-torch/` is a modifiable implementation area, but agents
must **not** arbitrarily modify native-torch files. Once
`design/native-torch-migration.md` exists it is authoritative for which files
may be copied, modified, kept untouched, excluded, or are required
dependencies. That manifest has not yet been established. Until it exists:

- Do not assume a PyTorch file must be migrated.
- Do not recursively inspect native-torch.
- Do not expand the migration scope.
- Do not redesign the architecture based on PyTorch internals.

### Python

Python is **not** a runtime dependency of SonicBoom v0. The Python frontend,
Python bindings, Python-only infrastructure, and Python-specific runtime
integration are outside the current v0 migration scope unless explicitly
required later.

This says nothing about whether the upstream PyTorch repository contains Python
sources — the migration manifest determines exactly which sources and
directories are relevant.

### Critical token-saving rule

Do **not** repeatedly rediscover SonicBoom's architecture from the native-torch
source tree. The intended workflow is:

```text
Architecture decision → migration manifest → explicit source set → implementation
```

NOT:

```text
Implementation → read all of PyTorch → rediscover architecture → redesign SonicBoom
```

The native-torch tree is a large implementation dependency. Do not read it
broadly when working on SonicBoom. Inspect only the specific files required by
an approved migration task.

### Missing dependency rule

When implementing against an approved migration manifest, if compilation
reveals an unlisted native-torch dependency — or an implementation requires
modifying a native-torch file not covered by the manifest — **STOP**. Do not
silently expand the migration scope. Report:

```text
missing file
missing symbol
required modification (if any)
source location
why it is required
```

The migration scope must then be explicitly amended.

## No architecture drift

Do **not**:

- replace the Layer 2 design with direct PyTorch types
- expose ATen/c10 types through public SonicBoom APIs
- make Guile depend directly on native-torch
- make the C API depend directly on native-torch
- move the stable C ABI into the Layer 1 implementation
- introduce Python as a runtime dependency
- turn SonicBoom into a PyTorch fork
- redesign the project because native-torch uses a different internal architecture
- invent an AgentOS integration architecture
- introduce unrelated dependencies or perform unrelated refactoring

If an implementation constraint conflicts with the architecture, report the
conflict instead of silently changing the architecture.

## Coding style of agent work

- Prefer small, explicit changes.
- Do not perform unrelated refactoring.
- Do not rename or reorganize existing project directories without explicit
  instruction.
- Do not add dependencies merely for convenience.
- Do not introduce abstractions not required by the current architecture.
- Keep the Layer 2 interface small and deliberate.
