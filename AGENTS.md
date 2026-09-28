# SonicBoom — Agent Instructions

SonicBoom is a native tensor runtime. This file is the concise, tool-agnostic
reference for coding agents. The full guidance lives in `CLAUDE.md`; the two
files must stay in agreement.

## Identity and structure

- **Layer 1 implementation:** `native-torch` (a selected native implementation
  derived from PyTorch). Never call it `pytorch`.
- **`third_party/native-torch/` is modifiable, not read-only.** It is a
  migrated and adapted implementation tree. Modification is permitted where
  required but must stay controlled (see migration rules).
- **First-class binding:** Guile (**3.0.9+**).
- Fixed layout: `bindings/guile/`, `capi/`, `core/`, `tests/`, `tools/`,
  `third_party/native-torch/`. Do not redesign it or add top-level directories.

## Language and toolchain

- SonicBoom's own code targets **C++23** with **g++-13**. Do not downgrade the
  standard for convenience or add older-standard compatibility constraints
  unless explicitly required.
- Applies to SonicBoom's own code only; do not rewrite or modernize native-torch
  to C++23.

## Architecture

```text
Layer 3    C ABI / Guile / AgentOS / bindings
Layer 2    SonicBoom stable runtime interface
Layer 1    native-torch implementation + adapter
Layer 0    CPU / CUDA / hardware
```

Dependency direction is strictly downward:

```text
Guile → C API → SonicBoom Core → native-torch → CPU / CUDA
```

Dependencies must not point upward. Guile and the C API must not depend
directly on native-torch. Layer 2 public headers must not expose native-torch
types.

## Layer boundaries

- **Layer 2** (`core/include/sonicboom/`): owned by SonicBoom. Must not expose
  `at::*`, `c10::*`, `TensorImpl`, `StorageImpl`, `intrusive_ptr`, `SymInt`,
  `DispatchKey(Set)`, or equivalent implementation types. All conversions to
  native-torch live behind this boundary.
- **Layer 1** (`core/layer1/`): adapter connecting SonicBoom types to
  native-torch types. Native-torch details must not leak into Layer 2.
- **C API** (`capi/`): the stable ABI. C-compatible, opaque handles, explicit
  status/error mechanisms. No C++ classes/templates/STL/native-torch types.
  C++ ABI compatibility is not a requirement; the C API is the ABI target.
- **Guile** (`bindings/guile/`): binds through the C API only; must not reach
  into `core/layer1/` or `third_party/native-torch/`.

## AgentOS

AgentOS is a potential downstream consumer. Integration is not yet defined —
do not invent or freeze it. Rule: AgentOS integration must not make SonicBoom's
Layer 2 or C API depend on AgentOS internals. Treat it as a future target, not
a current dependency.

## v0 scope

Included: `Tensor`, `Scalar`, `DType`, `Device`, `Layout`, `MemoryFormat`,
`Backend`, `Allocator`, `Value`, `ArgumentList`, `ResultList`, `OperatorHandle`,
`OperatorSchema`, `dispatch()`, boxed operator invocation, and
ExportedProgram-style runtime interpretation.

Deferred (do not add unless proven unavoidable): Autograd, Training,
SymInt/SymIntList, Generator, quantized dtypes, float8, UInt16/32/64, generic
code generation. v0 is a generic interpreter — no code-generation architecture
unless explicitly requested.

## Ownership and lifetime

- `Tensor` is opaque; the Layer 2 interface owns its lifetime contract. Do not
  expose `TensorImpl`/`StorageImpl`/`intrusive_ptr` through Layer 2.
- Operators are registered at init with no normal deregistration; operator
  handles are valid for the lifetime of the registered runtime state. Do not
  invent a new ownership model without justification.

## Error handling, RTTI, ABI

- Public interfaces must not depend on implementation-specific exception
  semantics. The C API uses explicit status/error mechanisms; C++ exceptions may
  be used internally where appropriate, but must not become an implicit
  cross-layer ABI contract. Do not impose unnecessary restrictions on internal
  exception handling.
- No RTTI in the public Layer 2 contract. No reliance on C++ ABI stability.

## native-torch: selective subset and migration rules

- `third_party/native-torch/` is a **selective, modifiable** migrated subset of
  upstream PyTorch — not a wholesale copy and not read-only. Its goal is the
  minimum native implementation required by v0, so it may differ substantially
  from upstream. SonicBoom owns the resulting architecture; do not collapse
  upstream PyTorch, the migration manifest, native-torch, and SonicBoom Core
  into a single "PyTorch dependency".
- The authoritative migration scope will be recorded in
  `tools/native-torch-migration.md`; it has not yet been established. Once it
  exists, it is authoritative.
- **Migration categories** (each source item gets exactly one): **COPY**
  (migrate as-is), **MODIFY** (required but must change), **ADAPT** (explicit
  Layer 1 adapter instead of direct exposure), **BLACK BOX** (below the
  boundary; don't inspect/modify unless an approved task requires it — not
  "forbidden forever"), **EXCLUDE** (outside v0).
- **Controlled modification:** do not arbitrarily modify native-torch files.
  Once the manifest exists it defines what may be copied, modified, left
  untouched, or excluded.
- **Python:** Python is **not** a runtime dependency of v0. The Python
  frontend/bindings/only infrastructure are outside v0 scope unless explicitly
  required later. Do **not** claim the upstream PyTorch repo contains no Python;
  the manifest determines which sources are relevant.
- **Token efficiency:** do not rediscover the architecture from native-torch.
  Do not read native-torch broadly. Inspect only files required by an approved
  migration task.
- **Missing dependency / modification rule:** if compilation reveals an
  unlisted native-torch dependency, or an implementation requires modifying a
  native-torch file not covered by the manifest, STOP. Report the missing file,
  missing symbol, required modification (if any), source location, and why it
  is required. Do not silently expand the migration scope.

## No architecture drift

Do not replace Layer 2 with PyTorch types, expose ATen/c10 types through public
APIs, make Guile or the C API depend on native-torch, move the C ABI into
Layer 1, introduce Python as a runtime dependency, turn SonicBoom into a
PyTorch fork, invent an AgentOS integration architecture, or introduce
unrelated dependencies/refactoring. If a constraint conflicts with the
architecture, report the conflict instead of silently changing it.

## Working style

Small, explicit changes. No unrelated refactoring, no directory renames without
instruction, no convenience dependencies, no abstractions not required by the
architecture. Keep the Layer 2 interface small and deliberate.
