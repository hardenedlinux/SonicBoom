# Backend HAL (Hardware Abstraction Layer) — design

Status: **proposal** (Step 0). This document defines the HAL that Steps 1–4
implement. It is the design contract, not the implementation; where a file/line
is cited it reflects the tree at commit `59ec7d0`.

## 0. Goals and principles

SonicBoom is a *general* AI framework ("a better PyTorch"); Gemma 4 is only the
first benchmark. The current CUDA path was built directly against the CUDA
runtime to make v0 correct, and now needs to be generalized so that:

1. **The core (`libsonicboom.so`) is backend-agnostic.** The planner and executor
   talk to a small HAL interface; they never see a CUDA stream, a CUDA Graph, a
   `cudaMemcpy`, or a device ordinal. A backend is a *plugin* that implements the
   HAL.
2. **First-party backends are CPU and CUDA only.** Vulkan is a community target;
   the architecture must permit it and keep the bar for a third-party backend low.
3. **ggml is not integrated** (llama.cpp is a marketing comparison target, not a
   dependency). The HAL may draw on ggml's backend interface and IREE's HAL for
   shape, but it is SonicBoom's own.
4. **Implementation precedence: MLIR > graph optimization (planner passes / pattern
   rewrites) > hand-written kernels.** The HAL is what lets the higher layers
   target a device generically; hand-written kernels remain the fallback and the
   numerical/perf reference.
5. **Layer-2 public headers and the C API never expose CUDA or native-torch
   types** (CLAUDE.md). The HAL's public surface is the new boundary: it is
   expressed in SonicBoom types only.

## 1. Current CUDA-assumption leakage inventory

### 1.1 Layer 2 public headers (`core/include/sonicboom/`)

These must stay free of CUDA/native-torch *types*. Today they are free of CUDA
*types* (no `cudaStream_t`, `__global__`, `at::Tensor`), but CUDA is hard-coded as
a *name* in the public device/backend identity and in one public backend class:

| File:line | What leaks | Severity |
|---|---|---|
| `core/include/sonicboom/device.h:27,35-37` | `enum class DeviceType { CPU, CUDA }`; `Device::cuda()` factory | Medium — hard-codes "CUDA" as *the* GPU |
| `core/include/sonicboom/backend.h:31` | `BackendId::cuda()` discriminator | Low — an opaque id, but named CUDA |
| `core/include/sonicboom/tensor.h:81` | `bool cuda_available()` | Medium — capability probe named after a vendor |
| `core/include/sonicboom/quant/quantized_matmul.h:44,53` | `MatmulBackend::Cuda` enum + `cuda_available()` | Medium |
| `core/include/sonicboom/quant/quantized_matmul_cuda.h:38` | `namespace sonicboom::quant::cuda` (whole header) | Medium — a CUDA-named public header (declarations only, no CUDA types) |
| `core/include/sonicboom/planner/sonic_cuda_backend.h:39` | `class SonicCudaBackend : public Backend` in the public tree | **High** — a vendor-named backend class in Layer 2 |
| `core/include/sonicboom/planner/sonic_backend.h:39` | `class SonicBackend` (CPU transformer backend) | Low — backend class, but CPU/generic |
| `core/include/sonicboom/model/exec.h:59-63,162,176,190` | `use_cuda` params / Cuda spine comments | Medium — CUDA baked into the model API |

Notes:

- `SonicBackend` / `SonicCudaBackend` both subclass `Backend`
  (`core/include/sonicboom/planner/backend.h:38`). `Backend` itself is clean — it
  only names `DeviceKind` and a `TensorValue` list. The leak is that the
  *concrete* CUDA backend's declaration lives in the public tree and is reached
  from `gemma4_plan.cpp` by including it directly.
- `runtime_value.h:36-62` is already HAL-shaped: `DeviceBuffer` is an opaque
  `uint64_t` handle, and `TensorValue` carries an `on_device` flag. This is the
  seed of the buffer abstraction — it just needs the handle to become a
  HAL-owned buffer id rather than "a CUDA device pointer".

### 1.2 Internal code (`core/src/`) — actual CUDA API use

| File:line | What it does | HAL replacement |
|---|---|---|
| `core/src/planner/device_pool.cpp:57,78,162` | `cudaGetDeviceCount`, `cudaMalloc`, `cudaFree` | `hal::Buffer::allocate/free` |
| `core/src/planner/device_pool.cpp:101-130` | `cudaMemcpy` H2D/D2H/D2D | `hal::Buffer::copy` / `hal::CommandBuffer::copy` |
| `core/src/planner/device_pool.cpp:141` | `cudaMemsetAsync` on the current stream | `hal::CommandBuffer::fill` |
| `core/src/planner/device_pool.cpp:151` | `cudaStreamSynchronize(0)` | `hal::Device::synchronize` |
| `core/src/planner/resource.cpp:270-298` | CUDA device enum: `gpu.name = "cuda"`, capability flags, supported-op set | `hal::Device::probe()` capability report |
| `core/src/planner/sonic_cuda_backend.cpp` | the CUDA backend: dispatch to `*_dev` kernels, seed/pos/read_token, caches | becomes the CUDA HAL plugin |
| `core/src/model/cuda_block.cpp:91-249` | single-block CUDA orchestration with raw `cudaMemcpy/cudaMalloc/cudaFree` | HAL buffers + command buffers |
| `core/src/nn/cuda_stream.h:34` | thread-local `cudaStream_t sb_current_stream()` + `StreamScope`/`CaptureScope` | `hal::Stream` / capture scope on the HAL command recorder |
| `core/src/nn/cuda_graph.h:43-100` | `GraphStep` (capture/replay) | `hal::CommandBuffer` pre-record/replay |
| `core/src/nn/cuda_elementwise.cu`, `cuda_resident.h` | `*_dev` CUDA kernels | CUDA HAL plugin kernel registry |
| `core/src/quant/cuda_quant_matmul.cu`, `cuda_resident.h` | q8_1/gemv CUDA kernels | CUDA HAL plugin kernel registry |

### 1.3 CUDA Graphs API call sites (specifically requested)

The capture/replay machinery is spread over four places today:

1. **`core/src/nn/cuda_graph.h`** — `GraphStep` holds `cudaGraphExec_t` and calls
   `cudaStreamCreate` / `cudaStreamBeginCapture(…, cudaStreamCaptureModeThreadLocal)`
   / `cudaStreamEndCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` /
   `cudaStreamSynchronize` (lines 43–100).
2. **`core/src/nn/cuda_stream.h`** — `sb_current_stream()` (thread-local
   `cudaStream_t`, line 34), `StreamScope` (sets the per-thread current stream,
   lines 42–51), and `CaptureScope` (sets a thread-local "capturing" flag that
   `device_pool::download` checks so the synchronous argmax D2H is skipped under
   capture).
3. **`core/src/model/gemma4_plan.cpp:751-762`** — the decode loop: warm one real
   step, `graph_step.begin()`, run one step under `CaptureScope` +
   `StreamScope(graph_step.stream())`, `graph_step.end()`, then either the replay
   loop (`graph_step.replay()` + `cuda_backend.read_token()`) or the per-step
   fallback loop.
4. **`core/src/planner/device_pool.cpp:112-118`** — `download()` is capture-aware
   (`if (sb_capturing()) return true;`) so the uncapturable synchronous
   `cudaMemcpy` becomes a no-op under capture and the result is read from the
   persistent `token_dev_` slot after replay.

All four must collapse into the HAL's command-recording model: a backend exposes
"record a step into a command buffer, then submit; optionally pre-record once and
replay". The executor/planner then never names a CUDA Graph or stream.

### 1.4 CMake

| File:line | What |
|---|---|
| `CMakeLists.txt:29-42` | `option(SONICBOOM_USE_CUDA)`; `enable_language(CUDA)`; `CMAKE_CUDA_ARCHITECTURES 86`; `CMAKE_CUDA_STANDARD 20`; `find_package(CUDAToolkit REQUIRED)` |
| `core/CMakeLists.txt:36-37` | `sonic_cuda_backend.cpp` excluded from the unconditional planner glob |
| `core/CMakeLists.txt:73-74` | `cuda_block.cpp` excluded from the unconditional model glob |
| `core/CMakeLists.txt:122-123` | `--whole-archive aten_core c10` (keeps static-initializer-registered kernels; see §4.2) |
| `core/CMakeLists.txt:131-143` | CUDA block: `SONICBOOM_USE_CUDA=1` define, `CUDA::cudart nvToolsExt` link, the four CUDA sources re-added |
| `tests/core/CMakeLists.txt:41-60` | CUDA differential tests gated on `SONICBOOM_USE_CUDA` |

### 1.5 Clean layers

- **`capi/`** — 0 CUDA references. Clean.
- **`bindings/guile/`** — 0 CUDA references. Clean.

## 2. HAL interface design

The HAL is a C++ interface under `core/include/sonicboom/hal/` (Layer 2, no CUDA
types). It is *not* the C ABI — it is the C++ seam between the planner/executor
and a backend. A backend plugin implements it behind the boundary.

### 2.1 Device discovery and capability

```cpp
namespace sonicboom::hal {

struct Capabilities {
  uint64_t device_memory_bytes;      // VRAM (or 0 for CPU / host memory budget)
  bool     has_int8_dot;             // dp4a / i8mm
  bool     has_fp16;                 // fp16 ALU + storage
  bool     has_bf16;
  uint32_t subgroup_size;            // 32 (NVIDIA), 64 (AMD), 1 (CPU)
  uint64_t shared_memory_bytes_per_block;
  uint32_t sm_count;                 // multiprocessors / cores
  uint64_t max_alloc_bytes;          // single-allocation cap
  bool     supports_command_replay;  // CUDA Graphs / pre-recorded command buffers
  bool     supports_async_copy;      // H2D/D2D overlap with compute
  // extensible: backends add named capabilities (key/value) without touching
  // this struct's ABI.
  std::vector<std::pair<std::string, std::string>> extra;
};

struct DeviceInfo {
  DeviceId    id;                 // opaque, assigned at discovery (not a CUDA ordinal)
  DeviceKind  kind;               // CPU | GPU | Accelerator
  std::string name;               // "cpu", "cuda:0", "vulkan:0" — informational
  Capabilities caps;
};

// Discovery entry point. The core library implements the CPU device inline and
// discovers others by dlopen-ing registered backend plugins (see §3).
std::vector<DeviceInfo> enumerate_devices();

}  // namespace sonicboom::hal
```

`DeviceId` replaces today's `nt::DeviceType`/`int32_t index` hard-coding: an
opaque id minted by the HAL, so a device is not "CUDA ordinal 0" but "device #3
as discovered from plugin libsonicboom-backend-cuda.so". The existing
`nt::Device`/`BackendId` (device.h / backend.h) become thin aliases over `DeviceId`
during the migration and are then deleted.

### 2.2 Buffer and memory spaces

```cpp
enum class MemoryKind { Host, DeviceLocal, HostVisible, Managed };

struct Buffer {
  uint64_t id = 0;             // HAL-assigned, stable for the buffer's lifetime
  uint64_t size_bytes = 0;
  DeviceId device;
  MemoryKind kind;
};

struct Allocator {
  // Per-plan arena: allocates a set of buffers once with fixed addresses so a
  // pre-recorded command buffer (replayed later) keeps referencing valid memory.
  // The caller pins (buffer_id -> size) pairs; the arena is kept alive for the
  // plan's lifetime. This generalizes device_pool's free-list, which today is
  // what makes CUDA Graph replay address-stable.
  std::expected<std::vector<Buffer>, Error> allocate(
      std::span<const std::pair<uint64_t /*logical*/, uint64_t /*bytes*/>> plan);
  void deallocate(std::span<const Buffer> bufs);
};

// Copies. Synchronous forms are the simple API; async forms go through a
// CommandBuffer (see §2.4) so they can be recorded/replayed.
bool copy(Buffer dst, Buffer src, uint64_t bytes);            // any -> any
bool copy_to_host(Buffer dst, const void* src, uint64_t bytes);
bool copy_from_host(void* dst, Buffer src, uint64_t bytes);
```

`TensorValue::DeviceBuffer` (`runtime_value.h`) becomes `hal::Buffer`; the
executor stops holding a raw `uint64_t` device pointer and holds a `hal::Buffer`
id instead. `device_pool.cpp` becomes the CPU-arena + CUDA-arena implementation of
`Allocator` (the stable-address guarantee it already provides for capture/replay
is exactly what `allocate(plan)` must expose).

### 2.3 Kernel registry

A backend registers kernels by a compound key. The key is the thing the planner
uses to decide *where* to run, and is the thing a pattern rewrite tests against.

```cpp
// Registration key: (op, dtype/quant, specialization).
struct KernelKey {
  OpKind      op;                  // RmsNorm, QuantizedMatmul, Mul, ...
  DType       input_dtype;         // Float32, ...
  QuantFormat quant;               // None, Q3_K, Q4_K, Q5_K, Q8_1 (activation) ...
  uint64_t    specialization;      // bitmask: fused-flags, launch-config variant, ...
};

struct KernelInfo {
  uint64_t cost_ns;               // backend-reported expected cost (for the planner)
  // A backend may advertise *extra* capability here, e.g. "this fused kernel is
  // bit-exact" vs "tolerance-ok". Pattern rewrites read this to decide whether
  // to apply a fusion on this backend.
  std::vector<std::pair<std::string, std::string>> extra;
};

class BackendPlugin {
 public:
  virtual DeviceId device() const = 0;
  // Capability probes the planner and pattern rewrites consult.
  virtual bool      supports(const KernelKey&) const = 0;
  virtual KernelInfo query(const KernelKey&) const = 0;
  // ... command recording below
};
```

This replaces two ad-hoc mechanisms today:
- `resource.cpp`'s hand-maintained `gpu.capability.supported_ops` list
  (`core/src/planner/resource.cpp:278-287`) becomes a registry populated by the
  backend plugin itself.
- `g_q8_enabled()` / `SONICBOOM_GEMV_F32` env switches become a registry query
  ("does this backend have a fused `QuantizedMatmul` with q8_1 activation?").

The requirement "pattern rewrites must be backend-aware" (Step 2) is satisfied by:
a rewrite checks `backend->supports(fused_key)` before replacing
`RmsNorm → Quantize` with a fused op, and leaves the unfused graph otherwise.

### 2.4 Command recording and submission

The core abstraction is a **command buffer** (IREE-style). A backend exposes three
levels:

```cpp
class CommandRecorder {
  // Emit a kernel launch + its buffer arguments. The kernels were registered via
  // §2.3; the recorder references them by KernelKey + a per-buffer binding.
  virtual void dispatch(const KernelKey&, std::span<const BufferBinding>) = 0;
  virtual void copy(Buffer dst, Buffer src, uint64_t bytes) = 0;
  virtual void fill(Buffer dst, uint8_t value, uint64_t bytes) = 0;
  virtual void barrier() = 0;   // ordering fence between passes
};

class CommandBuffer {
 public:
  // Finalize the recorded sequence into a submit-able buffer.
  virtual bool finalize() = 0;
  virtual bool submit() = 0;          // enqueue for execution
  virtual bool wait() = 0;            // fence: block until complete
  virtual bool valid() const = 0;
};

class BackendPlugin {
  // Record a single step (the executor's inner loop) into a fresh command buffer.
  virtual std::unique_ptr<CommandRecorder> begin_record() = 0;
  virtual std::unique_ptr<CommandBuffer>   end_record(CommandRecorder&) = 0;
  // Pre-record once, replay many times. `replay()` is submit+wait in one call
  // (or returns a fence if async). This is the general shape-bucketed replay
  // mechanism: a backend may cache one CommandBuffer per (shape, config) key.
  virtual std::expected<std::unique_ptr<CommandBuffer>, Error> pre_record(
      std::function<void(CommandRecorder&)> fn) = 0;
};
```

How each backend maps it:

| Backend | record | submit / fence | pre_record / replay |
|---|---|---|---|
| **CUDA** | append kernel launches to a `cudaStream` | `cudaStreamSynchronize` (or an event) | `cudaGraphBeginCapture` → record → `EndCapture` → `Instantiate`; replay = `cudaGraphLaunch` + sync. This is exactly today's `GraphStep`, moved behind the interface. |
| **CPU** | append `(KernelKey, buffers)` to a `std::vector` of task descriptors | run the vector in order (the existing `SonicBackend`/thread-pool path) | pre_record = keep the vector and re-execute it per step (a "replay" that re-runs the tasks — no capture, but same interface). |
| **Vulkan** | record into a `vk::CommandBuffer` | `vkQueueSubmit` + fence | pre-record the command buffer once and re-submit; dynamic shape via push-constants / descriptor updates. |

The **fence** is the key to the D2H-on-host problem the current code works around
with `CaptureScope` + `sb_capturing()`. In the HAL model the executor expresses
"the argmax result is needed on the host" as a *host-read* of a buffer; the
backend decides whether that host-read can be recorded (it can't, in a CUDA
graph) and instead leaves it as a post-replay `copy_from_host`. The
`sb_capturing()` flag disappears — the backend owns that policy inside
`pre_record`.

`core/src/nn/cuda_stream.h`'s thread-local current stream also disappears: the
recorder carries the stream/queue explicitly, so there is no ambient
"current stream" for kernels to read. (Internally the CUDA plugin may still keep
a thread-local stream for the pre-record case, but that is the plugin's private
detail.)

### 2.5 Per-backend per-op cost data

`KernelInfo::cost_ns` (and the existing `cost_model.cpp`) feed the planner's
placement and fusion decisions. The HAL does not hard-code a cost table; each
backend reports its measured cost per `KernelKey`. The CPU backend can report the
thread-pool-measured costs; the CUDA backend reports device-kernel costs (as
already measured by the decode breakdown: gemv ≈ 64%, rms ≈ 15%, quantize ≈ 7%).
The planner consumes this through the existing `CostModel` interface.

### 2.6 CPU fallback

A backend may not implement every `OpKind`. The executor's dispatch is:

```
supports(op) → dispatch to backend
!supports(op) → move tensors to host, run the CPU reference (SonicBackend / sx)
```

This already exists in spirit (`SonicCudaBackend` returns a hard error for
Embedding/Softcap and those stay on the CPU `SonicBackend`), but today the
fallback is wired by hand in `gemma4_plan.cpp` (the two-backend map at
`gemma4_plan.cpp:718-722`). The HAL makes it uniform: the executor asks the
plugin "can you run this op?", and if not, routes the task to the CPU backend
with an implicit device↔host transfer. This is the mechanism the Vulkan stub
(Step 3) uses: it supports nothing, so everything falls back to CPU.

### 2.7 Training and dynamic shape

- **Autograd**: the HAL records ops, not graphs-with-autograd. Backward ops are
  just more `OpKind`s (e.g. `MatmulBackward`, `RmsNormBackward`) registered in
  the same kernel registry; the optimizer state (moments, scales) is plain
  `Buffer`s. A backend advertises `supports(backward_key)`; the CPU backend always
  does. No new HAL surface is required — autograd becomes a client of the same
  `dispatch`/`buffer` primitives.
- **Dynamic shape**: the generic mechanism is **shape-bucketed pre-record**.
  `pre_record` takes the shape/config as a key (the `specialization` field of
  `KernelKey`, plus a buffer-size descriptor); the backend caches one
  `CommandBuffer` per distinct shape signature. On a cache miss it re-records;
  on a hit it replays. This is how CUDA Graphs generalize from "one fixed decode
  step" to "one graph per (seq-len, batch) bucket", and how Vulkan re-submits a
  recorded command buffer with updated descriptors. The existing single-step
  CUDA Graph capture becomes `pre_record` with a one-entry shape cache.

## 3. Backend runtime loading (module split)

### 3.1 Target layout

```
libsonicboom.so                       # core: CPU backend + planner/executor + HAL
libsonicboom-backend-cuda.so          # CUDA plugin (dlopen'd at runtime)
libsonicboom-backend-vulkan.so        # (community) Vulkan plugin
```

The core library discovers plugins by scanning a fixed search path
(`SONICBOOM_BACKEND_DIR`, default `${libdir}/sonicboom/backends`) for
`libsonicboom-backend-*.so`, `dlopen`s each, and calls a `sonicboom_backend_register`
entry point that returns the `BackendPlugin` factory. No plugin found = CPU-only
runtime. This lets a Guix package ship `libsonicboom.so` with no CUDA toolchain
dependency, while an optional `sonicboom-cuda` package adds the plugin.

The `SONICBOOM_USE_CUDA=OFF` build must: not `find_package(CUDAToolkit)`, not
`enable_language(CUDA)`, not reference any CUDA source or header, and produce a
`libsonicboom.so` whose `ctest` passes with the CUDA tests compiled out (the
`tests/core/CMakeLists.txt:41-60` block already gates them on the flag; Step 4
makes the *runtime* tests skip cleanly by reporting "no CUDA device" through
`enumerate_devices()`).

### 3.2 Splitting the whole-archive native-torch CUDA parts

Today `aten_core` and `c10` are whole-archived into `libsonicboom.so`
(`core/CMakeLists.txt:122-123`) for two reasons:

1. `Register{Schema,CPU,...}*.cpp` register kernels via `TORCH_LIBRARY`
   static initializers; normal linking drops them (no referenced symbol).
2. `c10/cuda/impl/CUDAGuardImpl.cpp` is static-initializer-only
   (`C10_REGISTER_GUARD_IMPL(CUDA, …)`); without whole-archive,
   `getDeviceGuardImpl(CUDA)` is unregistered and the comment ("not linked with
   support for cuda devices") bites.

Assessment for the split:

- **CUDA kernels in `aten/src/ATen/native/cuda/**` and `c10/cuda/**`** are the
  bulk that should move to the CUDA plugin. `CUDAGuardImpl.cpp` belongs there:
  it registers the CUDA device-guard impl, which is only meaningful when CUDA is
  present. Moving it out of the whole-archive core into the CUDA plugin (also
  whole-archived, but only when the plugin is built) is correct and self-contained.
- **`ExcludedKernelsStubs`** (`third_party/native-torch/CMakeLists.txt`) is the
  existing mechanism that, on CPU-only native-torch builds, replaces
  CUDA-registered kernels with stubs so the dispatcher is total. The split
  reuses it: the core's `aten_core` is built with the CUDA kernels *excluded*
  (stubbed), and the CUDA plugin carries the real `native/cuda` sources. This is
  the natural seam — it already exists upstream, so the migration does not invent
  a new boundary, it just moves which target compiles the `native/cuda` globs.
- **Risk / obstacle to watch:** if any CPU kernel in `aten_core` transitively
  references a symbol from a CUDA source that was excluded, the core link fails.
  That must be resolved per-symbol (either move the shared helper into a common
  object or keep a thin stub). This is the one place Step 4 may hit a real wall;
  if it does, the obstacle and options are recorded here and the step stops for
  review rather than force-proceeding (per the directive).

The SonicBoom-owned CUDA sources (`.cu` kernels, `sonic_cuda_backend.cpp`,
`cuda_block.cpp`, `cuda_stream.h`, `cuda_graph.h`) move wholesale into the CUDA
plugin; they are not part of the core link at all once the HAL is in place.

### 3.3 What stays whole-archived

`aten_core` + `c10` stay whole-archived *in the core* for the CPU-registered
kernels and the CPU device-guard impl. Only the CUDA `native/cuda`/`c10/cuda`
objects migrate to the plugin (also whole-archived there). The `--no-whole-archive`
scope (core/CMakeLists.txt:123) is unchanged for the core's MLIR/LLVM links.

## 4. Migration checklist

Ordered, each step independently buildable/testable:

1. **Step 0 (this doc)** — done here.
2. **Step 1 — HAL interface, behavior invariant.** Add `core/include/sonicboom/hal/*`
   (device/capability, buffer, kernel registry, command recorder/buffer). Route
   `SonicBackend` (CPU) and `SonicCudaBackend` through it. Move CUDA Graphs behind
   `pre_record`; delete `cuda_stream.h`/`cuda_graph.h` from the executor/planner
   path. Replace `device_pool` with `hal::Allocator`. Gate: ctest green, greedy
   token stream identical, 64-step tok/s within ±3%. Split into small commits.
3. **Step 2 — T1–T5 fusions on top of the HAL, as planner passes/rewrites.**
   - T1: hoist `quantize_q8_1` out of `launch_gemv_q8_1` into an explicit
     `Quantize` graph node; `gemv` consumes q8_1; CSE merges identical `Quantize`s.
   - T2: fuse `rms_sum_sq`/`rms_inv`/`rms_apply` into one kernel.
   - T3: rewrite `RmsNorm → Quantize` to a fused op (backend-aware via `supports()`).
   - T4: fuse `rms_heads_sum/_inv/_apply`; evaluate `per-head norm → rope → kv_append`.
   - T5: emit q8_1 directly from the `gelu × up` elementwise output.
   - Unfused paths retained as numeric/perf reference. Each T: own commit, 64-step
     tok/s + depth-512 decode tok/s; report divergence + top-1/top-2 logit margin.
4. **Step 3 — community-backend scaffolding.** Vulkan stub (compiles, registers,
   discovered, zero ops → full CPU fallback; no real Vulkan compute; default off);
   op-level consistency suite runnable against any backend vs the CPU reference
   (CUDA must pass); `design/backend-author-guide.md`.
5. **Step 4 — runtime loading.** CUDA backend → `libsonicboom-backend-cuda.so`,
   dlopen'd; `SONICBOOM_USE_CUDA=OFF` build with no CUDA references and clean
   `ctest` (CUDA tests skipped). Resolve the §3.2 native-torch split; stop and
   report if it hits an obstacle.

## 5. Open questions / deferred

- Whether `DeviceId` should be a stable integer or a string handle across process
  boundaries (matters only for the C ABI, not the C++ HAL — leave for the C API
  work).
- Cost-model inputs from real device probes vs. static tables (leave to Step 1's
  measurement, not decided here).
- `nt::Device` / `nt::BackendId` / `MatmulBackend::Cuda` are public today; their
  replacement by HAL types is a source-compat break to time with a minor bump.
