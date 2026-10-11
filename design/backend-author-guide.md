# Backend author guide

Status: **accepted** (Step 3). This is the contract a third-party backend
author implements to add a new execution backend to SonicBoom. It reflects the
HAL as built in Steps 1–3; it is not the aspirational full design in
`backend-hal.md` (that document's §2–§3 describe the plugin/discovery layout
that Step 4 implements on top of what is described here).

## 1. What a backend is

A backend is a `Backend` subclass (`core/include/sonicboom/planner/backend.h`)
that executes compute tasks on one device. The runtime executor never touches a
device directly: it hands each task to the backend registered for that task's
`BackendTag` and reads back `TensorValue`s. The stable HAL surface a backend
implements is three small, device-neutral pieces:

| Surface | Where | What it abstracts |
|---|---|---|
| `Backend` | `core/include/sonicboom/planner/backend.h` | `kind()`, `execute()`, optional `record_step()` |
| `hal::Buffer` / `hal::Allocator` | `core/include/sonicboom/hal/buffer.h` | stable-address device memory + copies |
| `hal::StepProgram` | `core/include/sonicboom/hal/command.h` | a pre-recorded, re-playable step |

No CUDA / Vulkan / native-torch type appears in any of these headers — a
backend's device-specific machinery lives entirely behind `execute()` and
`record_step()`.

### The `Backend` contract

```cpp
class Backend {
  virtual DeviceKind kind() const noexcept = 0;              // CPU | GPU | Accelerator
  virtual std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) = 0;
  virtual std::unique_ptr<hal::StepProgram> record_step(
      std::function<bool()> step_fn) { return nullptr; }      // optional pre-record
};
```

- `execute()` runs one compute task. A host backend reads/writes `TensorValue::host`;
  a device backend reads/writes `TensorValue::device`. Return
  `std::unexpected(RuntimeError{...})` on any unsupported op, allocation, or
  kernel failure.
- `record_step()` is the pre-record/replay seam. A backend with launch overhead
  worth amortizing (CUDA graphs, Vulkan command buffers) captures one decode step
  and replays it; a backend without that need returns `nullptr` and the caller
  falls back to per-step `execute()`. The CPU backend's "record" keeps the step
  body and re-runs it sequentially — the same interface, no capture.

### How an op reaches a backend

`route_op(OpKind)` (`core/include/sonicboom/planner/partition.h`) is the single
routing seam: it maps an `OpKind` to a `BackendTag` (`Mlir`, `NativeTorch`,
`Sonic`). The executor holds a `std::map<BackendTag, Backend*>` and dispatches by
tag. The concrete backends today are:

| Backend | Tag | Device | Notes |
|---|---|---|---|
| `CpuBackend` | `Mlir` | CPU | MLIR JIT (7-op lowering set) |
| `NativeTorchBackend` | `NativeTorch` | CPU | native-torch boxed dispatcher |
| `SonicBackend` | `Sonic` | CPU | transformer ops (`nn::*`/`quant::*` CPU kernels) |
| `SonicCudaBackend` | `Sonic` (device) | GPU | transformer ops (`nn::cuda::*`/`quant::cuda::*` kernels) |

A task whose tag has no registered backend fails with a `RuntimeError`; the
"fall back to CPU" behavior is realized by *not routing* an op to a backend that
does not support it (the planner places `Embedding`/`Softcap` on the CPU
`SonicBackend` because the CUDA backend has no `*_dev` kernel for them).

## 2. The minimal template (Vulkan stub)

`core/src/planner/vulkan_backend.{h,cpp}` is the reference template: a backend
that registers **zero** kernels, so everything routed to it fails and the caller
falls back to CPU. It is the smallest complete `Backend`:

```cpp
class VulkanBackend : public Backend {
  DeviceKind kind() const noexcept override { return DeviceKind::Accelerator; }
  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc&, const std::vector<TensorValue>&) override {
    return std::unexpected(RuntimeError{RuntimeErrorCode::BackendFailure,
        "Vulkan backend is a stub: no kernels registered", Phase::Execution});
  }
};
```

It is gated behind `SONICBOOM_USE_VULKAN` (default **OFF**), links no Vulkan
loader, and lives under `core/src/` — not the public Layer 2 tree — because a
community backend is a *plugin*, not part of the stable runtime interface (Step 4
moves it to `libsonicboom-backend-vulkan.so`). To add a real backend, copy this
template and fill in the parts below.

## 3. Authoring checklist

1. **Claim a `DeviceKind`.** `CPU` for host backends, `GPU` for the first-party
   device backends, `Accelerator` for a third-party accelerator (the Vulkan stub
   uses `Accelerator`).
2. **Advertise capability.** A device backend lists its supported `OpKind`s in
   `CpuResourceProvider::snapshot()` (`core/src/planner/resource.cpp`, the
   `gpu.capability.supported_ops` list). The planner routes an op to the device
   only when it appears there; anything absent stays on the CPU. A real Vulkan
   backend would add its own device entry here (or, post-Step-4, report it from
   its plugin).
3. **Register your kernels.** Dispatch each supported `OpKind` in `execute()` to
   your device kernels, reading weights/attributes from the task's `GraphNodeDesc`
   (the `(layer, slot)` attributes are how the Gemma 4 backend resolves weights
   without exposing them as graph tensors). Return an error for anything you do
   not implement.
4. **Own your memory.** Allocate through a `hal::Allocator` so buffers have a
   stable address for their lifetime (the property `record_step` replay relies
   on), and return device-resident `TensorValue`s so activations stay on the
   device between tasks. Emit H2D/D2H only at the host/device boundary.
5. **Implement `record_step()`** only if your backend has per-step launch
   overhead worth amortizing; return `nullptr` otherwise.
6. **Gate the build** behind a `SONICBOOM_USE_*` option (default OFF) in the root
   `CMakeLists.txt` and re-add the source in `core/CMakeLists.txt`, exactly as the
   Vulkan stub and the CUDA backend do.

## 4. The consistency suite

A backend is judged against the **CPU reference**: the `nn::*` / `quant::*` CPU
kernels (accumulating in `double` / `ggml_float`) and the `SonicBackend`/`sx`
interpreter. The op-level differential tests are the consistency suite:

| Test | Device surface vs CPU reference | Tolerance |
|---|---|---|
| `tests/core/test_cuda_elementwise.cpp` | `nn::cuda::*` (rms_norm, gelu, mul/add/scale, cast_fp16, rope, matvec, flash attention) vs `nn::*` / `scaled_dot_product_attention` | 1e-3 relative |
| `tests/core/test_cuda_quant.cpp` | `quant::cuda::matvec_f32`/`matmul_f32` vs `quant::matvec_f32` | 1e-3 (1e-1 for large-shape gemv: fp32 non-deterministic reduction order) |
| `tests/core/test_cuda_gather.cpp` | `quant::cuda::gather_f32` vs `quant::dequantize_row_f32` | bit-exact |
| `tests/core/test_nn.cpp` | CPU reference self-consistency (no device) | exact |

These are the "CUDA must pass" gates: they run only in a `SONICBOOM_USE_CUDA=ON`
build and skip cleanly when no device is present. A new backend adds its own
differential test in the same shape — build synthetic inputs, run the device
kernel and the CPU reference, compare within a stated tolerance.

**Tolerance policy** (per the v1 rules): GPU numerics need not be bit-identical
to the CPU path. Device kernels that accumulate in fp32 against a CPU
double-accumulation reference are accepted on fp32 tolerance; kernels that are
architecturally bit-exact (gather dequantize, the single-pass rms_norm which
reproduces the CPU reduction order) are checked bit-exactly. The fused internal
q8_1 kernels (`quantize_q8_1`, `rms_norm_quantize_q8_1`, `gelu_mul_quantize_q8_1`
— T1/T3/T5) have **no public CPU q8_1 reference**, so they are validated
end-to-end by `test_generate`'s golden token stream (bit-identical to the
pre-fusion baseline) rather than differentially.

## 5. Where this is going (Step 4)

The long-term layout is a plugin per backend:

```
libsonicboom.so                  # core: CPU backend + planner/executor + HAL
libsonicboom-backend-cuda.so     # CUDA plugin (dlopen'd at runtime)
libsonicboom-backend-vulkan.so   # community Vulkan plugin
```

Step 4 moves the CUDA backend (and its `.cu` sources) out of the core into
`libsonicboom-backend-cuda.so`, adds the dlopen discovery entry point, and makes
`SONICBOOM_USE_CUDA=OFF` build a core with no CUDA reference. Until then, the
backends described here are compiled into `libsonicboom.so` directly; the
`Backend`/`hal::*` surface above is unchanged by that move — a plugin implements
the same interface and is discovered, not compiled in.
