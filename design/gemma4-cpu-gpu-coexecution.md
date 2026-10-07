# Gemma 4 CPU/GPU co-execution — current state + optimization roadmap

Status: **recorded 2026-10-07.** The CPU/GPU co-execution path is implemented and
is the current CUDA runtime path (decode + prefill). Optimization is
**deliberately deferred** — v0 targets numerical correctness first. This document
captures (1) how the co-execution works today, (2) the measured v0 numbers, and
(3) the bottleneck and levers already identified, so future optimization work
starts here instead of re-deriving it.

## 1. What "co-execution" means here

A single decode/prefill step runs some operators on the GPU and some on the CPU,
with explicit H2D/D2H transfers between them. It is *not* a whole-graph
"everything on CPU or everything on GPU" switch at the operator level — within a
CUDA run, the split is per-operator.

```text
embed (CPU) ──H2D──► matmul / rms_norm / rope / attention / elementwise (GPU)
                                                                        │
                                                                D2H     ▼
argmax (host driver) ◄── logits ◄── Softcap (CPU) ◄──────────────────────┘
```

The split is fixed by which operators have a `*_dev` (device) kernel:

- **GPU** (`SonicCudaBackend`): every operator with a `*_dev` kernel — quantized
  matmuls, rms-norms, RoPE, attention, and elementwise.
- **CPU** (`SonicBackend`): `Embedding` and `Softcap` only — the two operators
  with no `*_dev` kernel.

## 2. How it is wired (the mechanism)

Reference: `core/src/model/gemma4_plan.cpp:653-729` (`generate_spine_cuda`).

1. The decode graph is built and planned against a CUDA-aware resource snapshot
   (`CpuResourceProvider::snapshot()` + `AnalyticalCpuCostModel`), so the planner
   assigns device-local memory spaces to GPU-capable nodes.
2. `TransferScheduler` (`core/include/sonicboom/planner/transfer_scheduler.h`)
   walks the execution order and emits explicit Transfer tasks wherever an f32
   activation is produced in one memory space and consumed in another
   (host↔device); the transfer cost is accumulated into the plan.
3. `RuntimeExecutor` registers `BackendTag::Sonic → SonicCudaBackend` by default,
   then overrides host-memory-space compute tasks back to the CPU `SonicBackend`.
4. The host driver loop encodes token id / position as i64 inputs, executes the
   plan, reads the logits, and runs greedy argmax — identical to the CPU spine.

Backend routing seam: `route_op` (`core/include/sonicboom/planner/partition.h:53`)
maps every transformer operator to `BackendTag::Sonic`; the Gemma 4 plan emitter
pins every node (including the shared `Add`) to Sonic. `Sonic` is implemented
twice — `SonicBackend` (CPU) and `SonicCudaBackend` (GPU).

Top-level dispatch (`core/src/model/exec.cpp:542-558`): `generate()` chooses
`Cuda → generate_spine_cuda` (heterogeneous), `CpuQ8K → generate_spine`
(all CPU), `CpuF32 → generate_reference` (scalar oracle).

## 3. Prefill

- CPU prefill (`generate_prefill_spine`, `gemma4_plan.cpp:539`): the batched
  prefill graph runs through the spine; full-sequence SDPA is the general CPU
  kernel (`nn::scaled_dot_product_attention`).
- CUDA prefill (`generate_prefill_spine_cuda`, `gemma4_plan.cpp:745`):
  `SonicCudaBackend::prefill` fills the **device-resident KV caches** with batched
  f32 matmul + flash attention (the specialized device carve-out); the decode
  loop then reuses those caches.

## 4. Device buffer / transfer plumbing

- `TransferScheduler` — emits Transfer tasks; cost accumulated via the cost model.
- `core/src/planner/device_pool.h` — internal device buffer pool (opaque handles,
  no CUDA type leaks into `<sonicboom/>`): `acquire/release/upload/download/
  device_copy/synchronize/clear_all`. No-CUDA builds degrade to no-ops.

## 5. Secondary per-op dispatch (bespoke block path)

`core/src/model/exec.cpp:73-121` also has per-op dispatch wrappers
(`rms_norm_d`, `gelu_fp16_d`, `mul_d`, `add_d`, `scale_d`, `rope_heads_d`,
`matvec_f32_d`, `matvec_bf16_d`) that route each op to `nn::cuda::*` or the CPU
reference via a single `use_cuda` flag. Separately, `cuda_forward_resident`
(`core/src/model/cuda_block.h`, `cuda_block.cu`) runs a whole forward with
activations resident on the device, synchronizing only at the end.

## 6. Measured v0 state (full tables in `design/benchmark.md`)

- Decode: SonicBoom CUDA spine **6.20 tok/s** vs llama.cpp **42.63** (6.9×); CPU
  spine **3.90** vs **6.54** (1.7×).
- Operator-level gemv (SonicBoom `cuda::matvec_f32_dev` vs cuBLAS): FFN shapes
  competitive (≈1.09× fp16); attention shapes 6.9–9.9× slower.
- Root cause of the prefill gap: the batched path is **per-column gemv** (one
  gemv per column, `quantized_matmul_cuda.h`) — no weight reuse, so n=32 ≈ 32×
  the n=1 cost.

## 7. Optimization roadmap (DEFERRED — after v0 correctness)

Recorded here so optimization research does not have to re-discover it.

- **Bottleneck (decode):** per-step host↔device handshake — each decode step pays
  H2D + D2H transfers and a sync at the boundaries, so per-op offload does not
  raise throughput (CUDA decode stays flat ≈6.2 tok/s).
- **Lever 1 (primary):** keep activations resident on the device (Option A =
  `cuda_forward_resident`, sync once at the end) — eliminates the per-call sync.
- **Lever 2 (prefill):** replace the per-column gemv with a fused/blocked matmul
  that reuses weights — targets the 6.9–9.9× attention-shape gap.

These are research directions, not current work. Do not start optimizing until
v0 functionality is stable.
