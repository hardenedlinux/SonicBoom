// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

// SonicCudaBackend (Phase 6 M3): the device-resident sibling of SonicBackend.
// It executes Sonic-routed compute tasks whose activations are already resident
// in device buffers (TensorValue::on_device == true) by dispatching to the
// SonicBoom-owned *_dev kernels in core/src/nn/cuda_resident.h and
// core/src/quant/cuda_resident.h. Outputs are allocated from the internal device
// pool (core/src/planner/device_pool.h) and returned as device TensorValues, so
// activations stay on the device between tasks; the executor emits explicit
// H2D/D2H Transfer tasks only at the host/device boundary.
//
// Like SonicBackend it reads weights straight from the bound model via the
// (layer, slot) attributes; weights are never graph tensors. Embedding and
// Softcap are placed on the CPU SonicBackend (there is no *_dev kernel for
// them), so this backend never sees them and returns a hard error if it does.

#include <sonicboom/hal/buffer.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/graph.h>

#include <memory>
#include <span>

namespace sonicboom::planner {

class SonicCudaBackend : public Backend {
public:
  // Both `model` and `graph` must outlive the backend. `n_ctx` sizes the
  // per-KV-layer decode cache (see SonicBackend).
  SonicCudaBackend(const model::Gemma4Model& model, const Graph& graph,
                   uint64_t n_ctx = 0);

  ~SonicCudaBackend() override;

  DeviceKind kind() const noexcept override { return DeviceKind::GPU; }

  // Non-const: the backend holds per-session state (the device-resident KV
  // cache) that an Attention/AttentionShared task appends to / reads across
  // decode steps.
  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

  // Seed the device-resident token-id buffer (Phase 3.2 B4): uploads `token` as
  // a float index so the first decode step's Embedding gather reads it from the
  // device. The decode loop calls this once before stepping; thereafter Argmax
  // overwrites the buffer in place and no host round-trip carries the token id.
  bool seed_token(uint64_t token);

  // Stage the current decode position into the device-resident `pos_buf_` slot
  // (uint64_t). The decode loop calls this once per step before execute(); the
  // RoPE, KV-append and attention kernels read it in-kernel, so the position is
  // an input to a capturable graph rather than a per-kernel launch argument.
  // The prefill path stages the chunk's absolute start position the same way.
  bool set_pos(uint64_t pos);

  // Stage a chunk's token ids into the persistent device-resident `tokens_buf_`
  // slot (n <= sliding_window, [n] uint64). The batched embedding gather reads
  // them in-kernel, so the chunk's ids are a graph input rather than a host
  // value baked into a synchronous H2D memcpy at capture time (CUDA Graphs for
  // chunked prefill).
  bool set_prefill_tokens(std::span<const uint64_t> tokens);

  // Download the current token id from the persistent device-resident `token_buf_`
  // slot (one float) into `host`. Used after a CUDA Graph replay to recover the
  // argmax result without the executor's host-output path (which is a synchronous
  // cudaMemcpy and therefore not capturable). The synchronous D2H also serves as
  // the replay's completion barrier.
  bool read_token(float* host) const;

  // Block until all device work completes (cudaDeviceSynchronize-equivalent via
  // the device pool). Used to drain the warm-up step before capture so the
  // recorded kernels never race the warm kernels on their shared scratch.
  bool synchronize() const;

  // HAL command seam: pre-record one decode step by running `step_fn` (which
  // calls execute() for each task of the step) under CUDA Graph capture on a
  // private stream, and return a re-playable program. Returns nullptr when the
  // graph is disabled, capture fails, or the step body reports failure — the
  // caller falls back to per-step execute(). The CUDA stream / graph machinery
  // stays behind this method; the planner and executor never see it.
  std::unique_ptr<hal::StepProgram> record_step(
      std::function<bool()> step_fn) override;

  // Share another backend's KV cache (same session). The prefill graph's
  // FlashAttention nodes fill the cache that the decode graph's Attention nodes
  // then read, so the prefill backend and decode backend must reference the same
  // cache. Both backends stay bound to their own Graph (for node lookup); only
  // the per-KV-layer device cache is shared. The last owner frees the device
  // buffers (the shared_ptr holds a device-buffer-freeing deleter).
  void share_caches(const SonicCudaBackend& other);

private:
  // One KV layer's persistent device-resident decode cache (mirrors
  // SonicBackend::KvCache): K is RoPE'd + per-head-normed then fp16-rounded; V
  // per-head-normed then fp16-rounded. Flat [n_slots, n_heads_kv, head_dim] in
  // f32; slot for absolute position j is `j % n_slots`. Allocated as hal::Buffer
  // through the device allocator (hal::DeviceAllocator) so the persistent slots
  // have a stable address for the pre-recorded step's replay.
  struct KvCacheDev {
    hal::Buffer k_buf;
    hal::Buffer v_buf;
    uint64_t bytes = 0;  // per-buffer byte size (k and v equal)
    uint64_t n_slots = 0;
    uint64_t n_heads_kv = 0;
    uint64_t head_dim = 0;
  };

  const model::Gemma4Model& model_;
  const Graph& graph_;
  // index == KV layer (0 .. n_layer_kv-1). Shared across the decode backend and
  // a prefill backend bound to a different graph (see share_caches).
  std::shared_ptr<std::vector<KvCacheDev>> caches_;
  hal::Buffer token_buf_;           // persistent device buffer: current token id (float)
  hal::Buffer pos_buf_;             // persistent device buffer: current position (uint64_t)
  hal::Buffer tokens_buf_;          // persistent device buffer: prefill chunk token ids ([chunk] uint64)
  uint64_t n_ctx_ = 0;              // decode length (global layers' key window)

  void ensure_caches(uint64_t n_ctx);
};

} // namespace sonicboom::planner
