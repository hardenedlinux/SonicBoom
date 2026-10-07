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

#include <sonicboom/model/loader.h>
#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/graph.h>

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

private:
  // One KV layer's persistent device-resident decode cache (mirrors
  // SonicBackend::KvCache): K is RoPE'd + per-head-normed then fp16-rounded; V
  // per-head-normed then fp16-rounded. Flat [n_slots, n_heads_kv, head_dim] in
  // f32; slot for absolute position j is `j % n_slots`.
  struct KvCacheDev {
    uint64_t k_handle = 0;
    uint64_t v_handle = 0;
    uint64_t bytes = 0;  // per-buffer byte size (k and v equal)
    uint64_t n_slots = 0;
    uint64_t n_heads_kv = 0;
    uint64_t head_dim = 0;
  };

  const model::Gemma4Model& model_;
  const Graph& graph_;
  std::vector<KvCacheDev> caches_;  // index == KV layer (0 .. n_layer_kv-1)

  void ensure_caches(uint64_t n_ctx);
};

} // namespace sonicboom::planner
