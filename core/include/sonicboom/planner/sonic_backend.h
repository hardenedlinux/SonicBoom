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

// SonicBackend (Phase 6): the third execution backend, alongside CpuBackend
// (MLIR JIT) and NativeTorchBackend (native-torch dispatcher). It executes
// Sonic-routed compute tasks — one transformer node per task — by dispatching to
// the SonicBoom-owned nn::* / quant::* kernels, reading weights straight from
// the bound Gemma 4 model via the (layer, slot) attributes baked by the plan
// emitter. Weights are never graph tensors, so the planner's buffer model stays
// on host float32 activations.
//
// v0 (M2) is CPU-only: activations are host float32 (sx::Bytes) and the two
// Int64 graph inputs (token_id / pos) are decoded on entry. The M3 CUDA backend
// will add a DeviceKind::GPU variant over the same OpKind dispatch.

#include <sonicboom/model/loader.h>
#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/graph.h>

#include <memory>

namespace sonicboom::planner {

class SonicBackend : public Backend {
public:
  // Both `model` and `graph` must outlive the backend: the model owns the weight
  // spans, the graph owns the node attributes / tensor sizes the dispatch reads.
  // `n_ctx` sizes the per-KV-layer decode cache (global layers allocate `n_ctx`
  // slots, sliding-window layers a `sliding_window` ring); 0 allocates empty
  // caches — used by the CPU backend in the CUDA spine, which never sees an
  // Attention task.
  SonicBackend(const model::Gemma4Model& model, const Graph& graph,
               uint64_t n_ctx = 0);

  DeviceKind kind() const noexcept override { return DeviceKind::CPU; }

  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

  // Share another backend's KV cache (same session). The prefill graph's
  // FlashAttention nodes fill the cache that the decode graph's Attention nodes
  // then read, so the prefill backend and decode backend must reference the same
  // cache. Both backends stay bound to their own Graph (for node lookup); only
  // the per-KV-layer cache state is shared.
  void share_caches(const SonicBackend& other);

private:
  // One KV layer's persistent decode cache (K is RoPE'd + per-head-normed then
  // fp16-rounded; V is per-head-normed then fp16-rounded — the llama.cpp
  // flash-attention KV representation). Flat [n_slots, n_heads_kv, head_dim];
  // slot for absolute position j is `j % n_slots`.
  struct KvCache {
    std::vector<float> k;
    std::vector<float> v;
    uint64_t n_slots = 0;
    uint64_t n_heads_kv = 0;
    uint64_t head_dim = 0;
  };

  const model::Gemma4Model& model_;
  const Graph& graph_;
  std::shared_ptr<std::vector<KvCache>> caches_;  // index == KV layer

  void ensure_caches(uint64_t n_ctx);
};

} // namespace sonicboom::planner
