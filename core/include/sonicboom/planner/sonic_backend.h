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

// SonicBackend (Phase 6): the transformer execution backend, alongside
// CpuBackend (whole-region sx::Executable). It executes Sonic-routed compute
// tasks — one transformer node per task — by dispatching to
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
#include <sonicboom/planner/gdn_state.h>
#include <sonicboom/planner/graph.h>

#include <memory>

namespace sonicboom::planner {

class NodeDump; // node_dump.h: optional per-node output dump (debug only)

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
  // Out of line: the node dump is an incomplete type here.
  ~SonicBackend() override;

  DeviceKind kind() const noexcept override { return DeviceKind::CPU; }

  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

  // The prefill segment length this execution runs with (Backend::
  // bind_segment_extent): it replaces the planned `batch` of every batched
  // node and sizes every dynamic tensor's buffer.
  void bind_segment_extent(uint64_t extent) override;

  // Share another backend's KV cache and recurrent-block state (same session).
  // The prefill graph's FlashAttention nodes fill the cache that the decode
  // graph's Attention nodes then read, and a prefill segment's gdn_scan nodes
  // leave the state that the decode steps' scans continue from, so the prefill
  // backend and decode backend must reference the same caches and states. Both
  // backends stay bound to their own Graph (for node lookup); only the
  // per-KV-layer cache and the per-sequence recurrent state are shared.
  void share_caches(const SonicBackend& other);

  // The recurrent-block state set this backend updates (indexed the way a
  // gdn_scan node's `state` attribute numbers them); null for a graph with no
  // scan. Read-only seam for diagnostics that compare a step's committed state
  // across backends.
  const std::shared_ptr<StateSet>& states() const noexcept { return states_; }

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
  // One state per recurrent block of the graph, indexed the way a gdn_scan
  // node's `state` attribute numbers them; null for a graph with no scan.
  std::shared_ptr<StateSet> states_;
  uint64_t segment_extent_ = 0;  // bound prefill segment length (0: static)
  std::unique_ptr<NodeDump> dump_;  // null unless the debug dump is on

  void ensure_caches(uint64_t n_ctx);
};

} // namespace sonicboom::planner
