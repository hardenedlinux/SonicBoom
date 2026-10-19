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
#include <sonicboom/planner/kv_cache.h>

#include <memory>
#include <span>

namespace sonicboom::planner {

// The verify step's widest row set: a draft length of at most 7 gives k+1 = 8
// query rows; the token slot is sized for this many argmax results.
inline constexpr int kMaxVerifyRows = 16;

class SonicCudaBackend : public Backend {
public:
  // `spec` describes the KV-cache geometry (copied into the backend) and `graph`
  // must outlive the backend. `n_ctx` sizes the per-KV-layer decode cache (see
  // SonicBackend). The compute model is bound separately via bind_model() — the
  // constructor no longer reads a Gemma4Model for cache geometry (Section C).
  SonicCudaBackend(const KvCacheSpec& spec, const Graph& graph,
                   uint64_t n_ctx = 0);

  // Bind the compute model the backend reads weights / per-layer config from.
  // Required before execute(); the pointer is stored (the model must outlive the
  // backend). Decoupled from the constructor so cache geometry (KvCacheSpec) and
  // compute (model) are independent inputs.
  void bind_model(const model::Gemma4Model& model) { model_ = &model; }

  ~SonicCudaBackend() override;

  DeviceKind kind() const noexcept override { return DeviceKind::GPU; }

  // Non-const: the backend holds per-session state (the device-resident KV
  // cache) that an Attention task appends to / reads across decode steps.
  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

  // The prefill segment length this execution runs with (Backend::
  // bind_segment_extent). On a plan lowered from a symbolic IR it replaces the
  // planned `batch` of every batched node and sizes every dynamic tensor's
  // buffer, so one plan serves every segment of its shape class.
  void bind_segment_extent(uint64_t extent) override { segment_extent_ = extent; }

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

  // Async staging of a chunk's token ids / start position onto a StepProgram's
  // own stream (ordered before its next replay). These are the host-non-blocking
  // forms of set_prefill_tokens / set_pos, used by the chunked-prefill replay
  // loop so the host enqueues staging + replay for every segment without waiting
  // on the device between segments. Returns false on failure.
  bool stage_prefill_tokens(hal::StepProgram& prog,
                            std::span<const uint64_t> tokens);
  bool stage_pos(hal::StepProgram& prog, uint64_t pos);

  // Verify step: stage the absolute position of each query row into the
  // per-row position slot. The attention dispatch hands row r a pointer to
  // element r, and the decode kernels read the position from the device slot,
  // so row r appends/attends at start_pos + r while the captured graph stays
  // position-agnostic. Decode reads element 0 only (set_pos).
  bool set_verify_pos(uint64_t start_pos, int rows);
  bool stage_verify_pos(hal::StepProgram& prog, uint64_t start_pos, int rows);

  // Download the current token id from the persistent device-resident `token_buf_`
  // slot (one float) into `host`. Used after a CUDA Graph replay to recover the
  // argmax result without the executor's host-output path (which is a synchronous
  // cudaMemcpy and therefore not capturable). The synchronous D2H also serves as
  // the replay's completion barrier.
  bool read_token(float* host) const;

  // Download the verify step's per-row argmax results (rows <= kMaxVerifyRows)
  // from the persistent token slot. The synchronous D2H is the replay barrier.
  bool read_verify_tokens(float* host, int rows) const;

  // --- decode chunking (§10) ------------------------------------------------
  //
  // k greedy decode steps are captured into one graph: step j reads the token
  // step j-1's Argmax wrote (device-side feedback, no host round-trip), reads
  // position `pos_slots_[j]`, and writes its own token to `token_slots_[j]`. The
  // host stages one carry token and k positions, replays once, then reads the k
  // tokens in one D2H. k = SONICBOOM_DECODE_CHUNK (default 4; 1 keeps the
  // single-step loop and is the ablation switch).
  uint32_t decode_chunk() const { return chunk_; }

  // Turn chunking on for a decode backend: allocate the k-position and
  // k-token rings. Only the decode runtime calls this — a prefill backend must
  // keep the single stageable position/token slots. Returns false when the
  // rings cannot be allocated (the caller keeps the single-step path).
  bool enable_decode_chunking(uint32_t k);

  // Which step of the chunk the next execute() belongs to. Set once before each
  // execute inside the capture (and the warm-up); at replay the recorded
  // pointers are already the per-step ones, so it is not consulted.
  bool set_step_index(uint32_t index);

  // Stage a chunk's carry token and k positions onto the program's stream,
  // ordered before its next replay. `positions.size()` must be the chunk size.
  bool stage_chunk(hal::StepProgram& prog, float carry,
                   std::span<const uint64_t> positions);

  // Download the chunk's `count` tokens from the ring in one D2H.
  bool read_chunk_tokens(float* host, uint32_t count) const;

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

  // Adopt a persistent session's device cache (prefix reuse): replaces this
  // backend's freshly-allocated constructor cache with the session's shared
  // cache, so a later prefill fills the same buffers the next request reads.
  void share_caches(const KvCacheSession& session);

  // Publish this backend's device cache into a session (first prefix-cache
  // prefill): the session takes shared ownership so the cache survives this
  // backend's destruction. No-op-safe when the session already owns a cache.
  void publish_cache(KvCacheSession& session) const;

  // Allocate a standalone cache set for `spec` at decode length `n_ctx`, with
  // the same geometry and freeing deleter as the backend's own cache. A session
  // that owns one of these has a cache independent of every other session on
  // the same runtime (the speculative test compares two decodes from identical
  // starting states). Null on allocation failure.
  static std::shared_ptr<std::vector<KvCacheDev>>
  allocate_cache_set(const KvCacheSpec& spec, uint64_t n_ctx);

  // The identity of the KV cache vector this backend currently shares. A
  // recorded step (record_step) bakes the cache's device pointers into the
  // captured graph, so a capture may be replayed only while the cache this
  // backend shares is the one it was recorded against; this value is how a
  // caller states that. The shared_ptr keeps the captured-against cache alive
  // for as long as the holder keeps it, so comparing the raw pointer cannot
  // alias a different, later vector.
  std::shared_ptr<void> cache_identity() const { return caches_; }

private:
  const model::Gemma4Model* model_ = nullptr;  // compute model (weights/config), bound via bind_model()
  KvCacheSpec spec_;                           // KV-cache geometry (owned copy)
  const Graph& graph_;
  // index == KV layer (0 .. n_layer_kv-1). Shared across the decode backend and
  // a prefill backend bound to a different graph (see share_caches).
  std::shared_ptr<std::vector<KvCacheDev>> caches_;
  hal::Buffer token_buf_;           // persistent device buffer: current token id (float)
  hal::Buffer pos_buf_;             // persistent device buffer: current position (uint64_t)
  hal::Buffer tokens_buf_;          // persistent device buffer: prefill chunk token ids ([chunk] uint64)
  hal::Buffer pos_ring_;            // chunk mode: k positions ([k] uint64_t)
  hal::Buffer token_ring_;          // chunk mode: k emitted tokens ([k] float)
  hal::Buffer carry_buf_;           // chunk mode: the token the chunk's first step reads
  uint32_t chunk_ = 1;              // decode steps captured per graph
  uint32_t step_index_ = 0;         // which step the next execute() is
  uint64_t n_ctx_ = 0;              // decode length (global layers' key window)
  uint64_t segment_extent_ = 0;     // bound prefill segment length (0: static)

  // The device address of the position / token a kernel reads or writes for the
  // current step: the single-step slots outside chunk mode, the step's own ring
  // slot inside it.
  uint64_t pos_slot() const;
  uint64_t token_in_slot() const;
  uint64_t token_out_slot() const;

  void ensure_caches(uint64_t n_ctx);
};

} // namespace sonicboom::planner
