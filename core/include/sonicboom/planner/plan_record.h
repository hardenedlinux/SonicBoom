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

// Execution-plan record: what a plan actually did, step by step.
//
// The static plan (ExecutionPlan) describes what the runtime *should* do; this
// record describes what it *did* — the executed steps in order, each with the
// node it came from, its attributes, the tensors and buffers it touched, and the
// kernels it launched together with their launch configuration. It exists so two
// plans can be diffed exactly (the plan diff tool).
//
// Recording is opt-in and costs nothing when off: the hot path checks one
// thread-local pointer (recorder() below) and takes no other action. The record
// is a diagnostic artifact — it never feeds execution.
//
// Determinism: the serialized form carries no pointers or addresses. Device
// buffers and streams are named by a per-run ordinal assigned in first-seen
// order, which is stable because the execution order is.

#include <sonicboom/sx/ir.h>

// Deliberately no planner/ids.h or planner/partition.h: the record names
// tensors and buffers by their raw logical ids, so it needs no strongly-typed
// id aliases — and both of those headers pull in <expected>, which the CUDA
// kernel translation units that include this (through plan_hook.h) cannot parse.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::planner {

// One kernel launch. `kernel` is the launched kernel's name; `tparams` is the
// concrete template instantiation when the kernel is a template (empty
// otherwise). Grid/block/smem are the launch configuration as passed to <<<>>>;
// `stream` is the per-run ordinal of the stream it was launched on.
// A stream ordinal that names no stream. The executor runs on the host side and
// cannot see the CUDA stream, so it begins a step with this and end_step()
// adopts the stream of the step's first launch — the stream every launch in the
// step actually ran on.
inline constexpr uint32_t kStreamUnknown = 0xFFFFFFFFu;

struct LaunchRecord {
  std::string kernel;
  std::string tparams;
  uint32_t grid[3] = {1, 1, 1};
  uint32_t block[3] = {1, 1, 1};
  uint64_t smem_bytes = 0;
  uint32_t stream = kStreamUnknown;
};

// One buffer a step read or wrote. `tensor` is the graph TensorId the buffer
// carries; `buffer` is the *logical* BufferId the plan assigned (never an
// address, so the record is stable across runs and processes); `offset` and
// `size_bytes` are the byte range the tensor occupies within it. shape/dtype
// describe the tensor as the plan declares it.
struct BufferRecord {
  std::string role;  // "in" | "out" | "const" | "in_graph" | "out_graph"
  uint32_t index = 0;   // position within the step's inputs (or outputs)
  uint32_t tensor = 0;  // logical TensorId
  uint32_t buffer = 0;  // logical BufferId (0 when unassigned)
  uint64_t offset = 0;
  uint64_t size_bytes = 0;
  std::string shape;
  std::string dtype;
};

// A node attribute rendered as text (name + its value).
struct AttrRecord {
  std::string name;
  std::string value;
};

// One executed step: the plan task it came from, the operator it implemented,
// its attributes, the buffers it touched, and the kernels it launched.
struct StepRecord {
  uint32_t index = 0;   // 0-based step ordinal within the run
  uint64_t task_id = 0;
  uint32_t node_id = 0;
  std::string op;       // op_kind_name
  std::string backend;  // backend_tag_name
  std::string name;     // node name (diagnostics)
  std::vector<AttrRecord> attrs;   // sorted by name
  std::vector<BufferRecord> buffers;
  std::vector<LaunchRecord> launches;
  uint32_t stream = kStreamUnknown;  // kStreamUnknown iff the step launched nothing
  std::string weight_quant;      // quantized-weight type, "" when none
  std::string activation_quant;  // quantized-activation layout, "" when none
};

// A whole run: the model/mode it was recorded for, the plan identity, and every
// executed step in order.
struct PlanRunRecord {
  static constexpr uint32_t kSchemaVersion = 1;
  uint32_t schema_version = kSchemaVersion;

  std::string model;  // "gemma4" | "qwen2"
  std::string mode;   // "decode" | "prefill"
  uint64_t n_ctx = 0; // decode: the context length planned for
  uint64_t n = 0;     // decode: steps recorded; prefill: tokens in the segment
  uint64_t plan_id = 0;
  uint64_t model_fingerprint = 0;
  std::string backend;  // the backend tag the device steps ran on
  // The device the run was recorded on, e.g. "NVIDIA GeForce RTX 3050 sm 20 cc
  // 8.6". Not decoration: part of the record *is* device-dependent — the
  // decode-attention split-KV grid is sized from the device's SM count
  // (nn::cuda::decode_attention) — so a caller comparing a record against a
  // stored one must be able to tell that it was taken on a different GPU.
  std::string device;
  std::vector<StepRecord> steps;
};

// Collects a run's steps on the thread that executes it. Install it with
// set_thread_recorder(); every step the executor runs on that thread is then
// appended. A recorder is not thread-safe by itself; one thread runs one run.
class PlanRecorder {
 public:
  // Begin recording a step. `attrs` are copied and sorted by name.
  void begin_step(uint32_t index, uint64_t task_id, uint32_t node_id,
                  std::string_view op, std::string_view backend,
                  std::string_view name, std::span<const sx::Attribute> attrs,
                  const void* stream);
  void end_step();

  // Attach a tensor/buffer the current step touches.
  void add_buffer(BufferRecord b);
  void set_weight_quant(std::string_view q);
  void set_activation_quant(std::string_view q);

  // Record one launch. Called from the launch hook (plan_hook.h) — not from
  // application code.
  void add_launch(std::string_view kernel, std::string_view tparams,
                  unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                  unsigned by, unsigned bz, uint64_t smem_bytes,
                  const void* stream);

  // Run-level metadata; call before the run when known.
  void set_run(std::string_view model, std::string_view mode, uint64_t n_ctx,
               uint64_t n, uint64_t plan_id, uint64_t model_fingerprint,
               std::string_view backend);

  // The device this run was recorded on (set by the caller that owns the CUDA
  // context; empty when unknown). See PlanRunRecord::device.
  void set_device(std::string_view device);

  const std::vector<StepRecord>& steps() const noexcept { return steps_; }
  void clear();

  // The record assembled from the run metadata and every recorded step.
  PlanRunRecord finish() const;

 private:
  uint32_t stream_ordinal(const void* stream);

  std::vector<StepRecord> steps_;
  // Streams are the one thing a step cannot name logically — a stream is a
  // handle, not an id — so they get a per-run first-seen ordinal. Buffers need
  // no such mapping: the plan's BufferId is already a stable logical id.
  std::vector<std::pair<const void*, uint32_t>> streams_;

  // Run metadata (finish()).
  PlanRunRecord run_;
};

// Render a shape as "[d0,d1,...]"; the one formatting the record uses for both
// the buffer and the step lines.
std::string shape_text(std::span<const int64_t> shape);

// The recorder installed on this thread (null when recording is off), defined in
// plan_record.cpp. The hook and the executor test it through recorder() below;
// this is the whole cost of recording being disabled, so the accessor is inline
// and reads the thread-local directly rather than calling across the library
// boundary on every kernel launch.
namespace plan_detail {
extern thread_local PlanRecorder* tls_recorder_storage;
} // namespace plan_detail

inline PlanRecorder* recorder() noexcept {
  return plan_detail::tls_recorder_storage;
}

// Install (or clear, with null) this thread's recorder. This is the one opt-in
// seam: the executor and every launch hook test recorder() and nothing else, so
// a recorder installed here sees the whole run. A caller decides whether to
// install one — the model layer's record_* entry points always do, so a tool
// asking for a record does not also have to set an environment variable.
void set_thread_recorder(PlanRecorder* rec) noexcept;

// Deterministic text serialization. Stable ordering, no pointers or addresses.
std::string serialize_plan_run(const PlanRunRecord& run);

} // namespace sonicboom::planner
