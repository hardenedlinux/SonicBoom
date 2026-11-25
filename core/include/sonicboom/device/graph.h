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

// A compiled graph executed on the GPU, in two modes.
//
// The graph is a Blade (S-Expr v0.1) document the Guile tracer emitted. It is
// lowered with `bladeToBoom`, run through one Boom IR pass phase, and then
// executed here node by node: every tensor lives in one device buffer, and
// each node dispatches one hand-written CUDA kernel (core/src/training/
// device_graph_kernels.cu) or one of the training module's existing backward
// kernels (core/src/training/backward_internal.h).
//
// Mode::Training is the traced training step the class was written for: the
// forward nodes, the explicit adjoint nodes `backward!` appended, and the
// AdamW updates, through the `Training` pass phase. Its behaviour is the
// reference the inference mode is held to (its tests and goldens are the
// guard).
//
// Mode::Inference is a forward-only graph (the TTS forward mode): the
// `Forward` pass phase (semantic-neutral passes only), weights held in an
// external store by name instead of inline in the document, and CUDA-graph
// capture on by default. It executes the same node vocabulary minus the
// training-only ops.
//
// Binding is by name, in `inputs()` order: the tracer names every tensor
// argument of the traced function (`arg<i>`, or `arg<i>.<k>` inside a list),
// so the caller matches the graph's declared inputs to its own values. A
// training graph's parameters (literal constants, e.g. the causal mask) are
// uploaded once at compile time and are not inputs. An inference graph's
// weights are not uploaded at compile: they are external tensors the caller
// binds by name from a weight store (see `weights()` and `bind_weight`), so
// several executables share one resident copy.
//
// The interface is deliberately flat: raw host byte buffers in, raw host byte
// buffers out, no CUDA type and no MLIR type crossing this header. A step is
// therefore one `run` call.

#include <sonicboom/sx/ir.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace sonicboom::device {

// One tensor the compiled graph accepts or produces, in binding order.
struct DeviceTensorInfo {
  std::string name;
  sx::DType dtype = sx::DType::Float32;
  std::vector<int64_t> shape;
  std::uint64_t num_bytes = 0;
};

// One weight the compiled inference graph reads from an external store. The
// bytes are held on the device by the store; the graph only needs the device
// buffer handle, supplied with `bind_weight` before the first run.
struct DeviceWeightInfo {
  // The graph's own value name (the document's `(parameter ...)` identifier);
  // diagnostics name the weight this way.
  std::string name;
  // The store entry the document's `(data :bound "NAME")` names. It is the
  // key a weight store is asked for, and may differ from `name`.
  std::string bound_name;
  sx::DType dtype = sx::DType::Float32;
  std::vector<int64_t> shape;
  std::uint64_t num_bytes = 0;
  bool bound = false;
};

class DeviceGraph {
public:
  enum class Mode { Training, Inference };

  // Lower, pass, and plan DOC. Fails with a diagnostic when the lowering or a
  // pass rejects the graph, when the device cannot be used, or (inference)
  // when a weight parameter is not one a store can supply. An inference
  // graph compiles with its weights unbound; `bind_weight` supplies each one
  // before the first run.
  static std::expected<std::unique_ptr<DeviceGraph>, std::string>
  compile(const sx::Document& doc, Mode mode = Mode::Training);

  ~DeviceGraph();
  DeviceGraph(DeviceGraph&&) noexcept;
  DeviceGraph& operator=(DeviceGraph&&) noexcept;
  DeviceGraph(const DeviceGraph&) = delete;
  DeviceGraph& operator=(const DeviceGraph&) = delete;

  Mode mode() const noexcept;

  const std::vector<DeviceTensorInfo>& inputs() const noexcept;
  const std::vector<DeviceTensorInfo>& outputs() const noexcept;

  // The inference graph's external weights, in the document's parameter order.
  // Empty for a training graph (which takes inline constants only).
  const std::vector<DeviceWeightInfo>& weights() const noexcept;

  // Supply the device buffer holding weight `index`. The buffer stays owned by
  // the caller (the weight store) and must outlive the executable; binding
  // again replaces it and invalidates any captured launch sequence. Returns
  // false with a diagnostic on an out-of-range index or a null buffer.
  bool bind_weight(std::size_t index, std::uint64_t device_buffer,
                   std::string* error);

  // True when every external weight has a buffer (trivially true for a graph
  // with none). A run with a weight unbound fails before any device work and
  // names the weight.
  bool weights_bound() const noexcept;

  // Run one step. `input_data[i]` is exactly `inputs()[i].num_bytes` bytes of
  // host memory. On success `outputs` holds one byte vector per entry of
  // `outputs()`, in that order. A failed run leaves `*outputs` untouched.
  bool run(const std::byte* const* input_data,
           const std::size_t* input_sizes,
           std::vector<std::vector<std::byte>>* outputs,
           std::string* error);

  // A resident input is one the graph updates on the device: an optimizer
  // step's parameter/moment operand, whose result is written back into the
  // same device buffer. Its value therefore survives from one run to the next,
  // and a step does not have to upload it again. Always false in inference
  // mode.
  bool input_is_resident(std::size_t index) const noexcept;

  // Run one step with the resident state left on the device. The graph's
  // resident inputs are uploaded by the first successful run (this call or a
  // preceding run()); afterwards `input_data[i]` for a resident input may be
  // null and its bytes are ignored — the device buffer already holds the state
  // the previous step computed. Every other input is uploaded on every call.
  // `input_sizes[i]` is still validated for every input.
  //
  // Only the outputs named by `download[0..num_download)` are copied back, in
  // that order: a training step downloads the loss, not the parameters it keeps
  // on the device. `outputs` receives exactly `num_download` vectors.
  bool run_resident(const std::byte* const* input_data,
                    const std::size_t* input_sizes,
                    const std::size_t* download, std::size_t num_download,
                    std::vector<std::vector<std::byte>>* outputs,
                    std::string* error);

  // CUDA-graph capture of the node launches. Training defaults it from
  // SONICBOOM_TRAIN_GRAPH (capture unless "0", and never with a gradient
  // norm, whose accumulator kernel allocates); inference defaults it on,
  // with SONICBOOM_FORWARD_GRAPH=0 as the switch. Setting it off releases a
  // captured graph; setting it on lets the next run capture.
  void set_capture(bool on) noexcept;
  bool capture_enabled() const noexcept;

  // Device memory this executable holds: activation slots (the BufferReusePass
  // assignment), bound weights (owned by the weight store, counted here), and
  // compile-time scratch. Total is their sum.
  struct MemoryStats {
    std::uint64_t activation_bytes = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t total_bytes = 0;
  };
  MemoryStats memory_stats() const noexcept;

private:
  DeviceGraph();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sonicboom::device
