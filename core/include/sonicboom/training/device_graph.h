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

// A compiled training graph executed on the GPU.
//
// The graph is a Blade (S-Expr v0.1) document the Guile tracer emitted from a
// traced training step: the forward nodes and the explicit adjoint nodes
// `backward!` appended, plus the AdamW updates. It is lowered with
// `bladeToBoom`, run through the Boom IR `Training` pass phase, and then
// executed here node by node: every tensor lives in one device buffer, and
// each node dispatches one hand-written CUDA kernel (core/src/training/
// device_graph_kernels.cu) or one of the training module's existing backward
// kernels (core/src/training/backward_internal.h).
//
// Binding is by name, in `inputs()` order: the tracer names every tensor
// argument of the traced function (`arg<i>`, or `arg<i>.<k>` inside a list),
// so the caller matches the graph's declared inputs to its own values. The
// graph's parameters (literal constants, e.g. the causal mask) are uploaded
// once at compile time and are not inputs.
//
// The interface is deliberately flat: raw host byte buffers in, raw host byte
// buffers out, no CUDA type and no MLIR type crossing this header. A training
// step is therefore one `run` call: the caller binds the current parameter
// values, the targets, the moments and the step counter, and reads back the
// loss, the gradients and the updated parameters.

#include <sonicboom/sx/ir.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace sonicboom::training {

// One tensor the compiled graph accepts or produces, in binding order.
struct DeviceTensorInfo {
  std::string name;
  sx::DType dtype = sx::DType::Float32;
  std::vector<int64_t> shape;
  std::uint64_t num_bytes = 0;
};

class DeviceTrainingGraph {
public:
  // Lower, pass, and plan DOC. Fails with a diagnostic when the lowering or a
  // pass rejects the graph, or when the device cannot be used.
  static std::expected<std::unique_ptr<DeviceTrainingGraph>, std::string>
  compile(const sx::Document& doc);

  ~DeviceTrainingGraph();
  DeviceTrainingGraph(DeviceTrainingGraph&&) noexcept;
  DeviceTrainingGraph& operator=(DeviceTrainingGraph&&) noexcept;
  DeviceTrainingGraph(const DeviceTrainingGraph&) = delete;
  DeviceTrainingGraph& operator=(const DeviceTrainingGraph&) = delete;

  const std::vector<DeviceTensorInfo>& inputs() const noexcept;
  const std::vector<DeviceTensorInfo>& outputs() const noexcept;

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
  // and a step does not have to upload it again.
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

private:
  DeviceTrainingGraph();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sonicboom::training
