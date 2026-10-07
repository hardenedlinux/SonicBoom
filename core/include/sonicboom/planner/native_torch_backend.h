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

// Native Torch backend: the Layer 2 `nt::` peer to CpuBackend. It executes a
// single native-torch node per compute task by converting the raw sx::Bytes
// inputs into nt::Tensor, building an nt::ArgumentList from the node's
// attributes, dispatching through nt::OperatorHandle, and converting the result
// back to sx::Bytes. It speaks only the Layer 2 `nt::` API — no c10/ATen type
// appears here, and it never reaches into core/layer1/ or third_party/.

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/graph.h>

#include <memory>

namespace sonicboom::planner {

class NativeTorchBackend : public Backend {
public:
  // Capture the planner Graph: it supplies the per-node operator, attributes,
  // and input/output TensorDescs (shape + dtype) the bridge needs, because
  // sx::Bytes is metadata-free and Backend::execute() does not carry shapes.
  static std::expected<std::unique_ptr<NativeTorchBackend>, RuntimeError> compile(
      const Graph& graph);

  DeviceKind kind() const noexcept override { return DeviceKind::CPU; }

  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

private:
  explicit NativeTorchBackend(Graph graph);
  Graph graph_;
};

} // namespace sonicboom::planner
