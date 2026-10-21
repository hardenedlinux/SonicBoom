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

// The per-node CPU backend: one compute task, one graph node, compiled as a
// single-node Blade document and executed by the CPU Boom executor
// (sx::Executable). It sits beside CpuBackend, which compiles a whole region
// into one executable; this backend exists for operators the region partition
// keeps as their own task (Softmax — not part of the frozen seven-op S-Expr
// vocabulary that region lowering covers).
//
// It replaces the retired NativeTorchBackend: the node is no longer bridged to
// a native-torch operator through nt::, it is lowered through the same Boom IR
// and host kernels as every other CPU node, so both CPU backends share one
// kernel implementation. BackendTag::NativeTorch (and its "native_torch" IR
// spelling) is kept as the tag name for plan/IR compatibility; the backend
// behind it is this one.

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/sx/exec.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace sonicboom::planner {

class CpuKernelBackend : public Backend {
public:
  // Capture the planner Graph: it supplies the per-node operator, attributes,
  // and input/output TensorDescs (shape + dtype) the single-node document is
  // built from. Nothing is compiled here — a node's executable is built on its
  // first execution, so a graph that cannot be lowered reports that as an
  // execution error like the retired bridge did.
  static std::expected<std::unique_ptr<CpuKernelBackend>, RuntimeError> compile(
      const Graph& graph);

  DeviceKind kind() const noexcept override { return DeviceKind::CPU; }

  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

private:
  explicit CpuKernelBackend(Graph graph);
  Graph graph_;

  struct NodeProgram {
    std::unique_ptr<sx::Executable> exe;
    std::string error;  // non-empty once compilation failed
  };
  std::map<std::uint32_t, NodeProgram> programs_;
};

} // namespace sonicboom::planner
