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

// Execution backends. A Backend executes compute tasks on a device; the runtime
// executor never touches MLIR/LLVM/native-torch directly — it dispatches to a
// Backend. v0 provides two backends: CpuBackend, which wraps an sx::Executable
// JIT entry (one or more float32 inputs, one float32 output), and
// NativeTorchBackend, which dispatches a single native-torch node per task.

#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/runtime_value.h>
#include <sonicboom/planner/task.h>
#include <sonicboom/sx/exec.h>

#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sonicboom::planner {

class Backend {
public:
  virtual ~Backend() = default;

  // The device kind this backend executes on.
  virtual DeviceKind kind() const noexcept = 0;

  // Execute one compute task given its runtime input values (graph inputs only
  // — constants/weights are baked into the compiled unit) in graph-input order,
  // returning one value per graph output. Host backends read/write `.host`; a
  // device backend reads/writes `.device`.
  //
  // Non-const: a backend may hold per-session state (e.g. the Gemma KV cache)
  // that accumulates across the executor's repeated step executions. The executor
  // keeps one backend instance per BackendTag for the lifetime of a generate
  // session, so state persists between steps but is scoped to that session.
  virtual std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) = 0;
};

// CPU backend: compiles an sx::Document to its whole-graph JIT entry once, then
// serves compute tasks by running that entry. All sx::ExecError results are
// mapped onto RuntimeError (BackendFailure / InvalidPlan).
class CpuBackend : public Backend {
public:
  static std::expected<std::unique_ptr<CpuBackend>, RuntimeError> compile(
      const sx::Document& doc,
      const std::unordered_map<std::string, sx::Bytes>& weights);

  DeviceKind kind() const noexcept override { return DeviceKind::CPU; }

  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;

private:
  explicit CpuBackend(std::unique_ptr<sx::Executable> exe);
  std::unique_ptr<sx::Executable> exe_;
};

} // namespace sonicboom::planner
