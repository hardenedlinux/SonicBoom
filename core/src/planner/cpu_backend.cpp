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

#include <sonicboom/planner/backend.h>

#include <utility>

namespace sonicboom::planner {

namespace {

RuntimeError backend_error(const sx::ExecError& e) {
  return RuntimeError(RuntimeErrorCode::BackendFailure, e.message,
                      Phase::Execution);
}

} // namespace

std::expected<std::unique_ptr<CpuBackend>, RuntimeError> CpuBackend::compile(
    const sx::Document& doc,
    const std::unordered_map<std::string, sx::Bytes>& weights) {
  auto exe = sx::Executable::compile(doc, weights);
  if (!exe)
    return std::unexpected(backend_error(exe.error()));
  // Construct directly (not make_unique) so the private constructor is
  // accessible from this member.
  return std::unique_ptr<CpuBackend>(new CpuBackend(std::move(*exe)));
}

CpuBackend::CpuBackend(std::unique_ptr<sx::Executable> exe)
    : exe_(std::move(exe)) {}

std::expected<std::vector<sx::Bytes>, RuntimeError> CpuBackend::execute(
    const TaskDesc& task, const std::vector<sx::Bytes>& inputs) const {
  if (task.kind != TaskKind::Compute)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "CPU backend can only execute compute tasks", Phase::Execution));

  std::vector<sx::Bytes> outputs;
  auto r = exe_->run(inputs, outputs);
  if (!r)
    return std::unexpected(backend_error(r.error()));
  return outputs;
}

} // namespace sonicboom::planner
