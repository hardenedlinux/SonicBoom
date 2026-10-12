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

#include <cstring>
#include <functional>
#include <utility>

namespace sonicboom::planner {

namespace {

RuntimeError backend_error(const sx::ExecError& e) {
  return RuntimeError(RuntimeErrorCode::BackendFailure, e.message,
                      Phase::Execution);
}

// CPU StepProgram: a recorded step is just its step body, re-executed in order
// on each replay. CPU work is synchronous, so sync() is a no-op and replay()
// returns the step body's success.
class CpuStepProgram final : public hal::StepProgram {
 public:
  explicit CpuStepProgram(std::function<bool()> step_fn)
      : step_(std::move(step_fn)) {}

  bool replay() override { return step_ ? step_() : false; }
  bool sync() override { return true; }  // replay() is already synchronous
  bool valid() const override { return static_cast<bool>(step_); }

  // The CPU backend's buffers are host memory, so staging is a plain copy.
  bool stage_upload(const hal::Buffer& dst, const void* host,
                    uint64_t bytes) override {
    if (dst.id == 0 || (!host && bytes != 0)) return false;
    if (bytes != 0) std::memcpy(reinterpret_cast<void*>(dst.id), host, bytes);
    return true;
  }

 private:
  std::function<bool()> step_;
};

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

std::expected<std::vector<TensorValue>, RuntimeError> CpuBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& inputs) {
  if (task.kind != TaskKind::Compute)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "CPU backend can only execute compute tasks", Phase::Execution));

  // The MLIR entry runs on host bytes; device values never reach the CPU path.
  std::vector<sx::Bytes> host_inputs;
  host_inputs.reserve(inputs.size());
  for (const auto& v : inputs) {
    if (v.on_device)
      return std::unexpected(RuntimeError(
          RuntimeErrorCode::InvalidPlan,
          "CPU backend received a device-resident input", Phase::Execution));
    host_inputs.push_back(v.host);
  }

  std::vector<sx::Bytes> outputs;
  auto r = exe_->run(host_inputs, outputs);
  if (!r)
    return std::unexpected(backend_error(r.error()));

  std::vector<TensorValue> result;
  result.reserve(outputs.size());
  for (auto& b : outputs)
    result.push_back(TensorValue::from_host(std::move(b)));
  return result;
}

std::unique_ptr<hal::StepProgram> CpuBackend::record_step(
    std::function<bool()> step_fn) {
  if (!step_fn) return nullptr;
  return std::make_unique<CpuStepProgram>(std::move(step_fn));
}

} // namespace sonicboom::planner
