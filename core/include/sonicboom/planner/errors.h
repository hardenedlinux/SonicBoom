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

// Unified error model for the static execution planner. Two error families —
// planning-time (PlannerError / PlannerErrorCode) and runtime
// (RuntimeError / RuntimeErrorCode) — each carrying a stable code, a message,
// a phase, optional structured context (node/task/tensor/device/memory-space
// IDs and byte budgets), and an optional nested cause. This is a C++ (source
// level) error model; the C ABI maps it onto its opaque-handle status model and
// never exposes these types.

#include <sonicboom/planner/ids.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sonicboom::planner {

// The stage at which an error arose. Used for telemetry and diagnostics.
enum class Phase : uint8_t {
  Resource,       // resource discovery / snapshot
  GraphAnalysis,  // graph adapter / analyzer
  Planning,       // static planning
  Memory,         // memory planning / allocation
  Transfer,       // transfer scheduling
  Validation,     // plan validation
  Execution,      // runtime execution
  Telemetry,
};
const char* phase_name(Phase p) noexcept;

enum class PlannerErrorCode : uint8_t {
  InvalidGraph,
  UnsupportedOperator,
  UnsupportedDType,
  UnsupportedShape,
  UnsupportedCapability,
  InsufficientMemory,
  CostModelError,
  InvalidPlan,
  ResourceChanged,
  ArithmeticOverflow,
  InvalidConfiguration,
  InternalError,
};
const char* planner_error_code_name(PlannerErrorCode c) noexcept;

enum class RuntimeErrorCode : uint8_t {
  InvalidPlan,
  AllocationFailure,
  BackendFailure,
  TransferFailure,
  SynchronizationFailure,
  ResourceChanged,
  Cancelled,
  InternalError,
};
const char* runtime_error_code_name(RuntimeErrorCode c) noexcept;

// Optional structured context shared by both error families.
struct ErrorContext {
  std::optional<GraphNodeId> node;
  std::optional<TaskId> task;
  std::optional<TensorId> tensor;
  std::optional<DeviceId> device;
  std::optional<MemorySpaceId> memory_space;
  std::optional<uint64_t> required_bytes;
  std::optional<uint64_t> available_bytes;
};

struct PlannerError {
  PlannerErrorCode code;
  std::string message;
  Phase phase = Phase::Planning;
  ErrorContext context;
  std::shared_ptr<PlannerError> cause;  // optional nested cause

  PlannerError() = default;
  PlannerError(PlannerErrorCode c, std::string m, Phase p = Phase::Planning)
      : code(c), message(std::move(m)), phase(p) {}
};

struct RuntimeError {
  RuntimeErrorCode code;
  std::string message;
  Phase phase = Phase::Execution;
  ErrorContext context;
  std::shared_ptr<RuntimeError> cause;  // optional nested cause

  RuntimeError() = default;
  RuntimeError(RuntimeErrorCode c, std::string m, Phase p = Phase::Execution)
      : code(c), message(std::move(m)), phase(p) {}
};

} // namespace sonicboom::planner
