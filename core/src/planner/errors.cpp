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

#include <sonicboom/planner/errors.h>

namespace sonicboom::planner {

const char* phase_name(Phase p) noexcept {
  switch (p) {
    case Phase::Resource: return "resource";
    case Phase::GraphAnalysis: return "graph_analysis";
    case Phase::Planning: return "planning";
    case Phase::Memory: return "memory";
    case Phase::Transfer: return "transfer";
    case Phase::Validation: return "validation";
    case Phase::Execution: return "execution";
    case Phase::Telemetry: return "telemetry";
  }
  return "?";
}

const char* planner_error_code_name(PlannerErrorCode c) noexcept {
  switch (c) {
    case PlannerErrorCode::InvalidGraph: return "InvalidGraph";
    case PlannerErrorCode::UnsupportedOperator: return "UnsupportedOperator";
    case PlannerErrorCode::UnsupportedDType: return "UnsupportedDType";
    case PlannerErrorCode::UnsupportedShape: return "UnsupportedShape";
    case PlannerErrorCode::UnsupportedCapability: return "UnsupportedCapability";
    case PlannerErrorCode::InsufficientMemory: return "InsufficientMemory";
    case PlannerErrorCode::CostModelError: return "CostModelError";
    case PlannerErrorCode::InvalidPlan: return "InvalidPlan";
    case PlannerErrorCode::ResourceChanged: return "ResourceChanged";
    case PlannerErrorCode::ArithmeticOverflow: return "ArithmeticOverflow";
    case PlannerErrorCode::InvalidConfiguration: return "InvalidConfiguration";
    case PlannerErrorCode::InternalError: return "InternalError";
  }
  return "?";
}

const char* runtime_error_code_name(RuntimeErrorCode c) noexcept {
  switch (c) {
    case RuntimeErrorCode::InvalidPlan: return "InvalidPlan";
    case RuntimeErrorCode::AllocationFailure: return "AllocationFailure";
    case RuntimeErrorCode::BackendFailure: return "BackendFailure";
    case RuntimeErrorCode::TransferFailure: return "TransferFailure";
    case RuntimeErrorCode::SynchronizationFailure: return "SynchronizationFailure";
    case RuntimeErrorCode::ResourceChanged: return "ResourceChanged";
    case RuntimeErrorCode::Cancelled: return "Cancelled";
    case RuntimeErrorCode::InternalError: return "InternalError";
  }
  return "?";
}

} // namespace sonicboom::planner
