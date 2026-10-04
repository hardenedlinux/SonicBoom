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

// Cost model interface and the deterministic v0 CPU estimator. Estimates are
// analytical and configurable — they are NOT real-latency predictors — and are
// kept strictly separate from measured telemetry. The model rejects a request
// it cannot estimate (unknown device/op/dtype, non-finite or negative values)
// with a CostModelError rather than returning a fictitious zero cost.
//
// Note: `estimate_compute` takes a single-op `ComputeRequest` (op + byte
// counts + dtype) rather than a full TaskDesc. The static planner decomposes a
// whole-graph compute task into per-node requests and sums the results; this
// keeps the cost model free of a dependency on the task/DAG types.

#include <sonicboom/planner/cost.h>
#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/ids.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>

namespace sonicboom::planner {

// A single-operator compute estimation request.
struct ComputeRequest {
  OpKind op = OpKind::WholeGraph;
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  sx::DType dtype = sx::DType::Float32;
};

// A single allocation estimation request.
struct MemoryRequest {
  uint64_t size_bytes = 0;
  uint64_t alignment_bytes = 1;
};

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual std::expected<ComputeCost, PlannerError> estimate_compute(
      const ComputeRequest& req, DeviceId device) const = 0;

  virtual std::expected<TransferCost, PlannerError> estimate_transfer(
      MemorySpaceId source, MemorySpaceId destination) const = 0;

  virtual std::expected<MemoryCost, PlannerError> estimate_memory(
      const MemoryRequest& req, MemorySpaceId destination) const = 0;
};

// Tunable constants for the analytical CPU estimator. All defaults are
// deterministic; latencies are in microseconds, bandwidths in bytes/us.
struct CpuCostParams {
  double per_op_base_us = 1.0;          // fixed cost of any dispatched op
  double per_output_byte_us = 1e-5;     // per output-byte compute cost
  double conv_factor = 32.0;            // conv multiplies the per-byte cost
  double gemm_factor = 16.0;            // gemm multiplies the per-byte cost
  double host_bandwidth_bytes_per_us = 10000.0;  // host copy bandwidth
  double host_transfer_fixed_us = 1.0;           // fixed host copy latency
  double allocation_fixed_us = 0.5;              // per-allocation latency
};

// Deterministic analytical CPU cost model over a fixed resource snapshot.
class AnalyticalCpuCostModel : public CostModel {
public:
  explicit AnalyticalCpuCostModel(const ResourceSnapshot& snapshot,
                                  CpuCostParams params = {})
      : snapshot_(snapshot), params_(params) {}

  std::expected<ComputeCost, PlannerError> estimate_compute(
      const ComputeRequest& req, DeviceId device) const override;
  std::expected<TransferCost, PlannerError> estimate_transfer(
      MemorySpaceId source, MemorySpaceId destination) const override;
  std::expected<MemoryCost, PlannerError> estimate_memory(
      const MemoryRequest& req, MemorySpaceId destination) const override;

private:
  ResourceSnapshot snapshot_;
  CpuCostParams params_;
};

} // namespace sonicboom::planner
