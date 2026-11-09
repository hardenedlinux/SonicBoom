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

#include <sonicboom/planner/cost_model.h>

#include <cmath>
#include <string>

namespace sonicboom::planner {

// --- Cost validation (cost.h) ----------------------------------------------

namespace {
bool finite_nonneg(double v) noexcept {
  return std::isfinite(v) && v >= 0.0;
}
} // namespace

bool ComputeCost::valid() const noexcept {
  return finite_nonneg(estimated_latency_us) && std::isfinite(confidence) &&
         confidence >= 0.0 && confidence <= 1.0;
}

bool TransferCost::valid() const noexcept {
  // Bandwidth must be strictly positive (a zero bandwidth is a division-by-zero
  // latency, not a valid "instant" transfer).
  return finite_nonneg(fixed_latency_us) &&
         std::isfinite(effective_bandwidth_bytes_per_us) &&
         effective_bandwidth_bytes_per_us > 0.0;
}

bool MemoryCost::valid() const noexcept {
  return finite_nonneg(allocation_latency_us);
}

bool PlanCostSummary::summarize() noexcept {
  double total = estimated_compute_us + estimated_transfer_us +
                 estimated_memory_us;
  if (!std::isfinite(total) || total < 0.0)
    return false;
  estimated_total_us = total;
  return true;
}

// --- AnalyticalCpuCostModel -------------------------------------------------

namespace {

double op_factor(OpKind op, const CpuCostParams& p) {
  switch (op) {
    case OpKind::Conv: return p.conv_factor;
    case OpKind::Gemm: return p.gemm_factor;
    default: return 1.0;
  }
}

PlannerError cost_error(std::string msg) {
  return PlannerError(PlannerErrorCode::CostModelError, std::move(msg),
                      Phase::Planning);
}

} // namespace

std::expected<ComputeCost, PlannerError>
AnalyticalCpuCostModel::estimate_compute(const ComputeRequest& req,
                                         DeviceId device) const {
  const Device* d = snapshot_.find_device(device);
  if (!d)
    return std::unexpected(cost_error("cost model: unknown device " +
                                      std::to_string(device.value)));
  if (d->kind != DeviceKind::CPU && d->kind != DeviceKind::GPU)
    return std::unexpected(cost_error(
        "cost model: only CPU/GPU compute is estimated (device '" + d->name +
        "' is " + device_kind_name(d->kind) + ")"));
  if (req.op == OpKind::WholeGraph)
    return std::unexpected(cost_error(
        "cost model: whole-graph cost must be decomposed into per-node "
        "requests by the planner"));
  if (!d->capability.supports(req.op, req.dtype))
    return std::unexpected(cost_error(
        "cost model: device '" + d->name + "' does not support op '" +
        op_kind_name(req.op) + "'"));

  const bool gpu = (d->kind == DeviceKind::GPU);
  const double base = gpu ? params_.gpu_per_op_base_us : params_.per_op_base_us;
  const double per_byte =
      gpu ? params_.gpu_per_output_byte_us : params_.per_output_byte_us;
  const double factor = gpu ? 1.0 : op_factor(req.op, params_);

  ComputeCost c;
  c.estimated_latency_us =
      base + static_cast<double>(req.output_bytes) * per_byte * factor;
  c.confidence = 0.5;  // analytical estimate, deliberately below 1.0
  if (!c.valid())
    return std::unexpected(cost_error(
        "cost model: produced an invalid compute cost (non-finite/negative)"));
  return c;
}

std::expected<TransferCost, PlannerError>
AnalyticalCpuCostModel::estimate_transfer(MemorySpaceId source,
                                          MemorySpaceId destination) const {
  const MemorySpace* s = snapshot_.find_memory_space(source);
  const MemorySpace* t = snapshot_.find_memory_space(destination);
  if (!s || !t)
    return std::unexpected(cost_error("cost model: unknown memory space"));
  const bool s_host = s->kind == MemoryKind::Host;
  const bool t_host = t->kind == MemoryKind::Host;

  TransferCost c;
  if (s_host && t_host) {
    c.fixed_latency_us = params_.host_transfer_fixed_us;
    c.effective_bandwidth_bytes_per_us = params_.host_bandwidth_bytes_per_us;
  } else {
    // host↔device or device↔device: the device transfer estimate.
    c.fixed_latency_us = params_.device_transfer_fixed_us;
    c.effective_bandwidth_bytes_per_us = params_.device_bandwidth_bytes_per_us;
  }
  return c;
}

std::expected<MemoryCost, PlannerError>
AnalyticalCpuCostModel::estimate_memory(const MemoryRequest& req,
                                        MemorySpaceId destination) const {
  const MemorySpace* m = snapshot_.find_memory_space(destination);
  if (!m)
    return std::unexpected(cost_error("cost model: unknown memory space"));

  MemoryCost c;
  c.allocation_latency_us = params_.allocation_fixed_us;
  // Deterministic alignment padding as allocation overhead (no overflow:
  // padding < alignment).
  if (m->alignment_bytes > 1) {
    uint64_t rem = req.size_bytes % m->alignment_bytes;
    c.allocation_overhead_bytes = (rem == 0) ? 0 : (m->alignment_bytes - rem);
  } else {
    c.allocation_overhead_bytes = 0;
  }
  return c;
}

} // namespace sonicboom::planner
