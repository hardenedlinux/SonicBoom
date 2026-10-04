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

// Cost model tests (P1): valid estimates, rejection of unknown/unsupported
// requests, cost-value validation (negative/NaN/Inf/zero-bandwidth), aggregate
// summarization, and determinism.

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/resource.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace pl = sonicboom::planner;
using sonicboom::sx::DType;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}
} // namespace

int main() {
  auto snap = pl::CpuResourceProvider::snapshot({.host_memory_budget_bytes = 1 << 20});
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel model(*snap);

  // --- valid compute cost ---------------------------------------------------
  auto cc = model.estimate_compute({.op = pl::OpKind::Conv,
                                    .input_bytes = 1000,
                                    .output_bytes = 500,
                                    .dtype = DType::Float32},
                                   pl::DeviceId{0});
  check(cc.has_value() && cc->valid(), "valid conv compute cost");
  check(cc->estimated_latency_us > 0.0, "conv cost > 0");
  check(cc->confidence >= 0.0 && cc->confidence <= 1.0, "confidence in range");

  // determinism
  auto cc2 = model.estimate_compute({.op = pl::OpKind::Conv,
                                     .input_bytes = 1000,
                                     .output_bytes = 500,
                                     .dtype = DType::Float32},
                                    pl::DeviceId{0});
  check(cc.has_value() && cc2.has_value() &&
            cc->estimated_latency_us == cc2->estimated_latency_us,
        "compute cost deterministic");

  // conv cost > elementwise cost for the same bytes
  auto relu = model.estimate_compute({.op = pl::OpKind::Relu,
                                      .input_bytes = 1000,
                                      .output_bytes = 500,
                                      .dtype = DType::Float32},
                                     pl::DeviceId{0});
  check(relu.has_value() && cc->estimated_latency_us > relu->estimated_latency_us,
        "conv cost > relu cost");

  // --- unsupported dtype -> rejected ---------------------------------------
  auto bad_dtype = model.estimate_compute(
      {.op = pl::OpKind::Add, .input_bytes = 8, .output_bytes = 8, .dtype = DType::Int64},
      pl::DeviceId{0});
  check(!bad_dtype.has_value(), "unsupported dtype rejected");

  // --- unknown device -> rejected ------------------------------------------
  auto unknown_dev = model.estimate_compute(
      {.op = pl::OpKind::Add, .input_bytes = 8, .output_bytes = 8, .dtype = DType::Float32},
      pl::DeviceId{99});
  check(!unknown_dev.has_value(), "unknown device rejected");

  // --- whole-graph must be decomposed --------------------------------------
  auto wg = model.estimate_compute(
      {.op = pl::OpKind::WholeGraph, .input_bytes = 8, .output_bytes = 8,
       .dtype = DType::Float32},
      pl::DeviceId{0});
  check(!wg.has_value(), "whole-graph estimate rejected (must decompose)");

  // --- valid transfer cost --------------------------------------------------
  auto tc = model.estimate_transfer(pl::MemorySpaceId{0}, pl::MemorySpaceId{0});
  check(tc.has_value() && tc->valid(), "valid host transfer cost");
  check(tc->effective_bandwidth_bytes_per_us > 0.0, "transfer bandwidth > 0");

  // unknown memory space
  auto bad_tc = model.estimate_transfer(pl::MemorySpaceId{0}, pl::MemorySpaceId{7});
  check(!bad_tc.has_value(), "unknown memory space transfer rejected");

  // --- valid memory cost ----------------------------------------------------
  auto mc = model.estimate_memory({.size_bytes = 100, .alignment_bytes = 64},
                                  pl::MemorySpaceId{0});
  check(mc.has_value() && mc->valid(), "valid memory cost");
  check(mc->allocation_overhead_bytes == 28, "alignment padding 100->128 = 28");

  // --- cost value validation -----------------------------------------------
  pl::ComputeCost neg;
  neg.estimated_latency_us = -1.0;
  check(!neg.valid(), "negative compute latency invalid");

  pl::ComputeCost nan;
  nan.estimated_latency_us = std::numeric_limits<double>::quiet_NaN();
  check(!nan.valid(), "NaN compute latency invalid");

  pl::ComputeCost inf;
  inf.estimated_latency_us = std::numeric_limits<double>::infinity();
  check(!inf.valid(), "Inf compute latency invalid");

  pl::ComputeCost bad_conf;
  bad_conf.estimated_latency_us = 1.0;
  bad_conf.confidence = 2.0;
  check(!bad_conf.valid(), "confidence out of range invalid");

  pl::TransferCost zero_bw;
  zero_bw.fixed_latency_us = 1.0;
  zero_bw.effective_bandwidth_bytes_per_us = 0.0;
  check(!zero_bw.valid(), "zero bandwidth invalid");

  pl::TransferCost neg_bw;
  neg_bw.fixed_latency_us = 1.0;
  neg_bw.effective_bandwidth_bytes_per_us = -5.0;
  check(!neg_bw.valid(), "negative bandwidth invalid");

  // --- plan cost summarization ---------------------------------------------
  pl::PlanCostSummary ok_sum;
  ok_sum.estimated_compute_us = 1.0;
  ok_sum.estimated_transfer_us = 2.0;
  ok_sum.estimated_memory_us = 3.0;
  check(ok_sum.summarize() && ok_sum.estimated_total_us == 6.0,
        "plan cost sum = 6.0");

  pl::PlanCostSummary inf_sum;
  inf_sum.estimated_compute_us = std::numeric_limits<double>::infinity();
  inf_sum.estimated_transfer_us = 0.0;
  inf_sum.estimated_memory_us = 0.0;
  check(!inf_sum.summarize(), "non-finite total rejected");

  if (g_failures == 0) {
    std::cout << "test_planner_cost_model OK\n";
    return 0;
  }
  std::cerr << "test_planner_cost_model FAILED: " << g_failures << " check(s)\n";
  return 1;
}
