// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Process-wide first-use counters for the speculative loop's diagnostic
// (SONICBOOM_SPEC_DIAG=1). A counter moves only while a request is running:
// entry/exit of run_spec_program_steps raises and lowers the depth. Load-time
// warm-up therefore contributes nothing, which is what the diagnostic checks.
//
// Internal to core/src; not part of Layer 2.
#pragma once

#include <atomic>
#include <cstdint>

namespace sonicboom::diag {

struct SpecDiagCounters {
  std::atomic<uint64_t> runtime_builds{0};
  std::atomic<uint64_t> graph_captures{0};
  std::atomic<uint64_t> jit_compiles{0};
  std::atomic<uint64_t> jit_loads{0};
};

struct SpecDiagSnapshot {
  uint64_t runtime_builds = 0;
  uint64_t graph_captures = 0;
  uint64_t jit_compiles = 0;
  uint64_t jit_loads = 0;

  bool any_moved(const SpecDiagSnapshot& prev) const {
    return runtime_builds != prev.runtime_builds ||
           graph_captures != prev.graph_captures ||
           jit_compiles != prev.jit_compiles || jit_loads != prev.jit_loads;
  }
  uint64_t d_runtime(const SpecDiagSnapshot& prev) const {
    return runtime_builds - prev.runtime_builds;
  }
  uint64_t d_captures(const SpecDiagSnapshot& prev) const {
    return graph_captures - prev.graph_captures;
  }
  uint64_t d_compiles(const SpecDiagSnapshot& prev) const {
    return jit_compiles - prev.jit_compiles;
  }
  uint64_t d_loads(const SpecDiagSnapshot& prev) const {
    return jit_loads - prev.jit_loads;
  }
};

inline SpecDiagCounters& spec_diag_counters() {
  static SpecDiagCounters c;
  return c;
}

inline std::atomic<int>& spec_diag_request_depth() {
  static std::atomic<int> d{0};
  return d;
}

inline SpecDiagSnapshot spec_diag_snapshot() {
  SpecDiagCounters& c = spec_diag_counters();
  SpecDiagSnapshot s;
  s.runtime_builds = c.runtime_builds.load(std::memory_order_relaxed);
  s.graph_captures = c.graph_captures.load(std::memory_order_relaxed);
  s.jit_compiles = c.jit_compiles.load(std::memory_order_relaxed);
  s.jit_loads = c.jit_loads.load(std::memory_order_relaxed);
  return s;
}

inline void spec_diag_bump(std::atomic<uint64_t>& counter) {
  if (spec_diag_request_depth().load(std::memory_order_relaxed) > 0)
    counter.fetch_add(1, std::memory_order_relaxed);
}

inline void spec_diag_note_runtime_build() {
  spec_diag_bump(spec_diag_counters().runtime_builds);
}
inline void spec_diag_note_graph_capture() {
  spec_diag_bump(spec_diag_counters().graph_captures);
}
inline void spec_diag_note_jit_compile() {
  spec_diag_bump(spec_diag_counters().jit_compiles);
}
inline void spec_diag_note_jit_load() {
  spec_diag_bump(spec_diag_counters().jit_loads);
}

// Raises the depth for a request's lifetime.
class SpecDiagRequestScope {
 public:
  SpecDiagRequestScope() { spec_diag_request_depth().fetch_add(1); }
  ~SpecDiagRequestScope() { spec_diag_request_depth().fetch_sub(1); }
  SpecDiagRequestScope(const SpecDiagRequestScope&) = delete;
  SpecDiagRequestScope& operator=(const SpecDiagRequestScope&) = delete;
};

} // namespace sonicboom::diag
