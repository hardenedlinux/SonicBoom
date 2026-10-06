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

// Unified C++ execution entry: submit an S-Expr v0.1 graph, get results.
//
// `Model` composes the stages a caller previously had to wire by hand —
// adapt_graph → resource snapshot → static planning → region partitioning →
// backend compilation → RuntimeExecutor — behind one facade. It does not
// redesign the IR, the lowering, or the routing seam: it reuses the existing
// pipeline verbatim (see the planner algorithm and co-execution audit design
// documents), so a model that runs through `Model` runs through exactly the
// same path as the manual integration tests.
//
// This is a Layer 2 C++ (source-level) interface. It exposes no native-torch,
// MLIR, or ATen/c10 type; the C ABI and the Guile binding sit above it.

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/execution_plan.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/sx/exec.h>
#include <sonicboom/sx/ir.h>

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sonicboom {

// A unified runtime failure: which stage failed and a human-readable message.
// The underlying sx::Error / PlannerError / RuntimeError is collapsed here so a
// caller has one error type to handle; `stage` preserves the origin.
struct ModelError {
  std::string stage;    // parse | resource | plan | weight | compile | execute
  std::string message;
};

// A compiled, runnable S-Expr model. `compile` runs the full planning pipeline
// and compiles every backend region; `run` executes the plan over raw buffers.
class Model {
public:
  // Compile a parsed document. `base_dir` resolves relative external-weight
  // sidecars ("" = the process working directory). Uses the v0 CPU resource
  // snapshot (auto host budget) and the analytical CPU cost model.
  static std::expected<Model, ModelError> compile(const sx::Document& doc,
                                                  std::string base_dir = "");

  // Convenience overload: parse `text` (S-Expr v0.1 source), then compile.
  static std::expected<Model, ModelError> compile(std::string_view text,
                                                  std::string base_dir = "");

  // Load a previously exported model from `dir` (reads `<dir>/<graph_file>`
  // plus its external weight sidecars, resolved relative to `dir`) and compile
  // it. The inverse of `export_model`.
  static std::expected<Model, ModelError> load(const std::string& dir,
                                               const std::string& graph_file = "model.sx");

  Model(Model&&) noexcept = default;
  Model& operator=(Model&&) noexcept = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model() = default;

  // Run the graph over one raw little-endian buffer per graph input (in graph
  // input order). Returns one buffer per graph output (in graph output order).
  // The buffers are bound against the planned tensor byte sizes; a mismatch or
  // backend failure is reported as a ModelError, never a crash.
  std::expected<std::vector<sx::Bytes>, ModelError> run(
      const std::vector<sx::Bytes>& inputs);

  // The graph's input / output tensor types, in graph order (the binding
  // contract a caller uses to shape its buffers).
  const std::vector<sx::TensorType>& input_types() const noexcept {
    return input_types_;
  }
  const std::vector<sx::TensorType>& output_types() const noexcept {
    return output_types_;
  }

private:
  Model() = default;

  planner::ResourceSnapshot snapshot_;
  planner::ExecutionPlan plan_;
  std::vector<std::unique_ptr<planner::Backend>> backends_;  // owned here
  std::unique_ptr<planner::RuntimeExecutor> executor_;
  std::vector<sx::TensorType> input_types_;
  std::vector<sx::TensorType> output_types_;
};

// Persist `doc` and its trained parameters to a self-contained artifact
// directory (inverse of `Model::load`):
//   <dir>/model.sx    — the S-Expr v0.1 text; parameters present in `params`
//                       are rewritten to `:external` references into weights.bin
//   <dir>/weights.bin — the concatenated little-endian bytes of those parameters
// `params` maps a parameter name to its trained bytes (byte count validated
// against the parameter's declared shape + dtype); parameters not present keep
// their existing inline/external data as-is (e.g. integer constants stay
// `:values`). The artifact is a plain text file + a flat binary sidecar — the
// same on-disk form the ONNX importer emits, loadable by `Model::load` with no
// Guile or Python.
std::expected<void, ModelError> export_model(
    const sx::Document& doc,
    const std::unordered_map<std::string, sx::Bytes>& params,
    const std::string& dir);

} // namespace sonicboom
