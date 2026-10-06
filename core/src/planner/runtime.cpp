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

#include <sonicboom/runtime.h>

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/native_torch_backend.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/sx/parser.h>
#include <sonicboom/sx/serialize.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace sonicboom {

namespace {

// Extract the region a compute task covers into a standalone document: the
// kept nodes are selected by their outputs, and the task's outputs become the
// region's graph outputs. This mirrors the slice used by the co-execution
// integration tests, so a model run through `Model` executes the same regions.
std::expected<sx::Document, ModelError> region_slice(const sx::Document& doc,
                                                     const planner::Graph& g,
                                                     const planner::TaskDesc& t) {
  const planner::ComputeTaskDesc* c = t.as_compute();
  if (!c)
    return std::unexpected(ModelError{"plan", "task has no compute payload"});

  std::vector<std::string> kept;
  for (planner::GraphNodeId nid : c->graph_nodes) {
    const planner::GraphNodeDesc* nd = g.find_node(nid);
    if (!nd)
      return std::unexpected(ModelError{"plan", "region node not found"});
    for (planner::TensorId o : nd->outputs) {
      const planner::TensorDesc* td = g.find_tensor(o);
      if (!td)
        return std::unexpected(ModelError{"plan", "region tensor not found"});
      kept.push_back(td->name);
    }
  }

  std::vector<std::string> outs;
  for (planner::TensorId o : t.outputs) {
    const planner::TensorDesc* td = g.find_tensor(o);
    if (!td)
      return std::unexpected(ModelError{"plan", "region output not found"});
    outs.push_back(td->name);
  }

  auto sliced = planner::slice_region(doc, kept, outs);
  if (!sliced)
    return std::unexpected(ModelError{"plan", sliced.error().message});
  return std::move(*sliced);
}

} // namespace

std::expected<Model, ModelError> Model::compile(const sx::Document& doc,
                                                std::string base_dir) {
  auto graph = planner::adapt_graph(doc);
  if (!graph)
    return std::unexpected(ModelError{"plan", graph.error().message});

  auto snapshot = planner::CpuResourceProvider::snapshot();
  if (!snapshot)
    return std::unexpected(ModelError{"resource", snapshot.error().message});

  planner::AnalyticalCpuCostModel cost(*snapshot);

  auto plan = planner::plan_execution(*graph, *snapshot, cost);
  if (!plan)
    return std::unexpected(ModelError{"plan", plan.error().message});

  auto weights = sx::load_external_weights(doc, base_dir);
  if (!weights)
    return std::unexpected(ModelError{"weight", weights.error().message});

  Model model;
  model.snapshot_ = *snapshot;
  model.plan_ = std::move(*plan);

  // Record the binding contract (input/output tensor types in graph order).
  auto to_type = [&graph](planner::TensorId id) -> std::expected<sx::TensorType, ModelError> {
    const planner::TensorDesc* td = graph->find_tensor(id);
    if (!td)
      return std::unexpected(ModelError{"plan", "graph tensor not found"});
    return sx::TensorType{td->dtype, sx::Shape{td->shape}};
  };
  for (planner::TensorId id : graph->inputs) {
    auto tt = to_type(id);
    if (!tt)
      return std::unexpected(tt.error());
    model.input_types_.push_back(*tt);
  }
  for (planner::TensorId id : graph->outputs) {
    auto tt = to_type(id);
    if (!tt)
      return std::unexpected(tt.error());
    model.output_types_.push_back(*tt);
  }

  // One NativeTorchBackend serves every native-torch node (the backend resolves
  // the node per task); every MLIR region gets its own CpuBackend compiled from
  // a slice_region sub-document (or the whole document for the homogeneous
  // path, which emits a single WholeGraph task).
  bool has_native = false;
  for (const auto& t : model.plan_.tasks)
    if (t.kind == planner::TaskKind::Compute && t.as_compute() &&
        t.as_compute()->backend == planner::BackendTag::NativeTorch)
      has_native = true;

  std::map<planner::BackendTag, planner::Backend*> tag_map;
  if (has_native) {
    auto nt = planner::NativeTorchBackend::compile(*graph);
    if (!nt)
      return std::unexpected(ModelError{"compile", nt.error().message});
    tag_map[planner::BackendTag::NativeTorch] = nt->get();
    model.backends_.push_back(std::move(*nt));
  }

  std::vector<std::pair<planner::TaskId, planner::Backend*>> mlir_task_backends;
  for (const auto& t : model.plan_.tasks) {
    if (t.kind != planner::TaskKind::Compute)
      continue;
    const planner::ComputeTaskDesc* c = t.as_compute();
    if (!c || c->backend != planner::BackendTag::Mlir)
      continue;

    const sx::Document* target = &doc;
    std::optional<sx::Document> sliced;
    if (c->op != planner::OpKind::WholeGraph) {
      auto s = region_slice(doc, *graph, t);
      if (!s)
        return std::unexpected(s.error());
      sliced = std::move(*s);
      target = &*sliced;
    }

    auto cpu = planner::CpuBackend::compile(*target, *weights);
    if (!cpu)
      return std::unexpected(ModelError{"compile", cpu.error().message});
    mlir_task_backends.emplace_back(t.id, cpu->get());
    model.backends_.push_back(std::move(*cpu));
  }

  // Wire the executor: native-torch via the tag map, each MLIR region via a
  // per-task backend (the Mlir tag alone is ambiguous across multiple regions).
  auto executor = std::make_unique<planner::RuntimeExecutor>(std::move(tag_map));
  for (auto [tid, be] : mlir_task_backends)
    executor->set_task_backend(tid, be);
  model.executor_ = std::move(executor);

  return model;
}

std::expected<Model, ModelError> Model::compile(std::string_view text,
                                                std::string base_dir) {
  auto doc = sx::parse_document(text);
  if (!doc)
    return std::unexpected(ModelError{"parse", doc.error().message});
  return compile(*doc, std::move(base_dir));
}

std::expected<Model, ModelError> Model::load(const std::string& dir,
                                             const std::string& graph_file) {
  std::filesystem::path p = std::filesystem::path(dir) / graph_file;
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return std::unexpected(
        ModelError{"load", "cannot open model file '" + p.string() + "'"});
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  if (f.bad())
    return std::unexpected(
        ModelError{"load", "failed reading model file '" + p.string() + "'"});
  return compile(text, dir);
}

std::expected<std::vector<sx::Bytes>, ModelError> Model::run(
    const std::vector<sx::Bytes>& inputs) {
  if (!executor_)
    return std::unexpected(ModelError{"execute", "model is not compiled"});
  auto r = executor_->execute(plan_, snapshot_, inputs);
  if (!r)
    return std::unexpected(ModelError{"execute", r.error().message});
  return std::move(r->outputs);
}

std::expected<void, ModelError> export_model(
    const sx::Document& doc,
    const std::unordered_map<std::string, sx::Bytes>& params,
    const std::string& dir) {
  // 1. Build the persisted document: parameters present in `params` become
  //    :external references into a single weights.bin sidecar, in declaration
  //    order; parameters not present keep their inline/external data.
  sx::Document persisted = doc;
  std::vector<sx::Bytes> sidecar;  // one buffer per externalized parameter
  uint64_t offset = 0;

  for (sx::Parameter& p : persisted.graph.parameters) {
    auto it = params.find(p.name);
    if (it == params.end())
      continue;  // constant or already-external parameter: keep as-is

    // The sidecar must be loadable: byte count must match the declared shape.
    auto expected = sx::tensor_byte_size(p.type);
    if (!expected)
      return std::unexpected(
          ModelError{"export", "parameter '" + p.name +
                                   "' has an invalid shape/dtype for export"});
    if (it->second.size() != static_cast<std::size_t>(*expected))
      return std::unexpected(
          ModelError{"export", "parameter '" + p.name + "' byte count " +
                                   std::to_string(it->second.size()) +
                                   " != declared " + std::to_string(*expected)});

    p.data = sx::ExternalData{"weights.bin",
                              static_cast<int64_t>(offset),
                              static_cast<int64_t>(it->second.size())};
    sidecar.push_back(it->second);
    offset += it->second.size();
  }

  // 2. Serialize the graph text.
  auto text = sx::serialize_document(persisted);
  if (!text)
    return std::unexpected(ModelError{"export", text.error().message});

  // 3. Write <dir>/model.sx and <dir>/weights.bin.
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec)
    return std::unexpected(
        ModelError{"export", "cannot create directory '" + dir + "': " +
                                 ec.message()});

  {
    std::ofstream sx_file(std::filesystem::path(dir) / "model.sx",
                          std::ios::binary);
    if (!sx_file)
      return std::unexpected(
          ModelError{"export", "cannot write model.sx in '" + dir + "'"});
    sx_file << *text;
    if (!sx_file)
      return std::unexpected(
          ModelError{"export", "failed writing model.sx in '" + dir + "'"});
  }

  if (!sidecar.empty()) {
    std::ofstream bin(std::filesystem::path(dir) / "weights.bin",
                      std::ios::binary);
    if (!bin)
      return std::unexpected(
          ModelError{"export", "cannot write weights.bin in '" + dir + "'"});
    for (const sx::Bytes& b : sidecar)
      bin.write(reinterpret_cast<const char*>(b.data()),
                static_cast<std::streamsize>(b.size()));
    if (!bin)
      return std::unexpected(
          ModelError{"export", "failed writing weights.bin in '" + dir + "'"});
  }

  return {};
}

} // namespace sonicboom
