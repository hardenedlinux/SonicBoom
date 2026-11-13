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

// A gated attention output (model/recipe.h SelfAttention::output_gate): the
// block emits the Q projection over two head widths, slices the gate half out
// of it, and multiplies the attention result by the gate's sigmoid before the
// output projection.
//
// This suite executes that region — the two Slice nodes, the Sigmoid and the
// Mul, with the attributes the emitter writes — on the CPU SonicBackend and
// checks the arithmetic against the definition:
//
//   q_i     = x[2i]          gate_i = x[2i+1]
//   y_i     = q_i * sigmoid(gate_i)
//
// over a single decode row and over the flat [n, 2*head_dim] prefill layout,
// where the same stride selects each row's halves.
//
// CPU-only; needs no model file and no device.

#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/sonic_backend.h>
#include <sonicboom/planner/task.h>
#include <sonicboom/sx/exec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace sbplanner = sonicboom::planner;
namespace sbmodel = sonicboom::model;
namespace sx = sonicboom::sx;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cout << "FAIL: " << what << "\n";
  }
}

sx::Bytes to_bytes(const std::vector<float>& v) {
  sx::Bytes b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<float> to_floats(const sx::Bytes& b) {
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), v.size() * sizeof(float));
  return v;
}

sx::Attribute attr_i(const char* name, int64_t v) { return sx::Attribute{name, v}; }

// Run one node of `g` on the CPU backend over already-computed host values.
bool run_node(sbplanner::SonicBackend& backend, sbplanner::Graph& g,
              uint32_t node_index, std::vector<std::vector<float>>& values,
              std::vector<float>& out) {
  const auto& node = g.nodes[node_index];
  sbplanner::TaskDesc task;
  task.id = sbplanner::TaskId{node_index};
  task.kind = sbplanner::TaskKind::Compute;
  task.inputs = node.inputs;
  task.outputs = node.outputs;
  sbplanner::ComputeTaskDesc compute;
  compute.op = node.op;
  compute.graph_nodes = {node.id};
  task.payload = compute;

  std::vector<sbplanner::TensorValue> in;
  for (auto t : node.inputs) in.push_back(sbplanner::TensorValue::from_host(to_bytes(values[t.value])));
  auto res = backend.execute(task, in);
  if (!res || res->empty()) {
    std::cout << "  execute failed at node " << node.name
              << (res ? ": no output" : ": " + res.error().message) << "\n";
    return false;
  }
  out = to_floats((*res)[0].host);
  values[node.outputs[0].value] = out;
  return true;
}

// The region the emitter writes for one block: a [2*head_dim] Q projection
// sliced into Q (offset 0) and the gate (offset head_dim), both stride
// 2*head_dim, then sigmoid(gate) and q * sigmoid(gate).
sbplanner::Graph gated_region(uint32_t head_dim, uint32_t rows) {
  sbplanner::Graph g;
  g.name = "gated-attention";
  auto tensor = [](uint32_t id, uint64_t n, bool input, bool output,
                   const std::string& name) {
    sbplanner::TensorDesc t;
    t.id = sbplanner::TensorId{id};
    t.shape = {int64_t(n)};
    t.dtype = sx::DType::Float32;
    t.size_bytes = n * sizeof(float);
    t.is_graph_input = input;
    t.is_graph_output = output;
    t.name = name;
    return t;
  };
  const uint64_t width = uint64_t(rows) * head_dim;
  g.tensors.push_back(tensor(0, 2 * width, true, false, "x"));
  g.tensors.push_back(tensor(1, width, false, false, "qq"));
  g.tensors.push_back(tensor(2, width, false, false, "qg"));
  g.tensors.push_back(tensor(3, width, false, false, "gs"));
  g.tensors.push_back(tensor(4, width, false, true, "gm"));
  g.inputs.push_back(sbplanner::TensorId{0});
  g.outputs.push_back(sbplanner::TensorId{4});

  const int64_t stride = int64_t(2 * head_dim);
  auto node = [](uint32_t id, sbplanner::OpKind op, std::vector<uint32_t> in,
                 uint32_t out, std::vector<sx::Attribute> attrs,
                 const std::string& name) {
    sbplanner::GraphNodeDesc n;
    n.id = sbplanner::GraphNodeId{id};
    n.op = op;
    for (uint32_t i : in) n.inputs.push_back(sbplanner::TensorId{i});
    n.outputs.push_back(sbplanner::TensorId{out});
    n.attributes = std::move(attrs);
    n.name = name;
    n.backend = sbplanner::BackendTag::Sonic;
    return n;
  };
  g.nodes.push_back(node(0, sbplanner::OpKind::Slice, {0}, 1,
                         {attr_i("offset", 0), attr_i("length", head_dim),
                          attr_i("stride", stride)},
                         "qq"));
  g.nodes.push_back(node(1, sbplanner::OpKind::Slice, {0}, 2,
                         {attr_i("offset", head_dim),
                          attr_i("length", head_dim), attr_i("stride", stride)},
                         "qg"));
  g.nodes.push_back(node(2, sbplanner::OpKind::Sigmoid, {2}, 3, {}, "gs"));
  g.nodes.push_back(node(3, sbplanner::OpKind::Mul, {1, 3}, 4, {}, "gm"));
  return g;
}

void run_case(uint32_t head_dim, uint32_t rows, const std::string& label) {
  sbplanner::Graph g = gated_region(head_dim, rows);
  sbmodel::Gemma4Model model;  // no weights: the region reads none
  sbplanner::SonicBackend backend(model, g, 0);

  std::vector<float> x(2 * rows * head_dim);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(float(i) * 0.37f) * 2.5f;
  std::vector<std::vector<float>> values(g.tensors.size());
  values[0] = x;

  std::vector<float> q, gate, s, y;
  check(run_node(backend, g, 0, values, q), label + ": the Q slice runs");
  check(run_node(backend, g, 1, values, gate), label + ": the gate slice runs");
  check(run_node(backend, g, 2, values, s), label + ": the sigmoid runs");
  check(run_node(backend, g, 3, values, y), label + ": the multiply runs");

  bool slices_ok = q.size() == rows * head_dim && gate.size() == s.size() &&
                   s.size() == y.size() && y.size() == q.size();
  double worst = 0.0;
  for (uint32_t r = 0; r < rows && slices_ok; ++r) {
    for (uint32_t i = 0; i < head_dim; ++i) {
      const float want_q = x[r * 2 * head_dim + i];
      const float want_g = x[r * 2 * head_dim + head_dim + i];
      const float want_s = 1.0f / (1.0f + std::exp(-want_g));
      const float want_y = want_q * want_s;
      const uint64_t at = uint64_t(r) * head_dim + i;
      if (q[at] != want_q || gate[at] != want_g) {
        check(false, label + ": the slices select Q then the gate per row");
        slices_ok = false;
        break;
      }
      worst = std::max(worst, double(std::fabs(y[at] - want_y)));
    }
  }
  if (slices_ok)
    check(worst < 1e-6, label + ": y == q * sigmoid(gate) (worst " +
                            std::to_string(worst) + ")");
}

} // namespace

int main() {
  // One decode row: a [2*head_dim] projection.
  run_case(8, 1, "decode");
  // The prefill layout: n rows of 2*head_dim, flattened.
  run_case(8, 3, "prefill");

  if (failures == 0) {
    std::cout << "gated attention: all checks passed\n";
    return 0;
  }
  std::cout << "gated attention: " << failures << " check(s) failed\n";
  return 1;
}
