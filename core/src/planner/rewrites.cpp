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

#include <sonicboom/planner/rewrites.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sonicboom::planner {

namespace {

uint32_t max_tensor_id(const Graph& g) {
  uint32_t m = 0;
  for (const auto& t : g.tensors)
    m = std::max(m, t.id.value);
  return m;
}

uint32_t max_node_id(const Graph& g) {
  uint32_t m = 0;
  for (const auto& n : g.nodes)
    m = std::max(m, n.id.value);
  return m;
}

} // namespace

void quantize_once_pass(Graph& g) {
  // Rewritten node list: Quantize nodes are inserted immediately before their
  // first QuantizedMatmulQ8 consumer, which is topologically after the q8
  // input's producer (a strictly earlier node) and before the consumer — so the
  // resulting node order stays topological.
  std::vector<GraphNodeDesc> out_nodes;
  out_nodes.reserve(g.nodes.size() + g.nodes.size() / 2);

  // CSE table: input tensor id -> the q8_1 tensor id already emitted for it.
  // Two Quantize nodes are identical iff their input tensor is identical (the
  // q8_1 quantization is a pure function of the activation), so one q8 tensor
  // per distinct input is enough.
  std::unordered_map<uint32_t, uint32_t> q8_of;

  uint32_t next_tensor = max_tensor_id(g) + 1;
  uint32_t next_node = max_node_id(g) + 1;

  for (GraphNodeDesc& node : g.nodes) {
    if (node.op != OpKind::QuantizedMatmul || node.inputs.size() != 1) {
      out_nodes.push_back(std::move(node));
      continue;
    }

    const TensorId x = node.inputs[0];
    uint32_t qt;
    auto it = q8_of.find(x.value);
    if (it == q8_of.end()) {
      // First use of this activation: emit a shared Quantize node. The output
      // tensor mirrors the input's f32 shape/size (a superset of the q8_1 bytes)
      // so the executor/planner treat it as an ordinary f32 intermediate; only
      // the CUDA kernels interpret the buffer as q8_1.
      const TensorDesc* xt = g.find_tensor(x);
      TensorDesc qt_desc;
      qt_desc.id = TensorId{next_tensor++};
      qt_desc.shape = xt ? xt->shape : std::vector<int64_t>{};
      qt_desc.dtype = sx::DType::Float32;
      qt_desc.size_bytes = xt ? xt->size_bytes : 0;
      qt_desc.name = "q8_" + (xt ? xt->name : std::to_string(x.value));
      qt = qt_desc.id.value;
      g.tensors.push_back(std::move(qt_desc));

      GraphNodeDesc qn;
      qn.id = GraphNodeId{next_node++};
      qn.op = OpKind::Quantize;
      qn.inputs = {x};
      qn.outputs = {TensorId{qt}};
      qn.name = op_kind_name(OpKind::Quantize);
      qn.backend = BackendTag::Sonic;
      out_nodes.push_back(std::move(qn));

      q8_of[x.value] = qt;
    } else {
      qt = it->second;
    }

    node.op = OpKind::QuantizedMatmulQ8;
    node.inputs[0] = TensorId{qt};
    node.name = op_kind_name(OpKind::QuantizedMatmulQ8);
    out_nodes.push_back(std::move(node));
  }

  g.nodes = std::move(out_nodes);
}

void fuse_rms_norm_quantize_pass(Graph& g) {
  // Producer of each tensor (tensor id -> index into g.nodes). Every tensor in a
  // planner graph has at most one producer (graph inputs have none).
  std::unordered_map<uint32_t, uint32_t> producer;
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    for (const TensorId& o : g.nodes[i].outputs)
      producer[o.value] = i;

  // Consumer count per tensor. An rms_norm output is fusable only when a single
  // Quantize node consumes it (attn_norm -> h -> q8_h), because the fusion
  // removes the f32 h buffer; a second reader would need it to still exist.
  std::unordered_map<uint32_t, uint32_t> consumers;
  for (const GraphNodeDesc& node : g.nodes)
    for (const TensorId& in : node.inputs)
      ++consumers[in.value];

  // rms_norm node index -> the single Quantize node index consuming its output.
  std::unordered_map<uint32_t, uint32_t> quantize_of_rms;
  std::unordered_set<uint32_t> fused_quantize;
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& node = g.nodes[i];
    if (node.op != OpKind::Quantize || node.inputs.size() != 1)
      continue;
    const uint32_t x = node.inputs[0].value;
    auto pit = producer.find(x);
    if (pit == producer.end())
      continue;
    if (g.nodes[pit->second].op != OpKind::RmsNorm)
      continue;
    if (consumers[x] != 1)
      continue;  // the f32 normalized activation has another reader
    quantize_of_rms[pit->second] = i;
    fused_quantize.insert(i);
  }

  if (quantize_of_rms.empty())
    return;

  // Tensors that disappear: the rms_norm's f32 output (its only consumer, the
  // Quantize node, is folded away). The Quantize node's q8_1 output tensor is
  // reused as the RmsNormQuantize output, so it stays.
  std::unordered_set<uint32_t> dead;
  for (const auto& [pidx, qidx] : quantize_of_rms)
    for (const TensorId& o : g.nodes[pidx].outputs)
      dead.insert(o.value);

  // Rebuild in place, preserving order. The RmsNorm node keeps its position
  // (and its input, which is produced earlier), so order stays topological.
  std::vector<GraphNodeDesc> out_nodes;
  out_nodes.reserve(g.nodes.size());
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    GraphNodeDesc node = g.nodes[i];
    if (node.op == OpKind::Quantize && fused_quantize.count(i)) {
      continue;  // folded into its producer's RmsNormQuantize
    }
    if (node.op == OpKind::RmsNorm) {
      auto it = quantize_of_rms.find(i);
      if (it != quantize_of_rms.end()) {
        // RmsNormQuantize: RmsNorm's input + attributes, Quantize's output.
        node.op = OpKind::RmsNormQuantize;
        node.outputs = g.nodes[it->second].outputs;
        node.name = op_kind_name(OpKind::RmsNormQuantize);
      }
    }
    out_nodes.push_back(std::move(node));
  }

  g.nodes = std::move(out_nodes);

  // Drop the dead f32 intermediate tensors (h, f, hn) — nothing references them
  // anymore, and removing them keeps memory planning from reserving a buffer.
  g.tensors.erase(
      std::remove_if(g.tensors.begin(), g.tensors.end(),
                     [&](const TensorDesc& t) { return dead.count(t.id.value) != 0; }),
      g.tensors.end());
}

} // namespace sonicboom::planner
