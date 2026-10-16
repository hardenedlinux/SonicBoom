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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
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

// Reads an Int attribute by name, mirroring the CUDA backend's attr_i so the
// passes and the executor agree on the attr encoding (sx::AttrValueKind::Int).
bool attr_i(const GraphNodeDesc& n, std::string_view name, int64_t& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::Int) {
      out = std::get<int64_t>(a.value);
      return true;
    }
  return false;
}

// block_q8_1_mmq size in bytes (144: a 16-byte scale union + 128 int8 quants;
// matches the static_assert on quant::cuda::BlockQ8_1Mmq) and the matmul's
// J_max (= I == 128) tail over-read guard, so the pass can size-check the f32
// output tensor it reuses as the mmq activation buffer.
constexpr size_t kMmqBlockBytes = 144;
constexpr size_t kMmqTailGuardBlocks = 128;

// Whether a `size_bytes` f32 buffer covers the mmq activation buffer for a
// [n, dim] position-major input: (dim/128)*n blocks of data + the J_max tail
// guard, 144 bytes each. dim must be a multiple of 128 (the kernels enforce it).
bool mmq_q8_fits(int dim, int n, uint64_t size_bytes) {
  const size_t blocks = (size_t(dim) / 128) * size_t(n) + kMmqTailGuardBlocks;
  return size_bytes >= blocks * kMmqBlockBytes;
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

void fuse_gelu_mul_quantize_pass(Graph& g) {
  std::unordered_map<uint32_t, uint32_t> producer;
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    for (const TensorId& o : g.nodes[i].outputs)
      producer[o.value] = i;

  std::unordered_map<uint32_t, uint32_t> consumers;
  for (const GraphNodeDesc& node : g.nodes)
    for (const TensorId& in : node.inputs)
      ++consumers[in.value];

  // For each Quantize(fh) whose producer is Mul(gelu, up) and whose gelu comes
  // from a single-use GeluFp16(gate), fold the three nodes into one
  // GeluMulQuantize(gate, up). Keys are gelu node indices.
  std::unordered_map<uint32_t, uint32_t> mul_of_gelu;
  std::unordered_map<uint32_t, uint32_t> quantize_of_gelu;
  std::unordered_set<uint32_t> fused_mul;
  std::unordered_set<uint32_t> fused_quantize;

  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& node = g.nodes[i];
    if (node.op != OpKind::Quantize || node.inputs.size() != 1)
      continue;
    const uint32_t fh = node.inputs[0].value;
    auto mit = producer.find(fh);
    if (mit == producer.end())
      continue;
    const GraphNodeDesc& mul = g.nodes[mit->second];
    if (mul.op != OpKind::Mul || mul.inputs.size() != 2)
      continue;
    if (consumers[fh] != 1)
      continue;  // the gelu*up product has another reader
    const uint32_t gelu_t = mul.inputs[0].value;
    auto git = producer.find(gelu_t);
    if (git == producer.end())
      continue;
    const GraphNodeDesc& gelu = g.nodes[git->second];
    if (gelu.op != OpKind::GeluFp16 || gelu.inputs.size() != 1)
      continue;
    if (consumers[gelu_t] != 1)
      continue;  // the gelu output has another reader
    mul_of_gelu[git->second] = mit->second;
    quantize_of_gelu[git->second] = i;
    fused_mul.insert(mit->second);
    fused_quantize.insert(i);
  }

  if (quantize_of_gelu.empty())
    return;

  // Dead tensors: the gelu output and the gelu*up product (both folded away).
  std::unordered_set<uint32_t> dead;
  for (const auto& [gidx, midx] : mul_of_gelu) {
    for (const TensorId& o : g.nodes[gidx].outputs)
      dead.insert(o.value);
    for (const TensorId& o : g.nodes[midx].outputs)
      dead.insert(o.value);
  }

  std::vector<GraphNodeDesc> out_nodes;
  out_nodes.reserve(g.nodes.size());
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    GraphNodeDesc node = g.nodes[i];
    if ((node.op == OpKind::Mul && fused_mul.count(i)) ||
        (node.op == OpKind::Quantize && fused_quantize.count(i))) {
      continue;  // folded into the GeluMulQuantize
    }
    if (node.op == OpKind::GeluFp16) {
      auto it = quantize_of_gelu.find(i);
      if (it != quantize_of_gelu.end()) {
        const GraphNodeDesc& mul = g.nodes[mul_of_gelu[i]];
        const GraphNodeDesc& q = g.nodes[it->second];
        // GeluMulQuantize: gelu's input (gate) + mul's up operand, Quantize's output.
        node.op = OpKind::GeluMulQuantize;
        node.inputs = {node.inputs[0], mul.inputs[1]};
        node.outputs = q.outputs;
        node.name = op_kind_name(OpKind::GeluMulQuantize);
      }
    }
    out_nodes.push_back(std::move(node));
  }

  g.nodes = std::move(out_nodes);

  g.tensors.erase(
      std::remove_if(g.tensors.begin(), g.tensors.end(),
                     [&](const TensorDesc& t) { return dead.count(t.id.value) != 0; }),
      g.tensors.end());
}

void fuse_rms_norm_mmq_pass(Graph& g) {
  // Consumer node indices per tensor.
  std::unordered_map<uint32_t, std::vector<uint32_t>> consumers;
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    for (const TensorId& in : g.nodes[i].inputs)
      consumers[in.value].push_back(i);

  // norm node index -> its QuantizedMatmul consumer indices, plus the shared
  // `ds`. Fused only when every consumer is a batched QuantizedMatmul sharing
  // one `ds` (a mixed-precision norm like attn_norm -> q/k(D4) + v(DS4) is left
  // unfused), and the f32 output buffer also covers the mmq activation's tail
  // guard (mmq_q8_fits).
  std::unordered_map<uint32_t, std::vector<uint32_t>> matmul_of_rms;
  std::unordered_map<uint32_t, int64_t> ds_of_rms;

  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& node = g.nodes[i];
    if (node.op != OpKind::RmsNorm || node.inputs.size() != 1 ||
        node.outputs.empty())
      continue;
    int64_t nb = 0, dim = 0;
    if (!attr_i(node, "batch", nb) || !attr_i(node, "dim", dim) || nb <= 0 ||
        dim <= 0)
      continue;
    const uint32_t out = node.outputs[0].value;
    auto cit = consumers.find(out);
    if (cit == consumers.end() || cit->second.empty())
      continue;

    int64_t ds = -1;
    std::vector<uint32_t> matmuls;
    bool ok = true;
    for (uint32_t c : cit->second) {
      const GraphNodeDesc& cn = g.nodes[c];
      if (cn.op != OpKind::QuantizedMatmul || cn.inputs.size() != 1) {
        ok = false;
        break;
      }
      int64_t cnb = 0, cds = 0;
      if (!attr_i(cn, "batch", cnb) || cnb != nb || !attr_i(cn, "ds", cds)) {
        ok = false;
        break;
      }
      // ds < 0 marks a weight type with no mmq scale layout (the legacy
      // 32-element block formats); such a matmul never fuses into mmq.
      if (cds < 0) {
        ok = false;
        break;
      }
      if (ds == -1)
        ds = cds;
      else if (ds != cds) {
        ok = false;
        break;
      }
      matmuls.push_back(c);
    }
    if (!ok)
      continue;

    const TensorDesc* ot = g.find_tensor(node.outputs[0]);
    if (!ot || !mmq_q8_fits(int(dim), int(nb), ot->size_bytes))
      continue;

    matmul_of_rms[i] = std::move(matmuls);
    ds_of_rms[i] = ds;
  }

  if (matmul_of_rms.empty())
    return;

  std::unordered_set<uint32_t> to_q8;
  for (const auto& [nidx, cs] : matmul_of_rms)
    for (uint32_t c : cs)
      to_q8.insert(c);

  // In-place rewrite: no node is added or removed, only op kind / name / attrs
  // change, so the topological order (and the producer/consumer tensor ids) is
  // untouched.
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    GraphNodeDesc& node = g.nodes[i];
    if (matmul_of_rms.count(i)) {
      node.op = OpKind::RmsNormQuantize;
      node.name = op_kind_name(OpKind::RmsNormQuantize);
      node.attributes.push_back(sx::Attribute{"ds", ds_of_rms[i]});
    } else if (to_q8.count(i)) {
      node.op = OpKind::QuantizedMatmulQ8;
      node.name = op_kind_name(OpKind::QuantizedMatmulQ8);
    }
  }
}

void fuse_gelu_mul_mmq_pass(Graph& g) {
  std::unordered_map<uint32_t, uint32_t> producer;
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    for (const TensorId& o : g.nodes[i].outputs)
      producer[o.value] = i;

  std::unordered_map<uint32_t, uint32_t> consumers;
  for (const GraphNodeDesc& node : g.nodes)
    for (const TensorId& in : node.inputs)
      ++consumers[in.value];

  // For each batched QuantizedMatmul (the FFN down projection) whose input fh
  // is produced by Mul(gelu, up) and whose gelu comes from a single-use
  // GeluFp16(gate), fold the three nodes into GeluMulQuantize(gate, up) +
  // QuantizedMatmulQ8. Keys are gelu node indices.
  std::unordered_map<uint32_t, uint32_t> matmul_of_gelu;
  std::unordered_map<uint32_t, uint32_t> mul_of_gelu;
  std::unordered_map<uint32_t, int64_t> batch_of_gelu;
  std::unordered_map<uint32_t, int64_t> dim_of_gelu;
  std::unordered_map<uint32_t, int64_t> ds_of_gelu;
  std::unordered_set<uint32_t> fused_mul;
  std::unordered_set<uint32_t> to_q8;

  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& node = g.nodes[i];
    if (node.op != OpKind::QuantizedMatmul || node.inputs.size() != 1)
      continue;
    int64_t nb = 0, dim = 0, ds = 0;
    if (!attr_i(node, "batch", nb) || !attr_i(node, "dim", dim) ||
        !attr_i(node, "ds", ds) || nb <= 0 || dim <= 0)
      continue;
    const uint32_t fh = node.inputs[0].value;
    auto mit = producer.find(fh);
    if (mit == producer.end())
      continue;
    const GraphNodeDesc& mul = g.nodes[mit->second];
    if (mul.op != OpKind::Mul || mul.inputs.size() != 2)
      continue;
    if (consumers[fh] != 1)
      continue;  // the gelu*up product has another reader
    const uint32_t gelu_t = mul.inputs[0].value;
    auto git = producer.find(gelu_t);
    if (git == producer.end())
      continue;
    const GraphNodeDesc& gelu = g.nodes[git->second];
    if (gelu.op != OpKind::GeluFp16 || gelu.inputs.size() != 1)
      continue;
    if (consumers[gelu_t] != 1)
      continue;  // the gelu output has another reader
    const TensorDesc* ft = g.find_tensor(mul.outputs[0]);
    if (!ft || !mmq_q8_fits(int(dim), int(nb), ft->size_bytes))
      continue;
    matmul_of_gelu[git->second] = i;
    mul_of_gelu[git->second] = mit->second;
    batch_of_gelu[git->second] = nb;
    dim_of_gelu[git->second] = dim;
    ds_of_gelu[git->second] = ds;
    fused_mul.insert(mit->second);
    to_q8.insert(i);
  }

  if (matmul_of_gelu.empty())
    return;

  // Dead tensors: the gelu f32 output (folded away); the mul's f32 output is
  // reused as the GeluMulQuantize q8 buffer, so it stays.
  std::unordered_set<uint32_t> dead;
  for (const auto& [gidx, midx] : mul_of_gelu)
    for (const TensorId& o : g.nodes[gidx].outputs)
      dead.insert(o.value);

  // Rebuild: the Mul node is dropped (folded into its producer's GeluMulQuantize).
  // The GeluFp16 node keeps its position (its gate input is produced earlier),
  // so the order stays topological.
  std::vector<GraphNodeDesc> out_nodes;
  out_nodes.reserve(g.nodes.size());
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    GraphNodeDesc node = g.nodes[i];
    if (node.op == OpKind::Mul && fused_mul.count(i)) {
      continue;  // folded into the GeluMulQuantize
    }
    if (matmul_of_gelu.count(i)) {
      const GraphNodeDesc& mul = g.nodes[mul_of_gelu[i]];
      node.op = OpKind::GeluMulQuantize;
      node.name = op_kind_name(OpKind::GeluMulQuantize);
      node.inputs = {node.inputs[0], mul.inputs[1]};
      node.outputs = mul.outputs;  // reuse the mul's f32 output as the q8 buffer
      node.attributes.push_back(sx::Attribute{"batch", batch_of_gelu[i]});
      node.attributes.push_back(sx::Attribute{"dim", dim_of_gelu[i]});
      node.attributes.push_back(sx::Attribute{"ds", ds_of_gelu[i]});
    } else if (to_q8.count(i)) {
      node.op = OpKind::QuantizedMatmulQ8;
      node.name = op_kind_name(OpKind::QuantizedMatmulQ8);
    }
    out_nodes.push_back(std::move(node));
  }

  g.nodes = std::move(out_nodes);

  g.tensors.erase(
      std::remove_if(g.tensors.begin(), g.tensors.end(),
                     [&](const TensorDesc& t) { return dead.count(t.id.value) != 0; }),
      g.tensors.end());
}

void fuse_bias_into_gemv_pass(Graph& g) {
  // Consumer count per tensor: a projection whose output feeds anything besides
  // the AddBias cannot take the bias over (that consumer would read the fused
  // node's output tensor instead).
  std::unordered_map<uint32_t, uint32_t> consumers;
  for (const GraphNodeDesc& node : g.nodes)
    for (const TensorId& in : node.inputs)
      ++consumers[in.value];

  std::unordered_map<uint32_t, uint32_t> producer;
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    for (const TensorId& o : g.nodes[i].outputs)
      producer[o.value] = i;

  // AddBias node index -> (matmul node index, bias weight slot). The AddBias
  // indices are exactly the nodes the rewrite removes.
  std::unordered_map<uint32_t, uint32_t> matmul_of_bias;
  std::unordered_map<uint32_t, int64_t> slot_of_bias;

  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& node = g.nodes[i];
    if (node.op != OpKind::AddBias || node.inputs.size() != 1) continue;
    int64_t bias_slot = 0;
    if (!attr_i(node, "slot", bias_slot)) continue;
    int64_t ab_batch = 0;
    if (attr_i(node, "batch", ab_batch)) continue;  // batched (prefill) form
    const uint32_t proj = node.inputs[0].value;
    auto pit = producer.find(proj);
    if (pit == producer.end()) continue;
    const GraphNodeDesc& m = g.nodes[pit->second];
    if (m.op != OpKind::QuantizedMatmulQ8 || m.outputs.size() != 1) continue;
    int64_t m_batch = 0;
    if (attr_i(m, "batch", m_batch)) continue;  // no epilogue on the mmq path
    if (consumers[proj] != 1) continue;         // the projection has another reader
    matmul_of_bias[i] = pit->second;
    slot_of_bias[i] = bias_slot;
  }

  if (matmul_of_bias.empty())
    return;

  // The matmul's own f32 output tensor is replaced by the AddBias's output
  // tensor (same shape, one buffer either way), so it becomes dead.
  std::unordered_set<uint32_t> dead;
  for (const auto& [bi, mi] : matmul_of_bias)
    for (const TensorId& o : g.nodes[mi].outputs)
      dead.insert(o.value);

  std::unordered_map<uint32_t, uint32_t> bias_of_matmul;  // matmul idx -> AddBias idx
  for (const auto& [bi, mi] : matmul_of_bias)
    bias_of_matmul[mi] = bi;

  std::vector<GraphNodeDesc> out_nodes;
  out_nodes.reserve(g.nodes.size());
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    if (matmul_of_bias.count(i))
      continue;  // folded into the matmul that produced its input
    GraphNodeDesc node = g.nodes[i];
    auto bit = bias_of_matmul.find(i);
    if (bit != bias_of_matmul.end()) {
      const uint32_t bi = bit->second;
      node.outputs = g.nodes[bi].outputs;  // take over the AddBias's output
      node.attributes.push_back(sx::Attribute{"bias_slot", slot_of_bias[bi]});
    }
    out_nodes.push_back(std::move(node));
  }

  g.nodes = std::move(out_nodes);

  g.tensors.erase(
      std::remove_if(g.tensors.begin(), g.tensors.end(),
                     [&](const TensorDesc& t) { return dead.count(t.id.value) != 0; }),
      g.tensors.end());
}

} // namespace sonicboom::planner
