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

// Device executor of a traced training step (sonicboom/training/device_graph.h).
//
// The pipeline is the one the planner uses, minus the plan: the Blade document
// is lowered with bladeToBoom, the Training phase's passes rewrite it, and
// boomToGraph hands the node list back. Every tensor then gets one device
// buffer (the BufferReusePass marks in-place and buffer-class sharing, which
// this executor may ignore — per-tensor buffers are the correct fallback it
// documents), constants are uploaded once, and a step is: upload the bound
// inputs, run the nodes in order, synchronize, download the outputs.
//
// Backward nodes call the training module's existing fused kernels
// (backward_internal.h) rather than re-implementing them, so a traced step uses
// the same adjoint arithmetic as tests/trace/test-backward.scm. The forward
// vocabulary and AdamW are new and live in device_graph_kernels.cu.
//
// Buffer handles are the training module's opaque uint64_t, so this file is
// compiled without CUDA headers; the CUDA entry points below are selected at
// build time. A library built without CUDA compiles a graph only to report that
// the device is unavailable — the graph is still lowered and verified, so a
// bad graph is diagnosed the same way on either build.

#include <sonicboom/training/device_graph.h>

#include "backward_internal.h"
#include "device_graph_kernels.h"

#include "sonicboom/compiler/BladeToBoom.h"
#include "sonicboom/compiler/BoomConversion.h"
#include "sonicboom/compiler/BoomPasses.h"
#include "sonicboom/planner/ir_passes.h"

#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace sonicboom::training {

namespace {

using planner::Graph;
using planner::GraphNodeDesc;
using planner::OpKind;
using planner::TensorDesc;
using planner::TensorId;

// --- shape helpers ----------------------------------------------------------

std::int64_t numel_of(const std::vector<std::int64_t>& shape) {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

std::vector<std::int64_t> row_major_strides(
    const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> s(shape.size(), 1);
  for (std::size_t i = shape.size(); i-- > 1;)
    s[i - 1] = s[i] * shape[i];
  return s;
}

// The source's stride along each output axis: right-aligned, 0 where the
// source has a 1 that the output expands (numpy broadcasting).
std::vector<std::int64_t> broadcast_strides(
    const std::vector<std::int64_t>& out_shape,
    const std::vector<std::int64_t>& src_shape) {
  std::vector<std::int64_t> out(out_shape.size(), 0);
  if (src_shape.size() > out_shape.size()) return out;
  const auto src = row_major_strides(src_shape);
  const std::size_t lead = out_shape.size() - src_shape.size();
  for (std::size_t i = 0; i < src_shape.size(); ++i) {
    const std::size_t oi = lead + i;
    out[oi] = src_shape[i] == 1 && out_shape[oi] != 1 ? 0 : src[i];
  }
  return out;
}

// --- node attributes --------------------------------------------------------

const sx::Attribute* find_attr(const GraphNodeDesc& node,
                               std::string_view name) {
  for (const sx::Attribute& a : node.attributes)
    if (a.name == name) return &a;
  return nullptr;
}

std::int64_t int_attr(const GraphNodeDesc& node, std::string_view name,
                      std::int64_t fallback) {
  const sx::Attribute* a = find_attr(node, name);
  if (a == nullptr) return fallback;
  if (const auto* v = std::get_if<std::int64_t>(&a->value)) return *v;
  if (const auto* v = std::get_if<double>(&a->value))
    return static_cast<std::int64_t>(*v);
  if (const auto* v = std::get_if<bool>(&a->value)) return *v ? 1 : 0;
  return fallback;
}

float float_attr(const GraphNodeDesc& node, std::string_view name,
                 float fallback) {
  const sx::Attribute* a = find_attr(node, name);
  if (a == nullptr) return fallback;
  if (const auto* v = std::get_if<double>(&a->value))
    return static_cast<float>(*v);
  if (const auto* v = std::get_if<std::int64_t>(&a->value))
    return static_cast<float>(*v);
  return fallback;
}

std::vector<std::int64_t> ints_attr(const GraphNodeDesc& node,
                                    std::string_view name) {
  const sx::Attribute* a = find_attr(node, name);
  if (a == nullptr) return {};
  if (const auto* v = std::get_if<std::vector<std::int64_t>>(&a->value))
    return *v;
  if (const auto* v = std::get_if<std::int64_t>(&a->value))
    return {*v};
  return {};
}

std::string op_text(const GraphNodeDesc& node) {
  return std::string(planner::op_kind_name(node.op));
}

// --- constants --------------------------------------------------------------

// Host bytes of an inline parameter, in the tensor's own element layout.
// External sidecars are rejected: a training graph is small and its constants
// (the causal mask, the literal scalars the adjoint rules introduce) are
// inline.
bool parameter_bytes(const sx::Parameter& p, std::vector<std::byte>* out,
                     std::string* error) {
  const auto* values = std::get_if<sx::ValuesData>(&p.data);
  if (values == nullptr) {
    *error = "parameter '" + p.name +
             "' is external data; the device training graph takes inline "
             "parameters only";
    return false;
  }
  const std::int64_t n = sx::numel(p.type);
  if (static_cast<std::int64_t>(values->values.size()) != n) {
    *error = "parameter '" + p.name + "' holds " +
             std::to_string(values->values.size()) + " values but its type has " +
             std::to_string(n);
    return false;
  }

  switch (p.type.dtype) {
  case sx::DType::Float32: {
    out->resize(static_cast<std::size_t>(n) * sizeof(float));
    auto* dst = reinterpret_cast<float*>(out->data());
    for (std::int64_t i = 0; i < n; ++i) {
      if (const auto* d = std::get_if<double>(&values->values[i]))
        dst[i] = static_cast<float>(*d);
      else
        dst[i] = static_cast<float>(
            std::get<std::int64_t>(values->values[i]));
    }
    return true;
  }
  case sx::DType::Int32: {
    out->resize(static_cast<std::size_t>(n) * sizeof(std::int32_t));
    auto* dst = reinterpret_cast<std::int32_t*>(out->data());
    for (std::int64_t i = 0; i < n; ++i) {
      if (const auto* v = std::get_if<std::int64_t>(&values->values[i]))
        dst[i] = static_cast<std::int32_t>(*v);
      else
        dst[i] = static_cast<std::int32_t>(
            std::get<double>(values->values[i]));
    }
    return true;
  }
  case sx::DType::Int64: {
    out->resize(static_cast<std::size_t>(n) * sizeof(std::int64_t));
    auto* dst = reinterpret_cast<std::int64_t*>(out->data());
    for (std::int64_t i = 0; i < n; ++i) {
      if (const auto* v = std::get_if<std::int64_t>(&values->values[i]))
        dst[i] = *v;
      else
        dst[i] = static_cast<std::int64_t>(
            std::get<double>(values->values[i]));
    }
    return true;
  }
  default:
    *error = "parameter '" + p.name + "' has dtype " +
             sx::dtype_name(p.type.dtype) +
             ", which the device training graph cannot upload";
    return false;
  }
}

} // namespace

// --- the compiled graph -----------------------------------------------------

struct DeviceTrainingGraph::Impl {
  Graph graph;
  std::vector<std::uint64_t> buffers;  // by tensor id (planner ids are dense)
  std::vector<TensorId> input_ids;
  std::vector<TensorId> output_ids;
  std::vector<DeviceTensorInfo> input_info;
  std::vector<DeviceTensorInfo> output_info;

  ~Impl() {
#ifdef SONICBOOM_USE_CUDA
    for (std::uint64_t b : buffers)
      if (b != 0) detail::dev_free(b);
#endif
  }

  const TensorDesc& tensor(TensorId id) const {
    return graph.tensors[id.value];
  }
  std::uint64_t buffer(TensorId id) const { return buffers[id.value]; }
};

DeviceTrainingGraph::DeviceTrainingGraph() : impl_(std::make_unique<Impl>()) {}
DeviceTrainingGraph::~DeviceTrainingGraph() = default;
DeviceTrainingGraph::DeviceTrainingGraph(DeviceTrainingGraph&&) noexcept =
    default;
DeviceTrainingGraph& DeviceTrainingGraph::operator=(
    DeviceTrainingGraph&&) noexcept = default;

const std::vector<DeviceTensorInfo>& DeviceTrainingGraph::inputs()
    const noexcept {
  return impl_->input_info;
}

const std::vector<DeviceTensorInfo>& DeviceTrainingGraph::outputs()
    const noexcept {
  return impl_->output_info;
}

// --- node dispatch ----------------------------------------------------------

namespace {

// What a node launch reads: the graph (shapes, dtypes) and the one device
// buffer per tensor id. Small enough to rebuild per step.
struct NodeContext {
  const Graph& graph;
  const std::vector<std::uint64_t>& buffers;
};

#ifdef SONICBOOM_USE_CUDA
bool dispatch_node(const NodeContext& s, const GraphNodeDesc& n,
                   std::string* error) {
  auto buf = [&](TensorId id) { return s.buffers[id.value]; };
  auto shape = [&](TensorId id) { return s.graph.tensors[id.value].shape; };
  auto numel = [&](TensorId id) { return numel_of(s.graph.tensors[id.value].shape); };
  auto dtype = [&](TensorId id) { return s.graph.tensors[id.value].dtype; };
  auto tensor = [&](TensorId id) -> const TensorDesc& {
    return s.graph.tensors[id.value];
  };
  auto fail = [&](const std::string& why) {
    *error = "node '" + n.name + "' (" + op_text(n) + "): " + why;
    return false;
  };
  auto arity = [&](std::size_t in, std::size_t out) {
    if (n.inputs.size() != in || n.outputs.size() != out)
      return fail("expected " + std::to_string(in) + " input(s) and " +
                  std::to_string(out) + " result(s), got " +
                  std::to_string(n.inputs.size()) + " and " +
                  std::to_string(n.outputs.size()));
    return true;
  };

  // Every kernel below reads float32; the graph's dtypes are the tracer's, and
  // a backward graph is float32 throughout.
  auto is_f32 = [&](TensorId id) {
    return tensor(id).dtype == sx::DType::Float32;
  };

  try {
    switch (n.op) {
    case OpKind::Matmul: {
      if (!arity(2, 1)) return false;
      const auto a = shape(n.inputs[0]);
      const auto b = shape(n.inputs[1]);
      if (a.size() != 2 || b.size() != 2) return fail("expects rank-2 operands");
      if (!detail::graph_matmul(buf(n.outputs[0]), buf(n.inputs[0]),
                                buf(n.inputs[1]), a[0], a[1], b[1]))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::Bmm: {
      if (!arity(2, 1)) return false;
      const auto a = shape(n.inputs[0]);
      const auto b = shape(n.inputs[1]);
      if (a.size() != 3 || b.size() != 3)
        return fail("expects rank-3 operands");
      if (!detail::graph_bmm(buf(n.outputs[0]), buf(n.inputs[0]),
                             buf(n.inputs[1]), a[0], a[1], a[2], b[2]))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::Permute: {
      if (!arity(1, 1)) return false;
      const auto dims = ints_attr(n, "dims");
      const auto in = shape(n.inputs[0]);
      const auto out = shape(n.outputs[0]);
      if (dims.size() != in.size() || out.size() != in.size())
        return fail("`dims` does not describe a permutation of the operand");
      const auto src = row_major_strides(in);
      std::vector<std::int64_t> src_strides;
      src_strides.reserve(dims.size());
      for (std::int64_t d : dims) {
        if (d < 0 || static_cast<std::size_t>(d) >= in.size())
          return fail("`dims` axis out of range");
        src_strides.push_back(src[static_cast<std::size_t>(d)]);
      }
      if (!detail::graph_permute(buf(n.outputs[0]), buf(n.inputs[0]),
                                 src_strides.data(),
                                 static_cast<std::int64_t>(out.size()),
                                 numel(n.outputs[0]), out.data()))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::IndexSelect: {
      if (!arity(2, 1)) return false;
      if (int_attr(n, "axis", 0) != 0)
        return fail("only axis 0 is supported on the device");
      const auto table = shape(n.inputs[0]);
      const auto ids = shape(n.inputs[1]);
      if (table.size() != 2 || ids.size() != 1)
        return fail("expects a [rows, dim] table and 1-D indices");
      if (tensor(n.inputs[1]).dtype != sx::DType::Int32)
        return fail("indices must be int32");
      if (!detail::graph_index_select(buf(n.outputs[0]), buf(n.inputs[0]),
                                      buf(n.inputs[1]), ids[0], table[1]))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::ReduceSum: {
      if (!arity(1, 1)) return false;
      const auto in = shape(n.inputs[0]);
      std::uint64_t mask = 0;
      for (std::int64_t a : ints_attr(n, "axes")) {
        if (a < 0) a += static_cast<std::int64_t>(in.size());
        if (a < 0 || static_cast<std::size_t>(a) >= in.size())
          return fail("`axes` out of range");
        mask |= std::uint64_t(1) << static_cast<unsigned>(a);
      }
      const auto out = shape(n.outputs[0]);
      const auto in_strides = row_major_strides(in);
      if (!detail::graph_reduce_sum(buf(n.outputs[0]), buf(n.inputs[0]),
                                    in.data(), in_strides.data(), out.data(),
                                    static_cast<std::int64_t>(in.size()), mask,
                                    numel(n.outputs[0])))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::Reshape: {
      if (!arity(1, 1)) return false;
      if (numel(n.inputs[0]) != numel(n.outputs[0]))
        return fail("element count differs");
      if (!detail::dev_copy(buf(n.outputs[0]), buf(n.inputs[0]),
                            numel(n.inputs[0])))
        return fail("copy failed");
      return true;
    }
    case OpKind::Add:
    case OpKind::Mul: {
      if (!arity(2, 1)) return false;
      const auto out = shape(n.outputs[0]);
      const auto sa = broadcast_strides(out, shape(n.inputs[0]));
      const auto sb = broadcast_strides(out, shape(n.inputs[1]));
      const std::int64_t n_out = numel(n.outputs[0]);
      const bool ok =
          n.op == OpKind::Add
              ? detail::graph_add(buf(n.outputs[0]), buf(n.inputs[0]),
                                  buf(n.inputs[1]), sa.data(), sb.data(),
                                  out.data(),
                                  static_cast<std::int64_t>(out.size()), n_out)
              : detail::graph_mul(buf(n.outputs[0]), buf(n.inputs[0]),
                                  buf(n.inputs[1]), sa.data(), sb.data(),
                                  out.data(),
                                  static_cast<std::int64_t>(out.size()), n_out);
      return ok ? true : fail("kernel launch failed");
    }
    case OpKind::Softmax: {
      if (!arity(1, 1)) return false;
      const auto in = shape(n.inputs[0]);
      const std::int64_t cols = in.empty() ? 1 : in.back();
      if (!detail::graph_softmax(buf(n.outputs[0]), buf(n.inputs[0]),
                                 numel(n.inputs[0]) / cols, cols))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::RmsNorm: {
      if (!arity(2, 1)) return false;
      const auto x = shape(n.inputs[0]);
      const std::int64_t dim = x.empty() ? 1 : x.back();
      if (!is_f32(n.inputs[1])) return fail("the weight must be float32");
      if (!detail::graph_rms_norm(buf(n.outputs[0]), buf(n.inputs[0]),
                                  buf(n.inputs[1]), numel(n.inputs[0]) / dim,
                                  dim, float_attr(n, "eps", 1e-6f)))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::Rope: {
      if (!arity(1, 1)) return false;
      const auto x = shape(n.inputs[0]);
      const std::int64_t head_dim = int_attr(n, "head_dim", 0);
      if (head_dim <= 0 || x.empty() || x.back() % head_dim != 0)
        return fail("`head_dim` does not divide the last dimension");
      const std::int64_t heads = x.back() / head_dim;
      const std::int64_t rows = numel(n.inputs[0]) / x.back();
      // The rotation is in place, and a graph result cannot overwrite its
      // operand (another node may read it): rotate a copy.
      if (!detail::dev_copy(buf(n.outputs[0]), buf(n.inputs[0]),
                            numel(n.inputs[0])))
        return fail("copy failed");
      if (!detail::graph_rope_inplace(
              buf(n.outputs[0]), rows, heads, head_dim,
              static_cast<std::uint64_t>(int_attr(n, "pos0", 0)),
              float_attr(n, "theta_base", 10000.0f),
              float_attr(n, "freq_scale", 1.0f)))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::SiluMul: {
      if (!arity(2, 1)) return false;
      if (!detail::graph_silu_mul(buf(n.outputs[0]), buf(n.inputs[0]),
                                  buf(n.inputs[1]), numel(n.outputs[0])))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::CrossEntropy: {
      if (!arity(2, 1)) return false;
      const auto logits = shape(n.inputs[0]);
      if (logits.size() != 2) return fail("expects [rows, cols] logits");
      if (tensor(n.inputs[1]).dtype != sx::DType::Int32)
        return fail("targets must be int32");
      if (!detail::graph_cross_entropy(buf(n.outputs[0]), buf(n.inputs[0]),
                                       buf(n.inputs[1]), logits[0], logits[1]))
        return fail("kernel launch failed");
      return true;
    }

    // --- fused adjoints (backward_internal.h) ------------------------------
    case OpKind::MatmulBwd: {
      if (!arity(3, 2)) return false;
      const auto a = shape(n.inputs[0]);
      const auto b = shape(n.inputs[1]);
      if (a.size() != 2 || b.size() != 2)
        return fail("expects rank-2 operands");
      if (!detail::matmul_bwd_dev(buf(n.outputs[0]), buf(n.outputs[1]),
                                  buf(n.inputs[0]), buf(n.inputs[1]),
                                  buf(n.inputs[2]), a[0], a[1], b[1]))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::SoftmaxBwd: {
      if (!arity(2, 1)) return false;
      const auto y = shape(n.inputs[0]);
      const std::int64_t cols = y.empty() ? 1 : y.back();
      if (!detail::softmax_bwd_dev(buf(n.outputs[0]), buf(n.inputs[0]),
                                   buf(n.inputs[1]),
                                   numel(n.inputs[0]) / cols, cols))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::CrossEntropyBwd: {
      if (!arity(2, 2)) return false;
      const auto logits = shape(n.inputs[0]);
      if (logits.size() != 2) return fail("expects [rows, cols] logits");
      if (tensor(n.inputs[1]).dtype != sx::DType::Int32)
        return fail("targets must be int32");
      if (!detail::cross_entropy_bwd_dev(
              buf(n.outputs[0]), buf(n.outputs[1]), buf(n.inputs[0]),
              buf(n.inputs[1]), logits[0], logits[1]))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::RmsNormBwd: {
      if (!arity(3, 2)) return false;
      const auto x = shape(n.inputs[0]);
      const std::int64_t dim = x.empty() ? 1 : x.back();
      if (!detail::rmsnorm_bwd_dev(
              buf(n.outputs[0]), buf(n.outputs[1]), buf(n.inputs[0]),
              buf(n.inputs[1]), buf(n.inputs[2]), numel(n.inputs[0]) / dim,
              dim, float_attr(n, "eps", 1e-6f)))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::RopeBwd: {
      if (!arity(1, 1)) return false;
      const auto x = shape(n.inputs[0]);
      const std::int64_t head_dim = int_attr(n, "head_dim", 0);
      if (head_dim <= 0 || x.empty() || x.back() % head_dim != 0)
        return fail("`head_dim` does not divide the last dimension");
      const std::int64_t heads = x.back() / head_dim;
      const std::int64_t rows = numel(n.inputs[0]) / x.back();
      if (!detail::rope_bwd_dev(
              buf(n.outputs[0]), buf(n.inputs[0]), rows, heads, head_dim,
              static_cast<std::uint64_t>(int_attr(n, "pos0", 0)),
              float_attr(n, "theta_base", 10000.0f),
              float_attr(n, "freq_scale", 1.0f)))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::SiluMulBwd: {
      if (!arity(3, 2)) return false;
      if (!detail::silu_mul_bwd_dev(buf(n.outputs[0]), buf(n.outputs[1]),
                                    buf(n.inputs[0]), buf(n.inputs[1]),
                                    buf(n.inputs[2]), numel(n.inputs[0])))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::EmbeddingBwd: {
      if (!arity(2, 1)) return false;
      if (tensor(n.inputs[0]).dtype != sx::DType::Int32)
        return fail("token ids must be int32");
      const auto dy = shape(n.inputs[1]);
      if (dy.size() != 2) return fail("expects [rows, dim] gradients");
      const std::int64_t vocab = int_attr(n, "vocab", 0);
      if (vocab <= 0) return fail("`vocab` must be positive");
      if (!detail::embedding_bwd_dev(buf(n.outputs[0]), buf(n.inputs[0]),
                                     buf(n.inputs[1]), dy[0], dy[1], vocab))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::AdamW: {
      if (!arity(5, 3)) return false;
      if (tensor(n.inputs[4]).dtype != sx::DType::Int32)
        return fail("the step counter must be int32");
      const std::int64_t n_elem = numel(n.inputs[0]);
      if (!detail::graph_adamw_step(
              buf(n.outputs[0]), buf(n.outputs[1]), buf(n.outputs[2]),
              buf(n.inputs[0]), buf(n.inputs[1]), buf(n.inputs[2]),
              buf(n.inputs[3]), n_elem, buf(n.inputs[4]),
              float_attr(n, "lr", 1e-3f), float_attr(n, "beta1", 0.9f),
              float_attr(n, "beta2", 0.999f), float_attr(n, "eps", 1e-8f),
              float_attr(n, "weight_decay", 0.01f)))
        return fail("kernel launch failed");
      return true;
    }
    case OpKind::ClipGradNorm: {
      if (n.inputs.empty() || n.outputs.size() != n.inputs.size())
        return fail("expects one result per gradient operand");
      const double max_norm = double(float_attr(n, "max_norm", 1.0f));
      // One double accumulator for the whole set: every tensor contributes in
      // one pass (graph_sumsq_acc), then the scale decision is made once.
      const detail::DevBuf acc = detail::dev_alloc(sizeof(double));
      if (acc == 0) return fail("cannot allocate the norm accumulator");
      bool ok = detail::dev_zero(acc, sizeof(double));
      for (std::size_t i = 0; ok && i < n.inputs.size(); ++i)
        ok = detail::graph_sumsq_acc(acc, buf(n.inputs[i]),
                                     numel(n.inputs[i]));
      double sumsq = 0.0;
      if (ok)
        ok = detail::dev_sync() &&
             detail::dev_download(acc, &sumsq, sizeof(sumsq));
      detail::dev_free(acc);
      if (!ok) return fail("kernel launch failed");
      const double norm = std::sqrt(sumsq);
      const double scale =
          norm > max_norm ? max_norm / (norm + 1e-6) : 1.0;
      for (std::size_t i = 0; i < n.inputs.size(); ++i)
        if (!detail::graph_scale_copy(buf(n.outputs[i]), buf(n.inputs[i]),
                                      numel(n.inputs[i]), scale))
          return fail("kernel launch failed");
      return true;
    }
    default:
      *error = "node '" + n.name + "' (" + op_text(n) +
               "): not supported by the device training executor";
      return false;
    }
  } catch (const std::exception& e) {
    *error = "node '" + n.name + "' (" + op_text(n) + "): " + e.what();
    return false;
  }
}
#else
// The kernels live in the CUDA-only translation unit; a library without CUDA
// never reaches this (compile() refuses first), and the stub keeps the
// unconditional graph checks link-clean.
bool dispatch_node(const NodeContext&, const GraphNodeDesc&,
                   std::string* error) {
  if (error) *error = "SonicBoom was built without CUDA";
  return false;
}
#endif

// The graph-level half of a node's checks: what the executor supports at all,
// and the properties of shapes, attributes and dtypes its kernels assume. These
// are facts about the compiled graph, so they are rejected at compile rather
// than on the first step.
bool validate_node(const Graph& g, const GraphNodeDesc& n, std::string* error) {
  auto shape = [&](TensorId id) -> const std::vector<std::int64_t>& {
    return g.tensors[id.value].shape;
  };
  auto dtype = [&](TensorId id) { return g.tensors[id.value].dtype; };
  auto numel = [&](TensorId id) { return numel_of(shape(id)); };
  auto fail = [&](const std::string& why) {
    *error = "node '" + n.name + "' (" + op_text(n) + "): " + why;
    return false;
  };
  auto arity = [&](std::size_t in, std::size_t out) {
    if (n.inputs.size() != in || n.outputs.size() != out)
      return fail("expected " + std::to_string(in) + " input(s) and " +
                  std::to_string(out) + " result(s), got " +
                  std::to_string(n.inputs.size()) + " and " +
                  std::to_string(n.outputs.size()));
    return true;
  };
  auto rope_ok = [&]() {
    const auto& x = shape(n.inputs[0]);
    const std::int64_t head_dim = int_attr(n, "head_dim", 0);
    if (head_dim <= 0) return fail("`head_dim` is missing");
    if (x.empty() || x.back() % head_dim != 0)
      return fail("`head_dim` does not divide the last dimension");
    return true;
  };

  switch (n.op) {
  case OpKind::Matmul:
  case OpKind::MatmulBwd: {
    if (!arity(n.op == OpKind::Matmul ? 2 : 3,
               n.op == OpKind::Matmul ? 1 : 2))
      return false;
    if (shape(n.inputs[0]).size() != 2 || shape(n.inputs[1]).size() != 2)
      return fail("expects rank-2 operands");
    return true;
  }
  case OpKind::Bmm:
    return arity(2, 1) &&
           (shape(n.inputs[0]).size() == 3 && shape(n.inputs[1]).size() == 3
                ? true
                : fail("expects rank-3 operands"));
  case OpKind::Permute: {
    if (!arity(1, 1)) return false;
    const auto& in = shape(n.inputs[0]);
    const auto dims = ints_attr(n, "dims");
    if (dims.size() != in.size())
      return fail("`dims` does not describe a permutation of the operand");
    for (std::int64_t d : dims)
      if (d < 0 || static_cast<std::size_t>(d) >= in.size())
        return fail("`dims` axis out of range");
    return true;
  }
  case OpKind::IndexSelect:
    if (!arity(2, 1)) return false;
    if (int_attr(n, "axis", 0) != 0)
      return fail("only axis 0 is supported on the device");
    if (shape(n.inputs[0]).size() != 2 || shape(n.inputs[1]).size() != 1)
      return fail("expects a [rows, dim] table and 1-D indices");
    if (dtype(n.inputs[1]) != sx::DType::Int32)
      return fail("indices must be int32");
    return true;
  case OpKind::ReduceSum: {
    if (!arity(1, 1)) return false;
    const auto& in = shape(n.inputs[0]);
    for (std::int64_t a : ints_attr(n, "axes")) {
      if (a < 0) a += static_cast<std::int64_t>(in.size());
      if (a < 0 || static_cast<std::size_t>(a) >= in.size())
        return fail("`axes` out of range");
    }
    return true;
  }
  case OpKind::Reshape:
    return arity(1, 1) &&
           (numel(n.inputs[0]) == numel(n.outputs[0])
                ? true
                : fail("element count differs"));
  case OpKind::Add:
  case OpKind::Mul: {
    if (!arity(2, 1)) return false;
    const auto& out = shape(n.outputs[0]);
    if (shape(n.inputs[0]).size() > out.size() ||
        shape(n.inputs[1]).size() > out.size())
      return fail("operand rank exceeds the result rank");
    return true;
  }
  case OpKind::Softmax:
    return arity(1, 1) && (!shape(n.inputs[0]).empty()
                               ? true
                               : fail("expects a ranked tensor"));
  case OpKind::RmsNorm:
    if (!arity(2, 1)) return false;
    if (dtype(n.inputs[1]) != sx::DType::Float32)
      return fail("the weight must be float32");
    return shape(n.inputs[0]).empty() ? fail("expects a ranked tensor") : true;
  case OpKind::Rope:
    return arity(1, 1) && rope_ok();
  case OpKind::RopeBwd:
    return arity(1, 1) && rope_ok();
  case OpKind::SiluMul:
    return arity(2, 1);
  case OpKind::CrossEntropy:
  case OpKind::CrossEntropyBwd: {
    if (!arity(2, n.op == OpKind::CrossEntropy ? 1 : 2)) return false;
    if (shape(n.inputs[0]).size() != 2)
      return fail("expects [rows, cols] logits");
    if (dtype(n.inputs[1]) != sx::DType::Int32)
      return fail("targets must be int32");
    return true;
  }
  case OpKind::SoftmaxBwd:
    return arity(2, 1) && (!shape(n.inputs[0]).empty()
                               ? true
                               : fail("expects a ranked tensor"));
  case OpKind::RmsNormBwd:
    if (!arity(3, 2)) return false;
    return shape(n.inputs[0]).empty() ? fail("expects a ranked tensor") : true;
  case OpKind::SiluMulBwd:
    return arity(3, 2);
  case OpKind::EmbeddingBwd: {
    if (!arity(2, 1)) return false;
    if (dtype(n.inputs[0]) != sx::DType::Int32)
      return fail("token ids must be int32");
    if (shape(n.inputs[1]).size() != 2)
      return fail("expects [rows, dim] gradients");
    if (int_attr(n, "vocab", 0) <= 0) return fail("`vocab` must be positive");
    return true;
  }
  case OpKind::AdamW: {
    if (!arity(5, 3)) return false;
    if (dtype(n.inputs[4]) != sx::DType::Int32)
      return fail("the step counter must be int32");
    if (numel(n.inputs[1]) != numel(n.inputs[0]))
      return fail("gradient and parameter differ in size");
    return true;
  }
  case OpKind::ClipGradNorm: {
    if (n.inputs.empty() || n.outputs.size() != n.inputs.size())
      return fail("expects one result per gradient operand");
    const double max_norm = double(float_attr(n, "max_norm", 1.0f));
    if (!std::isfinite(max_norm) || max_norm < 0.0)
      return fail("`max_norm` must be finite and non-negative");
    for (std::size_t i = 0; i < n.inputs.size(); ++i) {
      if (dtype(n.inputs[i]) != sx::DType::Float32)
        return fail("every gradient must be float32");
      if (shape(n.outputs[i]) != shape(n.inputs[i]))
        return fail("a result has a different shape from its gradient");
    }
    return true;
  }
  default:
    return fail("not supported by the device training executor");
  }
}

} // namespace

// --- compile ----------------------------------------------------------------

std::expected<std::unique_ptr<DeviceTrainingGraph>, std::string>
DeviceTrainingGraph::compile(const sx::Document& doc) {
  // Lower and pass first, on every build: a graph the IR rejects must be
  // rejected for that reason, not for the device's.
  mlir::MLIRContext context;
  auto module = compiler::bladeToBoom(doc, &context);
  if (!module)
    return std::unexpected("Blade -> Boom lowering failed (diagnostic above)");
  if (!compiler::runBoomPasses(*module, planner::IrPassPhase::Training))
    return std::unexpected("Boom IR Training passes failed (diagnostic above)");
  auto lowered = compiler::boomToGraph(*module);
  if (!lowered)
    return std::unexpected("Boom IR -> graph lowering failed (diagnostic above)");

#ifndef SONICBOOM_USE_CUDA
  return std::unexpected(
      "SonicBoom was built without CUDA; the device training graph needs it");
#else
  auto out = std::unique_ptr<DeviceTrainingGraph>(new DeviceTrainingGraph());
  Impl& s = *out->impl_;
  s.graph = std::move(*lowered);

  const Graph& g = s.graph;

  std::map<std::string, const sx::Parameter*> params;
  for (const sx::Parameter& p : doc.graph.parameters) params[p.name] = &p;

  s.buffers.assign(g.tensors.size(), 0);
  for (const TensorDesc& t : g.tensors) {
    if (t.size_bytes == 0)
      return std::unexpected("tensor '" + t.name + "' has no storage");
    const std::uint64_t b = detail::dev_alloc(t.size_bytes);
    if (b == 0)
      return std::unexpected("cannot allocate " +
                             std::to_string(t.size_bytes) + " bytes for tensor '" +
                             t.name + "'");
    s.buffers[t.id.value] = b;
  }

  for (const TensorDesc& t : g.tensors) {
    if (!t.is_constant) continue;
    auto it = params.find(t.name);
    if (it == params.end())
      return std::unexpected("constant tensor '" + t.name +
                             "' has no parameter in the document");
    std::vector<std::byte> bytes;
    std::string error;
    if (!parameter_bytes(*it->second, &bytes, &error))
      return std::unexpected(error);
    if (bytes.size() != t.size_bytes)
      return std::unexpected("constant tensor '" + t.name + "' is " +
                             std::to_string(t.size_bytes) + " bytes but its " +
                             "parameter holds " +
                             std::to_string(bytes.size()));
    if (!detail::dev_upload(s.buffers[t.id.value], bytes.data(), bytes.size()))
      return std::unexpected("cannot upload constant '" + t.name + "'");
  }

  auto info_of = [&](TensorId id) {
    const TensorDesc& t = g.tensors[id.value];
    DeviceTensorInfo info;
    info.name = t.name;
    info.dtype = t.dtype;
    info.shape = t.shape;
    info.num_bytes = t.size_bytes;
    return info;
  };
  s.input_ids = g.inputs;
  s.output_ids = g.outputs;
  for (TensorId id : g.inputs) s.input_info.push_back(info_of(id));
  for (TensorId id : g.outputs) s.output_info.push_back(info_of(id));

  // The graph is lowered; its nodes must be ones this executor runs, with the
  // dtypes their kernels read. Checked once here so a step never fails for a
  // reason that is a property of the graph.
  for (const GraphNodeDesc& n : g.nodes) {
    std::string error;
    if (!validate_node(g, n, &error)) return std::unexpected(error);
  }

  return std::expected<std::unique_ptr<DeviceTrainingGraph>, std::string>(
      std::move(out));
#endif
}

// --- run --------------------------------------------------------------------

bool DeviceTrainingGraph::run(const std::byte* const* input_data,
                              const std::size_t* input_sizes,
                              std::vector<std::vector<std::byte>>* outputs,
                              std::string* error) {
#ifndef SONICBOOM_USE_CUDA
  if (error) *error = "SonicBoom was built without CUDA";
  return false;
#else
  Impl& s = *impl_;
  if (input_data == nullptr || input_sizes == nullptr || outputs == nullptr) {
    if (error) *error = "run: null argument";
    return false;
  }
  if (s.input_info.size() != s.input_ids.size()) {
    if (error) *error = "run: malformed compiled graph";
    return false;
  }

  for (std::size_t i = 0; i < s.input_ids.size(); ++i) {
    const std::uint64_t want = s.input_info[i].num_bytes;
    if (input_sizes[i] != want) {
      if (error)
        *error = "input " + std::to_string(i) + " ('" +
                 s.input_info[i].name + "') is " +
                 std::to_string(input_sizes[i]) + " bytes, expected " +
                 std::to_string(want);
      return false;
    }
    if (input_data[i] == nullptr && want != 0) {
      if (error)
        *error = "input " + std::to_string(i) + " ('" +
                 s.input_info[i].name + "') is null";
      return false;
    }
    if (!detail::dev_upload(s.buffers[s.input_ids[i].value], input_data[i],
                            want)) {
      if (error) *error = "cannot upload input " + std::to_string(i);
      return false;
    }
  }

  const NodeContext ctx{s.graph, s.buffers};
  for (const GraphNodeDesc& n : s.graph.nodes) {
    std::string why;
    if (!dispatch_node(ctx, n, &why)) {
      if (error) *error = why;
      return false;
    }
  }

  if (!detail::dev_sync()) {
    if (error) *error = "device synchronization failed";
    return false;
  }

  std::vector<std::vector<std::byte>> result(s.output_ids.size());
  for (std::size_t i = 0; i < s.output_ids.size(); ++i) {
    const std::uint64_t bytes = s.output_info[i].num_bytes;
    result[i].resize(bytes);
    if (!detail::dev_download(s.buffers[s.output_ids[i].value], result[i].data(),
                              bytes)) {
      if (error)
        *error = "cannot download output " + std::to_string(i) + " ('" +
                 s.output_info[i].name + "')";
      return false;
    }
  }
  if (!detail::dev_sync()) {
    if (error) *error = "device synchronization failed";
    return false;
  }

  *outputs = std::move(result);
  return true;
#endif
}

} // namespace sonicboom::training
