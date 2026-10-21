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

// Direct native-torch kernel calls for the ops the switch selects
// (SONICBOOM_TRAIN_ATEN); the header is ATen-free on purpose.
#include "../../layer1/adapter/aten_direct.h"

#include "../planner/device_pool.h"

#include "sonicboom/compiler/BladeToBoom.h"
#include "sonicboom/compiler/BoomConversion.h"
#include "sonicboom/compiler/BoomPasses.h"
#include "sonicboom/planner/ir_passes.h"

#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
  // One device allocation per buffer slot. A slot is shared by the tensors the
  // Training phase's BufferReusePass put in one buffer class (or by an
  // in-place mark), and by an optimizer update's operand and result, so this is
  // where memory is released.
  std::vector<std::uint64_t> slot_buffers;
  std::vector<TensorId> input_ids;
  std::vector<TensorId> output_ids;
  std::vector<DeviceTensorInfo> input_info;
  std::vector<DeviceTensorInfo> output_info;

  // Resident inputs: the graph writes the next value of these back into their
  // own device buffers (the optimizer update), so their host bytes are needed
  // once. `state_uploaded` says a run has already filled them.
  std::vector<bool> resident_input;
  bool state_uploaded = false;

  // Scratch for the cross-entropy reduction, sized at compile; a kernel that
  // allocated here would make the step uncapturable.
  std::uint64_t ce_scratch = 0;

  // CUDA-graph capture of the node launches (SONICBOOM_TRAIN_GRAPH).
  bool capture_enabled = false;
  bool capture_failed = false;
  void* graph_exec = nullptr;
  // The switch values the captured graph was recorded with. A captured launch
  // bakes in the per-execution switches it read (which adamw rate operand, and
  // which kernel provider), so a run under a different configuration must
  // invalidate the graph instead of replaying it.
  std::string captured_key;

  // The training step's own stream (0 when the build has no CUDA).
  std::uint64_t stream = 0;

  ~Impl() {
#ifdef SONICBOOM_USE_CUDA
    if (graph_exec != nullptr) detail::dev_graph_free(graph_exec);
    for (std::uint64_t b : slot_buffers)
      if (b != 0) detail::dev_free(b);
    if (ce_scratch != 0) detail::dev_free(ce_scratch);
    if (stream != 0) detail::dev_stream_destroy(stream);
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
  std::uint64_t ce_scratch = 0;
};

// SONICBOOM_ADAMW_LR_ATTR=1 restores the pre-lr-tensor behaviour: a sixth
// (learning-rate) operand is ignored and the node's `lr` attribute is used
// instead. It is the ablation switch for the lr-input form; unset or 0 leaves
// the operand in charge. Read per execution so a test can toggle it.
bool adamw_lr_attr_mode() {
  const char* e = std::getenv("SONICBOOM_ADAMW_LR_ATTR");
  return e && *e && !(e[0] == '0' && e[1] == '\0');
}

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
      if (detail::aten_direct_enabled("index_select") &&
          detail::aten_direct_index_select(buf(n.outputs[0]), buf(n.inputs[0]),
                                   buf(n.inputs[1]), ids[0], table[0],
                                   table[1], detail::dev_stream_current()))
        return true;
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
              : (detail::aten_direct_enabled("mul") &&
                         detail::aten_direct_mul(buf(n.outputs[0]), buf(n.inputs[0]),
                                         buf(n.inputs[1]), sa.data(), sb.data(),
                                         out.data(),
                                         static_cast<std::int64_t>(out.size()),
                                         n_out, detail::dev_stream_current())
                     ? true
                     : detail::graph_mul(buf(n.outputs[0]), buf(n.inputs[0]),
                                         buf(n.inputs[1]), sa.data(), sb.data(),
                                         out.data(),
                                         static_cast<std::int64_t>(out.size()),
                                         n_out));
      return ok ? true : fail("kernel launch failed");
    }
    case OpKind::Softmax: {
      if (!arity(1, 1)) return false;
      const auto in = shape(n.inputs[0]);
      const std::int64_t cols = in.empty() ? 1 : in.back();
      if (detail::aten_direct_enabled("softmax") &&
          detail::aten_direct_softmax(buf(n.outputs[0]), buf(n.inputs[0]),
                              numel(n.inputs[0]) / cols, cols,
                              detail::dev_stream_current()))
        return true;
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
                                       buf(n.inputs[1]), logits[0], logits[1],
                                       s.ce_scratch))
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
      if (n.outputs.size() != 3 ||
          (n.inputs.size() != 5 && n.inputs.size() != 6))
        return fail("expected 5 or 6 inputs and 3 results, got " +
                    std::to_string(n.inputs.size()) + " and " +
                    std::to_string(n.outputs.size()));
      if (tensor(n.inputs[4]).dtype != sx::DType::Int32)
        return fail("the step counter must be int32");
      const std::int64_t n_elem = numel(n.inputs[0]);
      const bool lr_operand =
          n.inputs.size() == 6 && !adamw_lr_attr_mode();
      const bool ok =
          lr_operand
              ? detail::graph_adamw_step_lr(
                    buf(n.outputs[0]), buf(n.outputs[1]), buf(n.outputs[2]),
                    buf(n.inputs[0]), buf(n.inputs[1]), buf(n.inputs[2]),
                    buf(n.inputs[3]), n_elem, buf(n.inputs[4]),
                    buf(n.inputs[5]), float_attr(n, "beta1", 0.9f),
                    float_attr(n, "beta2", 0.999f),
                    float_attr(n, "eps", 1e-8f),
                    float_attr(n, "weight_decay", 0.01f))
              : detail::graph_adamw_step(
                    buf(n.outputs[0]), buf(n.outputs[1]), buf(n.outputs[2]),
                    buf(n.inputs[0]), buf(n.inputs[1]), buf(n.inputs[2]),
                    buf(n.inputs[3]), n_elem, buf(n.inputs[4]),
                    float_attr(n, "lr", 1e-3f),
                    float_attr(n, "beta1", 0.9f),
                    float_attr(n, "beta2", 0.999f),
                    float_attr(n, "eps", 1e-8f),
                    float_attr(n, "weight_decay", 0.01f));
      if (!ok) return fail("kernel launch failed");
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
    if (n.outputs.size() != 3 ||
        (n.inputs.size() != 5 && n.inputs.size() != 6))
      return fail("expected 5 or 6 inputs and 3 results, got " +
                  std::to_string(n.inputs.size()) + " and " +
                  std::to_string(n.outputs.size()));
    if (dtype(n.inputs[4]) != sx::DType::Int32)
      return fail("the step counter must be int32");
    // The optional sixth operand is the learning rate: one float32 element.
    if (n.inputs.size() == 6) {
      if (dtype(n.inputs[5]) != sx::DType::Float32)
        return fail("the learning-rate input must be float32");
      if (numel(n.inputs[5]) != 1)
        return fail("the learning-rate input must hold exactly one element");
    }
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

#ifdef SONICBOOM_USE_CUDA
// Point the calling thread at the graph's own stream for a run, so an upload,
// the node launches and a download are ordered together and the launches can
// be captured.
class StreamGuard {
public:
  explicit StreamGuard(std::uint64_t stream)
      : previous_(detail::dev_stream_push(stream)) {}
  ~StreamGuard() { detail::dev_stream_pop(previous_); }
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

private:
  std::uint64_t previous_;
};

// Launch every node of one step. Capture is enabled and the graph is not yet
// instantiated: record the launches, instantiate once, and replay. After that
// the instantiated graph is launched directly. With capture off, or when the
// runtime refuses to capture this stream, the nodes launch one by one.
// The per-execution switches a node launch reads. Two of them change which
// kernel or which operand the step uses, so a graph captured under one
// configuration must not be replayed under another.
std::string launch_config_key() {
  const char* aten = std::getenv("SONICBOOM_TRAIN_ATEN");
  std::string key = aten ? aten : "";
  key += '|';
  key += adamw_lr_attr_mode() ? '1' : '0';
  return key;
}

bool execute_nodes(const Graph& graph,
                   const std::vector<std::uint64_t>& buffers,
                   std::uint64_t ce_scratch, bool capture_enabled,
                   bool* capture_failed, void** graph_exec,
                   std::string* captured_key, std::string* error) {
  const NodeContext ctx{graph, buffers, ce_scratch};
  auto launch_all = [&]() {
    for (const GraphNodeDesc& n : graph.nodes) {
      std::string why;
      if (!dispatch_node(ctx, n, &why)) {
        if (error) *error = why;
        return false;
      }
    }
    return true;
  };

  const std::string key = launch_config_key();
  if (*graph_exec != nullptr) {
    if (*captured_key == key) return detail::dev_graph_launch(*graph_exec);
    // Re-record under the new configuration; the old launches are not valid.
    detail::dev_graph_free(*graph_exec);
    *graph_exec = nullptr;
    captured_key->clear();
  }
  if (!capture_enabled || *capture_failed) return launch_all();

  if (!detail::dev_capture_begin()) {
    *capture_failed = true;
    return launch_all();
  }
  if (!launch_all()) {
    // Close the recording so the stream is usable again, then discard it.
    void* discarded = nullptr;
    detail::dev_capture_end(&discarded);
    detail::dev_graph_free(discarded);
    return false;
  }
  if (!detail::dev_capture_end(graph_exec)) {
    *capture_failed = true;
    return launch_all();
  }
  *captured_key = key;
  return detail::dev_graph_launch(*graph_exec);
}
#endif

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

  // --- buffer slots ---------------------------------------------------------
  //
  // The Training phase's BufferReusePass records, on each single-result node,
  // the buffer class its result belongs to (`buf`) and whether it may be
  // written over operand 0 (`inplace`). Class members are never live at the
  // same time, so they share one allocation; a tensor with no mark (an input,
  // a constant, one of a multi-result node's results) gets its own.
  //
  // An optimizer update is a pure function of (param, grad, m1, m2) written to
  // three results, and its kernel reads an element's operands before writing
  // that element, so a result may target its operand's buffer. When that
  // operand is a graph input whose last use is this update, the result is
  // written into the input in place: the parameter and moments then survive on
  // the device from step to step and need no upload.
  std::vector<int64_t> slot_of(g.tensors.size(), -1);
  std::vector<std::uint64_t> slot_bytes;
  auto new_slot = [&](std::uint64_t bytes) {
    slot_bytes.push_back(bytes);
    return static_cast<int64_t>(slot_bytes.size()) - 1;
  };
  std::map<int64_t, int64_t> class_slot;

  std::vector<int64_t> last_use(g.tensors.size(), -1);
  for (std::size_t i = 0; i < g.nodes.size(); ++i)
    for (TensorId in : g.nodes[i].inputs) last_use[in.value] = int64_t(i);

  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& n = g.nodes[i];
    if (n.outputs.size() != 1) continue;  // a multi-result node: own buffers
    const TensorId rid = n.outputs[0];
    const std::uint64_t bytes = g.tensors[rid.value].size_bytes;
    if (bytes == 0)
      return std::unexpected("tensor '" + g.tensors[rid.value].name +
                             "' has no storage");
    int64_t slot = -1;
    if (int_attr(n, "inplace", 0) == 1 && !n.inputs.empty() &&
        slot_of[n.inputs[0].value] >= 0)
      slot = slot_of[n.inputs[0].value];
    if (slot < 0) {
      const int64_t cls = int_attr(n, "buf", -1);
      if (cls >= 0) {
        auto it = class_slot.find(cls);
        if (it == class_slot.end())
          it = class_slot.emplace(cls, new_slot(bytes)).first;
        slot = it->second;
      }
    }
    if (slot < 0) slot = new_slot(bytes);
    slot_of[rid.value] = slot;
    slot_bytes[slot] = std::max(slot_bytes[slot], bytes);
  }
  for (std::size_t t = 0; t < g.tensors.size(); ++t) {
    if (slot_of[t] >= 0) continue;
    if (g.tensors[t].size_bytes == 0)
      return std::unexpected("tensor '" + g.tensors[t].name +
                             "' has no storage");
    slot_of[t] = new_slot(g.tensors[t].size_bytes);
  }

  s.resident_input.assign(g.inputs.size(), false);
  std::vector<int64_t> input_pos(g.tensors.size(), -1);
  for (std::size_t i = 0; i < g.inputs.size(); ++i)
    input_pos[g.inputs[i].value] = int64_t(i);
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& n = g.nodes[i];
    if (n.op != OpKind::AdamW || n.outputs.size() != 3 || n.inputs.size() < 4)
      continue;
    // AdamW: (param, grad, m1, m2, step[, lr]) -> (param', m1', m2').
    static constexpr int kCarry[3][2] = {{0, 0}, {2, 1}, {3, 2}};
    for (const auto& c : kCarry) {
      const TensorId in = n.inputs[c[0]];
      const TensorId out = n.outputs[c[1]];
      if (input_pos[in.value] < 0) continue;
      if (last_use[in.value] != int64_t(i)) continue;
      if (g.tensors[in.value].size_bytes != g.tensors[out.value].size_bytes)
        continue;
      slot_of[out.value] = slot_of[in.value];
      s.resident_input[static_cast<std::size_t>(input_pos[in.value])] = true;
    }
  }

  // One allocation per slot some tensor still uses. A result moved onto its
  // operand's slot above leaves its own slot unreferenced, so it is dropped.
  std::vector<int64_t> slot_remap(slot_bytes.size(), -1);
  s.buffers.assign(g.tensors.size(), 0);
  for (std::size_t t = 0; t < g.tensors.size(); ++t) {
    const int64_t sl = slot_of[t];
    if (slot_remap[sl] < 0) {
      const std::uint64_t b = detail::dev_alloc(slot_bytes[sl]);
      if (b == 0)
        return std::unexpected("cannot allocate " +
                               std::to_string(slot_bytes[sl]) +
                               " bytes for tensor '" + g.tensors[t].name + "'");
      slot_remap[sl] = int64_t(s.slot_buffers.size());
      s.slot_buffers.push_back(b);
    }
    s.buffers[t] = s.slot_buffers[slot_remap[sl]];
  }

  // Cross-entropy reads its whole row set before reducing; the scratch is a
  // compile-time allocation so that a step allocates nothing (CUDA graphs).
  std::int64_t ce_rows = 0;
  bool has_clip = false;
  for (const GraphNodeDesc& n : g.nodes) {
    if (n.op == OpKind::CrossEntropy && !n.inputs.empty()) {
      const auto& shape = g.tensors[n.inputs[0].value].shape;
      if (shape.size() == 2) ce_rows = std::max(ce_rows, shape[0]);
    }
    if (n.op == OpKind::ClipGradNorm) has_clip = true;
  }
  if (ce_rows > 0) {
    s.ce_scratch = detail::dev_alloc(static_cast<std::uint64_t>(ce_rows) *
                                     sizeof(float));
    if (s.ce_scratch == 0)
      return std::unexpected("cannot allocate the cross-entropy scratch");
  }
  // A few backward kernels take row scratch from the planner's device pool,
  // whose first acquire for a size is a cudaMalloc. A captured step may not
  // allocate, so every size those kernels will ask for is warmed here (the
  // release leaves the handle in the pool).
  for (const GraphNodeDesc& n : g.nodes) {
    if (n.inputs.empty()) continue;
    const auto& sh = g.tensors[n.inputs[0].value].shape;
    std::int64_t rows = 0;
    if (n.op == OpKind::CrossEntropyBwd && sh.size() == 2) {
      rows = sh[0];
    } else if (n.op == OpKind::RmsNormBwd && !sh.empty()) {
      std::int64_t total = 1;
      for (std::int64_t d : sh) total *= d;
      if (sh.back() > 0) rows = total / sh.back();
    }
    if (rows <= 0) continue;
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(rows) * sizeof(float);
    const std::uint64_t h = planner::device::acquire(bytes);
    if (h != 0) planner::device::release(h, bytes);
  }
  // A global-norm clip allocates its accumulator per execution, which a
  // captured step cannot do; its graph keeps the direct path.
  const char* train_graph = std::getenv("SONICBOOM_TRAIN_GRAPH");
  s.capture_enabled =
      !(train_graph != nullptr && train_graph[0] == '0' &&
        train_graph[1] == '\0') &&
      !has_clip;

#ifdef SONICBOOM_USE_CUDA
  // The step's stream. Constants are uploaded on it (synchronously, before the
  // first run), so nothing is left in flight on another stream.
  s.stream = detail::dev_stream_create();
  if (s.stream == 0)
    return std::unexpected("cannot create the training step's CUDA stream");
#endif
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
#ifdef SONICBOOM_USE_CUDA
    StreamGuard guard(s.stream);
    if (!detail::dev_upload(s.buffers[t.id.value], bytes.data(), bytes.size()) ||
        !detail::dev_sync())
      return std::unexpected("cannot upload constant '" + t.name + "'");
#else
    (void)bytes;
#endif
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
  }

  StreamGuard guard(s.stream);
  for (std::size_t i = 0; i < s.input_ids.size(); ++i) {
    if (!detail::dev_upload(s.buffers[s.input_ids[i].value], input_data[i],
                            s.input_info[i].num_bytes)) {
      if (error) *error = "cannot upload input " + std::to_string(i);
      return false;
    }
  }

  if (!execute_nodes(s.graph, s.buffers, s.ce_scratch, s.capture_enabled,
                     &s.capture_failed, &s.graph_exec, &s.captured_key, error))
    return false;

  if (!detail::dev_sync()) {
    if (error) *error = "device synchronization failed";
    return false;
  }
  s.state_uploaded = true;

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

bool DeviceTrainingGraph::input_is_resident(std::size_t index) const noexcept {
  return index < impl_->resident_input.size() && impl_->resident_input[index];
}

bool DeviceTrainingGraph::run_resident(
    const std::byte* const* input_data, const std::size_t* input_sizes,
    const std::size_t* download, std::size_t num_download,
    std::vector<std::vector<std::byte>>* outputs, std::string* error) {
#ifndef SONICBOOM_USE_CUDA
  if (error) *error = "SonicBoom was built without CUDA";
  return false;
#else
  Impl& s = *impl_;
  if (input_data == nullptr || input_sizes == nullptr || outputs == nullptr ||
      (num_download != 0 && download == nullptr)) {
    if (error) *error = "run_resident: null argument";
    return false;
  }
  if (s.input_info.size() != s.input_ids.size()) {
    if (error) *error = "run_resident: malformed compiled graph";
    return false;
  }
  for (std::size_t j = 0; j < num_download; ++j) {
    if (download[j] >= s.output_ids.size()) {
      if (error)
        *error = "run_resident: output index " +
                 std::to_string(download[j]) + " is out of range";
      return false;
    }
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
    const bool held = i < s.resident_input.size() && s.resident_input[i] &&
                      s.state_uploaded;
    if (!held && input_data[i] == nullptr && want != 0) {
      if (error)
        *error = "input " + std::to_string(i) + " ('" +
                 s.input_info[i].name + "') is null and not resident";
      return false;
    }
  }

  StreamGuard guard(s.stream);
  for (std::size_t i = 0; i < s.input_ids.size(); ++i) {
    if (i < s.resident_input.size() && s.resident_input[i] &&
        s.state_uploaded)
      continue;  // the previous step's update is already on the device
    if (!detail::dev_upload(s.buffers[s.input_ids[i].value], input_data[i],
                            s.input_info[i].num_bytes)) {
      if (error) *error = "cannot upload input " + std::to_string(i);
      return false;
    }
  }

  if (!execute_nodes(s.graph, s.buffers, s.ce_scratch, s.capture_enabled,
                     &s.capture_failed, &s.graph_exec, &s.captured_key, error))
    return false;

  if (!detail::dev_sync()) {
    if (error) *error = "device synchronization failed";
    return false;
  }
  s.state_uploaded = true;

  std::vector<std::vector<std::byte>> result(num_download);
  for (std::size_t j = 0; j < num_download; ++j) {
    const std::size_t i = download[j];
    const std::uint64_t bytes = s.output_info[i].num_bytes;
    result[j].resize(bytes);
    if (!detail::dev_download(s.buffers[s.output_ids[i].value], result[j].data(),
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
