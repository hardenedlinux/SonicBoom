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

// The CPU Boom executor: the CPU peer of training::DeviceTrainingGraph.
//
// A Blade (S-Expr) document is lowered to Boom IR (compiler::bladeToBoom), run
// through the same semantic-neutral pass phase the device executor uses
// (BufferReusePass), and handed back as a planner::Graph (boomToGraph). Every
// tensor then gets one host buffer (class sharing from the pass is honoured),
// constants are materialized once, and a run is: copy the bound inputs in, run
// the nodes in order, copy the outputs out.
//
// Node kernels are, in order of preference:
//   - the existing SonicBoom CPU kernels (nn::* / training::*),
//   - native-torch CPU kernels through the direct-call adapter in
//     core/layer1/adapter/aten_cpu.{h,cpp} (non-owning views, `_out` kernels,
//     no dispatcher) where no SonicBoom kernel exists (conv2d, max_pool2d,
//     dense matmul/bmm),
//   - small inline loops for pure data movement (reshape, permute, broadcast,
//     reduction, gather).
//
// This replaces the retired S-Expr → MLIR/linalg → LLVM JIT CPU path: the
// executable facade (sx::Executable) is unchanged, the lowering behind it is
// Bloom IR.

#include <sonicboom/sx/exec.h>

#include "sonicboom/compiler/BladeToBoom.h"
#include "sonicboom/compiler/BoomConversion.h"
#include "sonicboom/compiler/BoomPasses.h"
#include "sonicboom/nn/activation.h"
#include "sonicboom/nn/elementwise.h"
#include "sonicboom/nn/norm.h"
#include "sonicboom/nn/rope.h"
#include "sonicboom/nn/softmax.h"
#include "sonicboom/planner/graph.h"
#include "sonicboom/planner/ir_passes.h"
#include "sonicboom/planner/op_kind.h"
#include "sonicboom/training/adamw.h"
#include "sonicboom/training/backward.h"
#include "sonicboom/training/clip_grad_norm.h"

#include "../../layer1/adapter/aten_cpu.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sonicboom::sx {

namespace {

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

const Attribute* find_attr(const GraphNodeDesc& node, std::string_view name) {
  for (const Attribute& a : node.attributes)
    if (a.name == name) return &a;
  return nullptr;
}

std::int64_t int_attr(const GraphNodeDesc& node, std::string_view name,
                      std::int64_t fallback) {
  const Attribute* a = find_attr(node, name);
  if (a == nullptr) return fallback;
  if (const auto* v = std::get_if<std::int64_t>(&a->value)) return *v;
  if (const auto* v = std::get_if<double>(&a->value))
    return static_cast<std::int64_t>(*v);
  if (const auto* v = std::get_if<bool>(&a->value)) return *v ? 1 : 0;
  return fallback;
}

float float_attr(const GraphNodeDesc& node, std::string_view name,
                 float fallback) {
  const Attribute* a = find_attr(node, name);
  if (a == nullptr) return fallback;
  if (const auto* v = std::get_if<double>(&a->value))
    return static_cast<float>(*v);
  if (const auto* v = std::get_if<std::int64_t>(&a->value))
    return static_cast<float>(*v);
  return fallback;
}

std::vector<std::int64_t> ints_attr(const GraphNodeDesc& node,
                                    std::string_view name) {
  const Attribute* a = find_attr(node, name);
  if (a == nullptr) return {};
  if (const auto* v = std::get_if<std::vector<std::int64_t>>(&a->value))
    return *v;
  if (const auto* v = std::get_if<std::int64_t>(&a->value)) return {*v};
  return {};
}

std::vector<std::int64_t> ints_attr_or(const GraphNodeDesc& node,
                                       std::string_view name,
                                       std::vector<std::int64_t> fallback) {
  const Attribute* a = find_attr(node, name);
  if (a == nullptr) return fallback;
  if (const auto* v = std::get_if<std::vector<std::int64_t>>(&a->value))
    return *v;
  if (const auto* v = std::get_if<std::int64_t>(&a->value)) return {*v};
  return fallback;
}

std::string node_text(const GraphNodeDesc& node) {
  return "node #" + std::to_string(node.id.value) + " '" + node.name + "' (" +
         planner::op_kind_name(node.op) + ")";
}

// --- constants --------------------------------------------------------------

// Host bytes of a parameter. An inline parameter is converted from its values;
// an external one takes the bytes load_external_weights already resolved.
bool parameter_bytes(const Parameter& p,
                     const std::unordered_map<std::string, Bytes>& weights,
                     std::vector<std::byte>* out, std::string* error) {
  const std::int64_t n = numel(p.type);
  if (const auto* external = std::get_if<ExternalData>(&p.data)) {
    (void)external;
    auto it = weights.find(p.name);
    if (it == weights.end()) {
      *error = "parameter '" + p.name +
               "' is external data but no sidecar bytes were supplied";
      return false;
    }
    const std::uint64_t expected =
        static_cast<std::uint64_t>(n) * dtype_size(p.type.dtype);
    if (it->second.size() != expected) {
      *error = "parameter '" + p.name + "' holds " +
               std::to_string(it->second.size()) + " bytes but its type has " +
               std::to_string(expected);
      return false;
    }
    *out = it->second;
    return true;
  }
  const auto* values = std::get_if<ValuesData>(&p.data);
  if (values == nullptr) {
    *error = "parameter '" + p.name + "' has no data";
    return false;
  }
  if (static_cast<std::int64_t>(values->values.size()) != n) {
    *error = "parameter '" + p.name + "' holds " +
             std::to_string(values->values.size()) + " values but its type has " +
             std::to_string(n);
    return false;
  }

  switch (p.type.dtype) {
  case DType::Float32: {
    out->resize(static_cast<std::size_t>(n) * sizeof(float));
    auto* dst = reinterpret_cast<float*>(out->data());
    for (std::int64_t i = 0; i < n; ++i) {
      if (const auto* d = std::get_if<double>(&values->values[i]))
        dst[i] = static_cast<float>(*d);
      else
        dst[i] =
            static_cast<float>(std::get<std::int64_t>(values->values[i]));
    }
    return true;
  }
  case DType::Int32: {
    out->resize(static_cast<std::size_t>(n) * sizeof(std::int32_t));
    auto* dst = reinterpret_cast<std::int32_t*>(out->data());
    for (std::int64_t i = 0; i < n; ++i) {
      if (const auto* v = std::get_if<std::int64_t>(&values->values[i]))
        dst[i] = static_cast<std::int32_t>(*v);
      else
        dst[i] =
            static_cast<std::int32_t>(std::get<double>(values->values[i]));
    }
    return true;
  }
  case DType::Int64: {
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
             std::string(dtype_name(p.type.dtype)) +
             ", which the CPU executor cannot materialize from values";
    return false;
  }
}

// --- inline kernels (data movement and simple elementwise) ------------------

void add_broadcast_kernel(const float* a, const float* b, float* out,
                          const std::int64_t* dims,
                          const std::int64_t* sa, const std::int64_t* sb,
                          int64_t rank, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t rem = i, oa = 0, ob = 0;
    for (int64_t d = rank; d-- > 0;) {
      const std::int64_t c = rem % dims[d];
      rem /= dims[d];
      oa += c * sa[d];
      ob += c * sb[d];
    }
    out[i] = a[oa] + b[ob];
  }
}

void mul_broadcast_kernel(const float* a, const float* b, float* out,
                          const std::int64_t* dims,
                          const std::int64_t* sa, const std::int64_t* sb,
                          int64_t rank, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t rem = i, oa = 0, ob = 0;
    for (int64_t d = rank; d-- > 0;) {
      const std::int64_t c = rem % dims[d];
      rem /= dims[d];
      oa += c * sa[d];
      ob += c * sb[d];
    }
    out[i] = a[oa] * b[ob];
  }
}

void relu_kernel(const float* x, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i)
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// Sum over the axes whose bit is set in MASK, into the row-major output. With
// keepdims the output keeps the input rank (size-1 reduced axes), so surviving
// input axis d is output axis d; without it they are compacted to the left.
void reduce_sum_kernel(const float* x, float* out,
                       const std::vector<std::int64_t>& in_shape,
                       const std::vector<std::int64_t>& out_shape,
                       std::uint64_t mask, std::int64_t n_in,
                       bool mean) {
  const std::size_t rank = in_shape.size();
  const auto out_strides = row_major_strides(out_shape);
  const std::int64_t n_out = numel_of(out_shape);
  for (std::int64_t i = 0; i < n_out; ++i) out[i] = 0.0f;
  const bool keepdims = out_shape.size() == in_shape.size();
  std::vector<std::size_t> out_axis_of(rank, 0);
  std::size_t oi = 0;
  std::int64_t count = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    if ((mask >> d) & 1u) {
      count *= in_shape[d];
    } else {
      out_axis_of[d] = keepdims ? d : oi++;
    }
  }
  for (std::int64_t i = 0; i < n_in; ++i) {
    std::int64_t rem = i, idx = 0;
    for (std::size_t d = rank; d-- > 0;) {
      const std::int64_t c = rem % in_shape[d];
      rem /= in_shape[d];
      if (!((mask >> d) & 1u)) idx += c * out_strides[out_axis_of[d]];
    }
    out[idx] += x[i];
  }
  if (mean && count > 1) {
    const float inv = 1.0f / static_cast<float>(count);
    for (std::int64_t i = 0; i < n_out; ++i) out[i] *= inv;
  }
}

// Row gather from a [rows, dim] table with int32 indices.
void gather_rows_kernel(const float* table, const std::int32_t* ids,
                        float* out, std::int64_t rows, std::int64_t dim) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::int32_t id = ids[r];
    if (id < 0) continue;
    std::memcpy(out + r * dim, table + static_cast<std::int64_t>(id) * dim,
                static_cast<std::size_t>(dim) * sizeof(float));
  }
}

// out[dst_strides] = x[src_strides] over the output's shape (permute).
void permute_kernel(const float* x, float* out,
                    const std::vector<std::int64_t>& out_shape,
                    const std::vector<std::int64_t>& src_strides,
                    std::int64_t n) {
  const std::size_t rank = out_shape.size();
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t rem = i, src = 0;
    for (std::size_t d = rank; d-- > 0;) {
      const std::int64_t c = rem % out_shape[d];
      rem /= out_shape[d];
      src += c * src_strides[d];
    }
    out[i] = x[src];
  }
}

// NCHW pad copy: FILL outside the source, source contents inside. The
// destination is [N, C, H + pad_t + pad_b, W + pad_l + pad_r].
void pad_nchw_kernel(float* dst, const float* src, std::int64_t N,
                     std::int64_t C, std::int64_t H, std::int64_t W,
                     std::int64_t pad_t, std::int64_t pad_l,
                     std::int64_t pad_b, std::int64_t pad_r, float fill) {
  const std::int64_t hp = H + pad_t + pad_b;
  const std::int64_t wp = W + pad_l + pad_r;
  const std::int64_t plane = hp * wp;
  for (std::int64_t nc = 0; nc < N * C; ++nc) {
    float* d = dst + nc * plane;
    std::fill(d, d + plane, fill);
    const float* s = src + nc * (H * W);
    for (std::int64_t y = 0; y < H; ++y)
      std::memcpy(d + (y + pad_t) * wp + pad_l, s + y * W,
                  static_cast<std::size_t>(W) * sizeof(float));
  }
}

// Cross-entropy forward: mean over rows of -log softmax(logits)[target].
float cross_entropy_loss(const float* logits, const std::int32_t* targets,
                         std::int64_t rows, std::int64_t cols) {
  double total = 0.0;
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x = logits + r * cols;
    double max = -std::numeric_limits<double>::infinity();
    for (std::int64_t c = 0; c < cols; ++c)
      max = std::max(max, static_cast<double>(x[c]));
    double sum = 0.0;
    for (std::int64_t c = 0; c < cols; ++c)
      sum += std::exp(static_cast<double>(x[c]) - max);
    const std::int64_t t = targets[r];
    if (t < 0 || t >= cols) continue;
    total += -(static_cast<double>(x[t]) - max - std::log(sum));
  }
  return static_cast<float>(rows > 0 ? total / static_cast<double>(rows) : 0.0);
}

// --- node dispatch ----------------------------------------------------------

// What a node launch reads: the graph (shapes, dtypes), one host buffer per
// tensor id, and the compile-time scratch.
struct NodeContext {
  const planner::Graph& graph;
  const std::vector<std::byte*>& buffers;
  std::byte* pad_scratch = nullptr;
  std::uint64_t pad_scratch_bytes = 0;
  std::int64_t* index_scratch = nullptr;
  std::uint64_t index_scratch_entries = 0;
  std::byte* gemm_scratch = nullptr;
  std::uint64_t gemm_scratch_bytes = 0;
  // ReduceMean: the axes constant, resolved at compile for each node.
  const std::vector<std::int64_t>* reduce_axes = nullptr;
};

bool dispatch_node(const NodeContext& s, const GraphNodeDesc& n,
                   std::string* error) {
  auto f32 = [&](TensorId id) {
    return reinterpret_cast<float*>(s.buffers[id.value]);
  };
  auto f32c = [&](TensorId id) {
    return reinterpret_cast<const float*>(s.buffers[id.value]);
  };
  auto i32c = [&](TensorId id) {
    return reinterpret_cast<const std::int32_t*>(s.buffers[id.value]);
  };
  auto shape = [&](TensorId id) { return s.graph.tensors[id.value].shape; };
  auto numel = [&](TensorId id) { return numel_of(shape(id)); };
  auto bytes = [&](TensorId id) { return s.graph.tensors[id.value].size_bytes; };
  auto fail = [&](const std::string& why) {
    *error = node_text(n) + ": " + why;
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

  switch (n.op) {
  // --- training forward vocabulary ----------------------------------------
  case OpKind::Matmul: {
    if (!arity(2, 1)) return false;
    const auto a = shape(n.inputs[0]);
    const auto b = shape(n.inputs[1]);
    if (!cpu::detail::aten_cpu_mm(f32(n.outputs[0]), f32c(n.inputs[0]),
                                  f32c(n.inputs[1]), a[0], a[1], b[1]))
      return fail("matmul kernel failed");
    return true;
  }
  case OpKind::Bmm: {
    if (!arity(2, 1)) return false;
    const auto a = shape(n.inputs[0]);
    const auto b = shape(n.inputs[1]);
    if (!cpu::detail::aten_cpu_bmm(f32(n.outputs[0]), f32c(n.inputs[0]),
                                   f32c(n.inputs[1]), a[0], a[1], a[2], b[2]))
      return fail("bmm kernel failed");
    return true;
  }
  case OpKind::Permute: {
    if (!arity(1, 1)) return false;
    const auto dims = ints_attr(n, "dims");
    const auto in = shape(n.inputs[0]);
    const auto out = shape(n.outputs[0]);
    const auto src = row_major_strides(in);
    std::vector<std::int64_t> src_strides;
    for (std::int64_t d : dims) src_strides.push_back(src[d]);
    permute_kernel(f32c(n.inputs[0]), f32(n.outputs[0]), out, src_strides,
                   numel(n.outputs[0]));
    return true;
  }
  case OpKind::IndexSelect: {
    if (!arity(2, 1)) return false;
    const auto table = shape(n.inputs[0]);
    const auto ids = shape(n.inputs[1]);
    gather_rows_kernel(f32c(n.inputs[0]), i32c(n.inputs[1]),
                       f32(n.outputs[0]), ids[0], table[1]);
    return true;
  }
  case OpKind::ReduceSum: {
    if (!arity(1, 1)) return false;
    const auto in = shape(n.inputs[0]);
    std::uint64_t mask = 0;
    for (std::int64_t a : ints_attr(n, "axes")) {
      if (a < 0) a += static_cast<std::int64_t>(in.size());
      mask |= std::uint64_t(1) << static_cast<unsigned>(a);
    }
    reduce_sum_kernel(f32c(n.inputs[0]), f32(n.outputs[0]), in,
                      shape(n.outputs[0]), mask, numel(n.inputs[0]), false);
    return true;
  }
  case OpKind::Reshape: {
    if (!arity(1, 1)) return false;
    if (numel(n.inputs[0]) != numel(n.outputs[0]))
      return fail("element count differs");
    std::memcpy(s.buffers[n.outputs[0].value], s.buffers[n.inputs[0].value],
                static_cast<std::size_t>(bytes(n.inputs[0])));
    return true;
  }
  case OpKind::Add:
  case OpKind::Mul: {
    if (!arity(2, 1)) return false;
    const auto out = shape(n.outputs[0]);
    const auto sa = broadcast_strides(out, shape(n.inputs[0]));
    const auto sb = broadcast_strides(out, shape(n.inputs[1]));
    const std::int64_t n_out = numel(n.outputs[0]);
    const std::int64_t rank = static_cast<std::int64_t>(out.size());
    if (n.op == OpKind::Add)
      add_broadcast_kernel(f32c(n.inputs[0]), f32c(n.inputs[1]),
                           f32(n.outputs[0]), out.data(), sa.data(),
                           sb.data(), rank, n_out);
    else
      mul_broadcast_kernel(f32c(n.inputs[0]), f32c(n.inputs[1]),
                           f32(n.outputs[0]), out.data(), sa.data(),
                           sb.data(), rank, n_out);
    return true;
  }
  case OpKind::Softmax: {
    if (!arity(1, 1)) return false;
    const auto in = shape(n.inputs[0]);
    // Softmax along one axis (the ONNX `axis` attribute, aten::_softmax's
    // single-dim semantics — not the opset<13 flatten-at-axis form). Default
    // -1 is the last axis, which is also the CM training graphs' layout.
    const std::int64_t rank = static_cast<std::int64_t>(in.size());
    std::int64_t axis = int_attr(n, "axis", -1);
    if (axis < 0) axis += rank;
    if (rank == 0) {
      if (axis != 0 && axis != -1) return fail("softmax axis out of range");
      axis = 0;
    }
    if (axis < 0 || axis >= rank)
      return fail("softmax axis out of range for rank " +
                  std::to_string(rank));
    const std::int64_t cols = in[static_cast<std::size_t>(axis)];
    std::int64_t inner = 1;
    for (std::int64_t d = axis + 1; d < rank; ++d)
      inner *= in[static_cast<std::size_t>(d)];
    const std::int64_t rows = numel(n.inputs[0]) / cols;
    const float* x = f32c(n.inputs[0]);
    float* y = f32(n.outputs[0]);
    // nn::softmax rejects a row whose maximum is -inf (every entry masked, or
    // non-finite input). The device kernel has no such rejection: it computes
    // exp(x - max) with max = -inf and propagates NaN. The executor matches the
    // device, so a diverged training graph fails the numerical comparison
    // identically on both backends instead of erroring here alone.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (inner == 1) {
      for (std::int64_t r = 0; r < rows; ++r) {
        if (!nn::softmax(std::span<const float>(x + r * cols, cols),
                         std::span<float>(y + r * cols, cols)))
          std::fill_n(y + r * cols, cols, nan);
      }
    } else {
      // Non-innermost axis: gather each strided slice, softmax it, scatter it
      // back. `outer` counts the elements before the axis.
      const std::int64_t outer_rows = rows / inner;
      std::vector<float> row(static_cast<std::size_t>(cols));
      for (std::int64_t o = 0; o < outer_rows; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
          float* dst = y + o * cols * inner + i;
          const float* src = x + o * cols * inner + i;
          for (std::int64_t k = 0; k < cols; ++k)
            row[static_cast<std::size_t>(k)] = src[k * inner];
          if (!nn::softmax(std::span<const float>(row),
                           std::span<float>(row)))
            std::fill(row.begin(), row.end(), nan);
          for (std::int64_t k = 0; k < cols; ++k)
            dst[k * inner] = row[static_cast<std::size_t>(k)];
        }
      }
    }
    return true;
  }
  case OpKind::RmsNorm: {
    if (!arity(2, 1)) return false;
    const auto x = shape(n.inputs[0]);
    const std::int64_t dim = x.empty() ? 1 : x.back();
    const std::int64_t rows = numel(n.inputs[0]) / dim;
    const float* in = f32c(n.inputs[0]);
    const float* w = f32c(n.inputs[1]);
    float* out = f32(n.outputs[0]);
    const float eps = float_attr(n, "eps", 1e-6f);
    for (std::int64_t r = 0; r < rows; ++r)
      if (!nn::rms_norm(std::span<const float>(in + r * dim, dim),
                        std::span<const float>(w, dim), eps,
                        std::span<float>(out + r * dim, dim)))
        return fail("rms_norm kernel failed");
    return true;
  }
  case OpKind::Rope: {
    if (!arity(1, 1)) return false;
    const auto x = shape(n.inputs[0]);
    const std::int64_t head_dim = int_attr(n, "head_dim", 0);
    const std::int64_t rows = numel(n.inputs[0]) / x.back();
    std::memcpy(s.buffers[n.outputs[0].value], s.buffers[n.inputs[0].value],
                static_cast<std::size_t>(bytes(n.inputs[0])));
    float* out = f32(n.outputs[0]);
    const std::uint64_t pos0 =
        static_cast<std::uint64_t>(int_attr(n, "pos0", 0));
    const float theta = float_attr(n, "theta_base", 10000.0f);
    const float freq_scale = float_attr(n, "freq_scale", 1.0f);
    for (std::int64_t r = 0; r < rows; ++r)
      if (!nn::rope_neox_heads(
              std::span<float>(out + r * x.back(), x.back()),
              static_cast<std::uint64_t>(head_dim),
              pos0 + static_cast<std::uint64_t>(r), theta, freq_scale, {}))
        return fail("rope kernel failed");
    return true;
  }
  case OpKind::SiluMul: {
    if (!arity(2, 1)) return false;
    const std::int64_t n_elem = numel(n.outputs[0]);
    float* out = f32(n.outputs[0]);
    if (!nn::silu(std::span<const float>(f32c(n.inputs[0]), n_elem),
                  std::span<float>(out, n_elem)))
      return fail("silu kernel failed");
    if (!nn::mul(std::span<const float>(out, n_elem),
                 std::span<const float>(f32c(n.inputs[1]), n_elem),
                 std::span<float>(out, n_elem)))
      return fail("mul kernel failed");
    return true;
  }
  case OpKind::CrossEntropy: {
    if (!arity(2, 1)) return false;
    const auto logits = shape(n.inputs[0]);
    f32(n.outputs[0])[0] = cross_entropy_loss(f32c(n.inputs[0]),
                                              i32c(n.inputs[1]), logits[0],
                                              logits[1]);
    return true;
  }
  case OpKind::AdamW: {
    const std::size_t n_in = n.inputs.size();
    if (n.outputs.size() != 3 || (n_in != 5 && n_in != 6))
      return fail("expected 5 or 6 inputs and 3 results");
    const std::int64_t n_elem = numel(n.inputs[0]);
    std::memcpy(s.buffers[n.outputs[0].value], s.buffers[n.inputs[0].value],
                static_cast<std::size_t>(bytes(n.inputs[0])));
    std::memcpy(s.buffers[n.outputs[1].value], s.buffers[n.inputs[2].value],
                static_cast<std::size_t>(bytes(n.inputs[2])));
    std::memcpy(s.buffers[n.outputs[2].value], s.buffers[n.inputs[3].value],
                static_cast<std::size_t>(bytes(n.inputs[3])));
    training::AdamWConfig cfg;
    cfg.lr = n_in == 6 ? f32c(n.inputs[5])[0] : float_attr(n, "lr", 1e-3f);
    cfg.beta1 = float_attr(n, "beta1", 0.9f);
    cfg.beta2 = float_attr(n, "beta2", 0.999f);
    cfg.eps = float_attr(n, "eps", 1e-8f);
    cfg.weight_decay = float_attr(n, "weight_decay", 0.01f);
    const std::uint64_t step =
        static_cast<std::uint64_t>(i32c(n.inputs[4])[0]);
    training::adamw_f32(f32(n.outputs[0]), f32c(n.inputs[1]),
                        f32(n.outputs[1]), f32(n.outputs[2]),
                        static_cast<std::size_t>(n_elem), cfg, step);
    return true;
  }
  case OpKind::ClipGradNorm: {
    if (n.inputs.empty() || n.outputs.size() != n.inputs.size())
      return fail("expects one result per gradient operand");
    std::vector<float*> grads;
    std::vector<std::size_t> sizes;
    for (TensorId in : n.inputs) {
      grads.push_back(f32(in));
      sizes.push_back(static_cast<std::size_t>(numel(in)));
    }
    const double max_norm = double(float_attr(n, "max_norm", 1.0f));
    const double norm = training::grad_global_norm(grads, sizes);
    const double scale =
        norm > max_norm ? max_norm / (norm + 1e-6) : 1.0;
    for (std::size_t i = 0; i < n.inputs.size(); ++i) {
      const float* src = f32c(n.inputs[i]);
      float* dst = f32(n.outputs[i]);
      const std::size_t count = sizes[i];
      for (std::size_t j = 0; j < count; ++j)
        dst[j] = static_cast<float>(static_cast<double>(src[j]) * scale);
    }
    return true;
  }

  // --- fused adjoints ------------------------------------------------------
  case OpKind::MatmulBwd: {
    if (!arity(3, 2)) return false;
    const auto a = shape(n.inputs[0]);
    const auto b = shape(n.inputs[1]);
    training::matmul_bwd_f32(f32c(n.inputs[0]), f32c(n.inputs[1]),
                             f32c(n.inputs[2]), a[0], a[1], b[1],
                             f32(n.outputs[0]), f32(n.outputs[1]));
    return true;
  }
  case OpKind::SoftmaxBwd: {
    if (!arity(2, 1)) return false;
    const auto y = shape(n.inputs[0]);
    const std::int64_t cols = y.empty() ? 1 : y.back();
    training::softmax_bwd_f32(f32c(n.inputs[0]), f32c(n.inputs[1]),
                              numel(n.inputs[0]) / cols, cols,
                              f32(n.outputs[0]));
    return true;
  }
  case OpKind::CrossEntropyBwd: {
    if (!arity(2, 2)) return false;
    const auto logits = shape(n.inputs[0]);
    f32(n.outputs[0])[0] = training::cross_entropy_bwd_f32(
        f32c(n.inputs[0]), i32c(n.inputs[1]), logits[0], logits[1],
        f32(n.outputs[1]));
    return true;
  }
  case OpKind::RmsNormBwd: {
    if (!arity(3, 2)) return false;
    const auto x = shape(n.inputs[0]);
    const std::int64_t dim = x.empty() ? 1 : x.back();
    training::rmsnorm_bwd_f32(f32c(n.inputs[0]), f32c(n.inputs[1]),
                              f32c(n.inputs[2]), numel(n.inputs[0]) / dim,
                              dim, float_attr(n, "eps", 1e-6f),
                              f32(n.outputs[0]), f32(n.outputs[1]));
    return true;
  }
  case OpKind::RopeBwd: {
    if (!arity(1, 1)) return false;
    const auto x = shape(n.inputs[0]);
    const std::int64_t head_dim = int_attr(n, "head_dim", 0);
    const std::int64_t heads = x.back() / head_dim;
    const std::int64_t rows = numel(n.inputs[0]) / x.back();
    training::rope_bwd_f32(
        f32c(n.inputs[0]), rows, heads, head_dim,
        static_cast<std::uint64_t>(int_attr(n, "pos0", 0)),
        float_attr(n, "theta_base", 10000.0f),
        float_attr(n, "freq_scale", 1.0f), f32(n.outputs[0]));
    return true;
  }
  case OpKind::SiluMulBwd: {
    if (!arity(3, 2)) return false;
    training::silu_mul_bwd_f32(f32c(n.inputs[0]), f32c(n.inputs[1]),
                               f32c(n.inputs[2]), numel(n.inputs[0]),
                               f32(n.outputs[0]), f32(n.outputs[1]));
    return true;
  }
  case OpKind::EmbeddingBwd: {
    if (!arity(2, 1)) return false;
    const auto dy = shape(n.inputs[1]);
    training::embedding_bwd_f32(i32c(n.inputs[0]), f32c(n.inputs[1]), dy[0],
                                dy[1], int_attr(n, "vocab", 0),
                                f32(n.outputs[0]));
    return true;
  }

  // --- frozen S-Expr v0.1 CNN vocabulary ----------------------------------
  case OpKind::Relu: {
    if (!arity(1, 1)) return false;
    relu_kernel(f32c(n.inputs[0]), f32(n.outputs[0]), numel(n.outputs[0]));
    return true;
  }
  case OpKind::Conv: {
    if (!arity(3, 1)) return false;
    const auto x = shape(n.inputs[0]);
    const auto w = shape(n.inputs[1]);
    const auto out = shape(n.outputs[0]);
    const auto pads = ints_attr_or(n, "pads", {0, 0, 0, 0});
    const auto strides = ints_attr_or(n, "strides", {1, 1});
    const std::int64_t pt = pads[0], pl = pads[1], pb = pads[2], pr = pads[3];
    const std::int64_t sh = strides[0], sw = strides[1];
    const float* src = f32c(n.inputs[0]);
    std::int64_t ih = x[2], iw = x[3];
    const std::int64_t padded_bytes =
        static_cast<std::int64_t>(x[0] * x[1] * (x[2] + pt + pb) *
                                  (x[3] + pl + pr)) *
        static_cast<std::int64_t>(sizeof(float));
    if (padded_bytes > static_cast<std::int64_t>(s.pad_scratch_bytes))
      return fail("padded input exceeds the compile-time scratch");
    if (pt || pl || pb || pr) {
      pad_nchw_kernel(reinterpret_cast<float*>(s.pad_scratch), src, x[0], x[1],
                      x[2], x[3], pt, pl, pb, pr, 0.0f);
      src = reinterpret_cast<const float*>(s.pad_scratch);
      ih += pt + pb;
      iw += pl + pr;
    }
    if (!cpu::detail::aten_cpu_conv2d(
            f32(n.outputs[0]), src, f32c(n.inputs[1]), f32c(n.inputs[2]), x[0],
            x[1], ih, iw, w[0], w[2], w[3], sh, sw, out[2], out[3]))
      return fail("conv2d kernel failed");
    return true;
  }
  case OpKind::MaxPool: {
    if (!arity(1, 1)) return false;
    const auto x = shape(n.inputs[0]);
    const auto out = shape(n.outputs[0]);
    const auto kernel = ints_attr(n, "kernel_shape");
    const auto pads = ints_attr_or(n, "pads", {0, 0, 0, 0});
    const auto strides = ints_attr_or(n, "strides", {1, 1});
    const std::int64_t pt = pads[0], pl = pads[1], pb = pads[2], pr = pads[3];
    const float* src = f32c(n.inputs[0]);
    std::int64_t ih = x[2], iw = x[3];
    const std::int64_t padded_bytes =
        static_cast<std::int64_t>(x[0] * x[1] * (x[2] + pt + pb) *
                                  (x[3] + pl + pr)) *
        static_cast<std::int64_t>(sizeof(float));
    if (padded_bytes > static_cast<std::int64_t>(s.pad_scratch_bytes))
      return fail("padded input exceeds the compile-time scratch");
    if (pt || pl || pb || pr) {
      pad_nchw_kernel(reinterpret_cast<float*>(s.pad_scratch), src, x[0], x[1],
                      x[2], x[3], pt, pl, pb, pr,
                      -std::numeric_limits<float>::infinity());
      src = reinterpret_cast<const float*>(s.pad_scratch);
      ih += pt + pb;
      iw += pl + pr;
    }
    const std::int64_t out_elems = out[0] * out[1] * out[2] * out[3];
    if (out_elems > s.index_scratch_entries)
      return fail("pooling output exceeds the compile-time index scratch");
    if (!cpu::detail::aten_cpu_max_pool2d(
            f32(n.outputs[0]), s.index_scratch, src, x[0], x[1], ih, iw,
            kernel[0], kernel[1], strides[0], strides[1], out[2], out[3]))
      return fail("max_pool kernel failed");
    return true;
  }
  case OpKind::ReduceMean: {
    if (n.inputs.size() < 2 || n.outputs.size() != 1)
      return fail("expects an input, an axes input and one result");
    if (s.reduce_axes == nullptr) return fail("axes were not resolved");
    const auto in = shape(n.inputs[0]);
    std::uint64_t mask = 0;
    for (std::int64_t a : *s.reduce_axes) {
      if (a < 0) a += static_cast<std::int64_t>(in.size());
      mask |= std::uint64_t(1) << static_cast<unsigned>(a);
    }
    reduce_sum_kernel(f32c(n.inputs[0]), f32(n.outputs[0]), in,
                      shape(n.outputs[0]), mask, numel(n.inputs[0]), true);
    return true;
  }
  case OpKind::Gemm: {
    if (n.inputs.size() < 2 || n.inputs.size() > 3)
      return fail("gemm expects 2 or 3 inputs");
    const auto out = shape(n.outputs[0]);
    const auto a = shape(n.inputs[0]);
    const auto b = shape(n.inputs[1]);
    const bool trans_a = int_attr(n, "transA", 0) != 0;
    const bool trans_b = int_attr(n, "transB", 0) != 0;
    const std::int64_t M = out[0], N = out[1];
    const std::int64_t K = trans_a ? a[0] : a[1];
    const float* a_ptr = f32c(n.inputs[0]);
    const float* b_ptr = f32c(n.inputs[1]);
    float* a_scratch = nullptr;
    float* b_scratch = nullptr;
    if (trans_a) {
      if (static_cast<std::uint64_t>(M * K) * sizeof(float) >
          s.gemm_scratch_bytes)
        return fail("transposed A exceeds the compile-time scratch");
      a_scratch = reinterpret_cast<float*>(s.gemm_scratch);
      for (std::int64_t k = 0; k < K; ++k)
        for (std::int64_t m = 0; m < M; ++m)
          a_scratch[m * K + k] = a_ptr[k * M + m];
      a_ptr = a_scratch;
    }
    if (trans_b) {
      const std::uint64_t a_bytes =
          trans_a ? static_cast<std::uint64_t>(M * K) * sizeof(float) : 0u;
      if (a_bytes + static_cast<std::uint64_t>(K * N) * sizeof(float) >
          s.gemm_scratch_bytes)
        return fail("transposed B exceeds the compile-time scratch");
      b_scratch = reinterpret_cast<float*>(s.gemm_scratch + a_bytes);
      for (std::int64_t m = 0; m < N; ++m)
        for (std::int64_t k = 0; k < K; ++k)
          b_scratch[k * N + m] = b_ptr[m * K + k];
      b_ptr = b_scratch;
    }
    if (!cpu::detail::aten_cpu_mm(f32(n.outputs[0]), a_ptr, b_ptr, M, K, N))
      return fail("gemm kernel failed");
    const double alpha = double(float_attr(n, "alpha", 1.0f));
    const double beta = double(float_attr(n, "beta", 1.0f));
    float* out_ptr = f32(n.outputs[0]);
    if (n.inputs.size() == 3) {
      const float* bias = f32c(n.inputs[2]);
      const std::int64_t bias_n = numel(n.inputs[2]);
      if (bias_n != N && bias_n != M * N)
        return fail("gemm bias must broadcast along the trailing axis");
      for (std::int64_t i = 0; i < M * N; ++i) {
        const std::int64_t b = bias_n == N ? i % N : i;
        out_ptr[i] = static_cast<float>(
            alpha * static_cast<double>(out_ptr[i]) +
            beta * static_cast<double>(bias[b]));
      }
    } else if (alpha != 1.0) {
      for (std::int64_t i = 0; i < M * N; ++i)
        out_ptr[i] = static_cast<float>(alpha * double(out_ptr[i]));
    }
    return true;
  }

  default:
    *error = node_text(n) + ": not supported by the CPU Boom executor";
    return false;
  }
}

// The graph-level half of a node's checks: what the executor supports at all,
// and the property of shapes, attributes and dtypes its kernels assume. These
// are facts about the compiled graph, so they are rejected at compile time.
bool validate_node(const planner::Graph& g, const GraphNodeDesc& n, std::string* error) {
  auto shape = [&](TensorId id) -> const std::vector<std::int64_t>& {
    return g.tensors[id.value].shape;
  };
  auto dtype = [&](TensorId id) { return g.tensors[id.value].dtype; };
  auto fail = [&](const std::string& why) {
    *error = node_text(n) + ": " + why;
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
  auto f32_in = [&](TensorId id) {
    return dtype(id) == DType::Float32
               ? true
               : fail("operand '" + g.tensors[id.value].name +
                      "' must be float32");
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
      return fail("only axis 0 is supported");
    if (shape(n.inputs[0]).size() != 2 || shape(n.inputs[1]).size() != 1)
      return fail("expects a [rows, dim] table and 1-D indices");
    if (dtype(n.inputs[1]) != DType::Int32)
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
    if (!arity(1, 1)) return false;
    return numel_of(shape(n.inputs[0])) == numel_of(shape(n.outputs[0]))
               ? true
               : fail("element count differs");
  case OpKind::Add:
  case OpKind::Mul:
    return arity(2, 1);
  case OpKind::Softmax:
    if (!arity(1, 1)) return false;
    return f32_in(n.inputs[0]) && f32_in(n.outputs[0]);
  case OpKind::RmsNorm:
    if (!arity(2, 1)) return false;
    if (shape(n.inputs[0]).empty()) return fail("expects a non-scalar input");
    if (!f32_in(n.inputs[0]) || !f32_in(n.inputs[1])) return false;
    return shape(n.inputs[1]).size() == 1 &&
                   shape(n.inputs[1])[0] == shape(n.inputs[0]).back()
               ? true
               : fail("the weight must be one vector of the last dimension");
  case OpKind::Rope:
  case OpKind::RopeBwd: {
    if (!arity(1, 1)) return false;
    const auto& x = shape(n.inputs[0]);
    const std::int64_t head_dim = int_attr(n, "head_dim", 0);
    if (head_dim <= 0) return fail("`head_dim` is missing");
    if (x.empty() || x.back() % head_dim != 0)
      return fail("`head_dim` does not divide the last dimension");
    return true;
  }
  case OpKind::SiluMul:
  case OpKind::SiluMulBwd:
    return arity(n.op == OpKind::SiluMul ? 2 : 3,
                 n.op == OpKind::SiluMul ? 1 : 2);
  case OpKind::CrossEntropy:
    if (!arity(2, 1)) return false;
    if (shape(n.inputs[0]).size() != 2) return fail("expects [rows, cols] logits");
    if (dtype(n.inputs[1]) != DType::Int32) return fail("targets must be int32");
    return true;
  case OpKind::SoftmaxBwd:
    if (!arity(2, 1)) return false;
    if (shape(n.inputs[0]).empty()) return fail("expects a non-scalar input");
    if (shape(n.inputs[1]) != shape(n.inputs[0]))
      return fail("dY must match the softmax output shape");
    return true;
  case OpKind::CrossEntropyBwd:
    if (!arity(2, 2)) return false;
    if (shape(n.inputs[0]).size() != 2) return fail("expects [rows, cols] logits");
    if (dtype(n.inputs[1]) != DType::Int32) return fail("targets must be int32");
    return true;
  case OpKind::RmsNormBwd:
    if (!arity(3, 2)) return false;
    if (shape(n.inputs[0]).empty()) return fail("expects a non-scalar input");
    return true;
  case OpKind::EmbeddingBwd:
    if (!arity(2, 1)) return false;
    if (dtype(n.inputs[0]) != DType::Int32) return fail("token ids must be int32");
    if (shape(n.inputs[1]).size() != 2)
      return fail("expects [rows, dim] gradients");
    if (int_attr(n, "vocab", 0) <= 0) return fail("`vocab` must be positive");
    return true;
  case OpKind::AdamW:
    if (n.outputs.size() != 3 ||
        (n.inputs.size() != 5 && n.inputs.size() != 6))
      return fail("expected 5 or 6 inputs and 3 results");
    if (dtype(n.inputs[4]) != DType::Int32)
      return fail("the step counter must be int32");
    return true;
  case OpKind::ClipGradNorm:
    return n.inputs.empty() || n.outputs.size() != n.inputs.size()
               ? fail("expects one result per gradient operand")
               : true;
  case OpKind::Relu:
    return arity(1, 1);
  case OpKind::Conv: {
    if (!arity(3, 1)) return false;
    if (shape(n.inputs[0]).size() != 4 || shape(n.inputs[1]).size() != 4)
      return fail("expects rank-4 NCHW input and OIHW weight");
    if (int_attr(n, "group", 1) != 1)
      return fail("conv with group != 1 is unsupported");
    const auto dilations = ints_attr_or(n, "dilations", {1, 1});
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1)
      return fail("conv with dilations != 1 is unsupported");
    if (ints_attr_or(n, "pads", {0, 0, 0, 0}).size() != 4)
      return fail("conv pads must have 4 entries");
    if (ints_attr_or(n, "strides", {1, 1}).size() != 2)
      return fail("conv strides must have 2 entries");
    return f32_in(n.inputs[0]) && f32_in(n.inputs[1]) && f32_in(n.inputs[2]);
  }
  case OpKind::MaxPool: {
    if (!arity(1, 1)) return false;
    if (shape(n.inputs[0]).size() != 4)
      return fail("expects a rank-4 NCHW input");
    if (int_attr(n, "ceil_mode", 0) != 0)
      return fail("max_pool with ceil_mode != 0 is unsupported");
    const auto dilations = ints_attr_or(n, "dilations", {1, 1});
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1)
      return fail("max_pool with dilations != 1 is unsupported");
    if (ints_attr(n, "kernel_shape").size() != 2)
      return fail("max_pool requires a 2-entry kernel_shape");
    if (ints_attr_or(n, "pads", {0, 0, 0, 0}).size() != 4)
      return fail("max_pool pads must have 4 entries");
    if (ints_attr_or(n, "strides", {1, 1}).size() != 2)
      return fail("max_pool strides must have 2 entries");
    return f32_in(n.inputs[0]) && f32_in(n.outputs[0]);
  }
  case OpKind::ReduceMean:
    if (n.outputs.size() != 1) return fail("expects exactly one result");
    if (n.inputs.size() < 2) return fail("expects an explicit axes input");
    return f32_in(n.inputs[0]);
  case OpKind::Gemm: {
    if (n.inputs.size() < 2 || n.inputs.size() > 3)
      return fail("gemm expects 2 or 3 inputs");
    if (shape(n.inputs[0]).size() != 2 || shape(n.inputs[1]).size() != 2)
      return fail("gemm requires rank-2 A and B");
    if (n.inputs.size() == 3) {
      const std::int64_t N = shape(n.outputs[0])[1];
      const std::int64_t M = shape(n.outputs[0])[0];
      const std::int64_t bias = numel_of(shape(n.inputs[2]));
      if (bias != N && bias != M * N)
        return fail("gemm bias must broadcast along the trailing axis");
    }
    return true;
  }
  default:
    *error = node_text(n) + ": not supported by the CPU Boom executor";
    return false;
  }
}

// The axis operand of a ReduceMean is a constant int64 vector; read it once at
// compile so dispatch does not depend on the buffers being initialized.
bool reduce_mean_axes(const planner::Graph& g, const GraphNodeDesc& n,
                      const std::vector<std::vector<std::byte>>& slot_bytes,
                      const std::vector<std::int64_t>& slot_of,
                      std::vector<std::int64_t>* axes, std::string* error) {
  if (n.inputs.size() < 2) {
    *error = "reduce_mean requires an explicit axes input in v0";
    return false;
  }
  const TensorDesc& t = g.tensors[n.inputs[1].value];
  if (!t.is_constant) {
    *error = "reduce_mean axes must be a constant";
    return false;
  }
  const std::vector<std::byte>& bytes =
      slot_bytes[static_cast<std::size_t>(slot_of[t.id.value])];
  const std::int64_t n_axes = numel_of(t.shape);
  axes->clear();
  if (t.dtype == DType::Int64) {
    if (bytes.size() < static_cast<std::size_t>(n_axes) * sizeof(std::int64_t)) {
      *error = "reduce_mean axes constant is too small";
      return false;
    }
    const auto* p = reinterpret_cast<const std::int64_t*>(bytes.data());
    axes->assign(p, p + n_axes);
  } else if (t.dtype == DType::Int32) {
    if (bytes.size() < static_cast<std::size_t>(n_axes) * sizeof(std::int32_t)) {
      *error = "reduce_mean axes constant is too small";
      return false;
    }
    const auto* p = reinterpret_cast<const std::int32_t*>(bytes.data());
    for (std::int64_t i = 0; i < n_axes; ++i) axes->push_back(p[i]);
  } else {
    *error = "reduce_mean axes must be int32 or int64";
    return false;
  }
  // Normalize negative axes and reject an out-of-range or duplicated one: a
  // duplicate would count the same dimension twice in the divisor.
  const std::int64_t rank =
      static_cast<std::int64_t>(g.tensors[n.inputs[0].value].shape.size());
  for (std::size_t i = 0; i < axes->size(); ++i) {
    std::int64_t& a = (*axes)[i];
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) {
      *error = "reduce_mean axis out of range for rank " + std::to_string(rank);
      return false;
    }
    for (std::size_t j = 0; j < i; ++j)
      if ((*axes)[j] == a) {
        *error = "reduce_mean has duplicate axes";
        return false;
      }
  }
  return true;
}

} // namespace

// --- the compiled graph -----------------------------------------------------

struct Executable::Impl {
  planner::Graph graph;
  // One allocation per buffer slot (BufferReusePass classes share a slot).
  std::vector<std::vector<std::byte>> slot_bytes;
  std::vector<std::int64_t> slot_of;  // by tensor id
  std::vector<std::byte*> buffers;    // by tensor id, into slot_bytes
  std::vector<TensorId> input_ids;
  std::vector<TensorId> output_ids;
  std::vector<TensorType> input_types;
  std::vector<TensorType> output_types;

  // Compile-time scratch: padded NCHW input (conv/max_pool), pooling indices,
  // transposed gemm operands. Sized from the graph so a run allocates nothing.
  // A run writes through them, so they are mutable on a const Impl.
  mutable std::vector<std::byte> scratch_pad;
  mutable std::vector<std::int64_t> scratch_index;
  mutable std::vector<std::byte> scratch_gemm;
  // ReduceMean axes, per graph node (resolved at compile).
  std::vector<std::vector<std::int64_t>> reduce_axes;

  bool run(const std::vector<Bytes>& inputs, std::vector<Bytes>& outputs,
           ExecError* error) const {
    if (inputs.size() != input_ids.size()) {
      *error = {ExecErrorKind::Binding,
                "expected " + std::to_string(input_ids.size()) +
                    " input buffer(s), got " + std::to_string(inputs.size())};
      return false;
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const TensorDesc& t = graph.tensors[input_ids[i].value];
      if (inputs[i].size() != t.size_bytes) {
        *error = {ExecErrorKind::Binding,
                  "input " + std::to_string(i) + " (" + t.name + ") is " +
                      std::to_string(inputs[i].size()) + " bytes, expected " +
                      std::to_string(t.size_bytes)};
        return false;
      }
    }
    for (std::size_t i = 0; i < inputs.size(); ++i)
      std::memcpy(buffers[input_ids[i].value], inputs[i].data(),
                  inputs[i].size());

    NodeContext ctx{graph, buffers, nullptr, 0, nullptr, 0, nullptr, 0,
                    nullptr};
    if (!scratch_pad.empty()) {
      ctx.pad_scratch = scratch_pad.data();
      ctx.pad_scratch_bytes = scratch_pad.size();
    }
    if (!scratch_index.empty()) {
      ctx.index_scratch = scratch_index.data();
      ctx.index_scratch_entries =
          static_cast<std::uint64_t>(scratch_index.size());
    }
    if (!scratch_gemm.empty()) {
      ctx.gemm_scratch = scratch_gemm.data();
      ctx.gemm_scratch_bytes = scratch_gemm.size();
    }
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
      const GraphNodeDesc& node = graph.nodes[i];
      if (node.op == OpKind::ReduceMean) ctx.reduce_axes = &reduce_axes[i];
      std::string why;
      if (!dispatch_node(ctx, node, &why)) {
        *error = {ExecErrorKind::Runtime, why};
        return false;
      }
    }
    outputs.resize(output_ids.size());
    for (std::size_t i = 0; i < output_ids.size(); ++i) {
      const TensorDesc& t = graph.tensors[output_ids[i].value];
      outputs[i].resize(t.size_bytes);
      std::memcpy(outputs[i].data(), buffers[output_ids[i].value],
                  t.size_bytes);
    }
    return true;
  }
};

// --- compile ----------------------------------------------------------------

std::expected<std::unique_ptr<Executable>, ExecError> Executable::compile(
    const Document& doc, const std::unordered_map<std::string, Bytes>& weights) {
  // Lower and pass first: a graph the IR rejects must be rejected for that
  // reason, not for a kernel's. The lowering reports its reason as an MLIR
  // diagnostic; capture it into the returned error (the caller has no other
  // channel for it — the frontends are MLIR-free).
  mlir::MLIRContext context;
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler capture(&context, [&diagnostic](
                                                    mlir::Diagnostic& d) {
    if (d.getSeverity() != mlir::DiagnosticSeverity::Error) return;
    if (!diagnostic.empty()) diagnostic += "; ";
    llvm::raw_string_ostream os(diagnostic);
    d.print(os);
    os.flush();
    while (!diagnostic.empty() &&
           (diagnostic.back() == '\n' || diagnostic.back() == ' '))
      diagnostic.pop_back();
  });
  auto compile_error = [&diagnostic](const char* what) {
    return std::unexpected(ExecError{
        ExecErrorKind::Compile,
        diagnostic.empty() ? std::string(what)
                           : std::string(what) + ": " + diagnostic});
  };

  auto module = compiler::bladeToBoom(doc, &context);
  if (!module) return compile_error("Blade -> Boom lowering failed");
  if (!compiler::runBoomPasses(*module, planner::IrPassPhase::Training))
    return compile_error("Boom IR passes failed");
  auto lowered = compiler::boomToGraph(*module);
  if (!lowered) return compile_error("Boom IR -> graph lowering failed");

  std::unique_ptr<Executable> out(new Executable());
  Impl& s = *out->impl_;
  s.graph = std::move(*lowered);
  const planner::Graph& g = s.graph;

  std::map<std::string, const Parameter*> params;
  for (const Parameter& p : doc.graph.parameters) params[p.name] = &p;

  // --- buffer slots ---------------------------------------------------------
  //
  // BufferReusePass marks each single-result node's result with the class it
  // belongs to (`buf`) and whether it may overwrite operand 0 (`inplace`).
  // Class members are never live at the same time, so they share one
  // allocation. The in-place mark is deliberately not honoured: the executor
  // always writes results into their own class (or their own slot), which is
  // the correct fallback the pass documents.
  std::vector<std::int64_t> slot_of(g.tensors.size(), -1);
  std::vector<std::uint64_t> slot_sizes;
  auto new_slot = [&](std::uint64_t bytes) {
    slot_sizes.push_back(bytes);
    return static_cast<std::int64_t>(slot_sizes.size()) - 1;
  };
  std::map<std::int64_t, std::int64_t> class_slot;

  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& n = g.nodes[i];
    if (n.outputs.size() != 1) continue;  // a multi-result node: own buffers
    const TensorId rid = n.outputs[0];
    const std::uint64_t bytes = g.tensors[rid.value].size_bytes;
    if (bytes == 0) {
      return std::unexpected(
          ExecError{ExecErrorKind::Compile,
                    "tensor '" + g.tensors[rid.value].name +
                        "' has no storage"});
    }
    int64_t slot = -1;
    const int64_t cls = int_attr(n, "buf", -1);
    if (cls >= 0) {
      auto it = class_slot.find(cls);
      if (it == class_slot.end())
        it = class_slot.emplace(cls, new_slot(bytes)).first;
      slot = it->second;
    }
    if (slot < 0) slot = new_slot(bytes);
    slot_of[rid.value] = slot;
    slot_sizes[slot] = std::max(slot_sizes[slot], bytes);
  }
  for (std::size_t t = 0; t < g.tensors.size(); ++t) {
    if (slot_of[t] >= 0) continue;
    if (g.tensors[t].size_bytes == 0) {
      return std::unexpected(
          ExecError{ExecErrorKind::Compile,
                    "tensor '" + g.tensors[t].name + "' has no storage"});
    }
    slot_of[t] = new_slot(g.tensors[t].size_bytes);
  }

  s.slot_bytes.resize(slot_sizes.size());
  for (std::size_t i = 0; i < slot_sizes.size(); ++i)
    s.slot_bytes[i].resize(slot_sizes[i]);
  s.slot_of = slot_of;
  s.buffers.assign(g.tensors.size(), nullptr);
  for (std::size_t t = 0; t < g.tensors.size(); ++t)
    s.buffers[t] = s.slot_bytes[static_cast<std::size_t>(slot_of[t])].data();

  // Constants are materialized into their buffers once.
  for (const TensorDesc& t : g.tensors) {
    if (!t.is_constant) continue;
    auto it = params.find(t.name);
    if (it == params.end()) {
      return std::unexpected(
          ExecError{ExecErrorKind::Compile,
                    "constant tensor '" + t.name +
                        "' has no parameter in the document"});
    }
    std::vector<std::byte> bytes;
    std::string error;
    if (!parameter_bytes(*it->second, weights, &bytes, &error)) {
      return std::unexpected(
          ExecError{ExecErrorKind::Weight, std::move(error)});
    }
    if (bytes.size() != t.size_bytes) {
      return std::unexpected(
          ExecError{ExecErrorKind::Weight,
                    "constant tensor '" + t.name + "' is " +
                        std::to_string(t.size_bytes) + " bytes but its "
                        "parameter holds " + std::to_string(bytes.size())});
    }
    std::memcpy(s.buffers[t.id.value], bytes.data(), bytes.size());
  }

  // Compile-time scratch.
  std::uint64_t pad_bytes = 0, index_entries = 0, gemm_bytes = 0;
  for (const GraphNodeDesc& n : g.nodes) {
    if (n.op == OpKind::Conv || n.op == OpKind::MaxPool) {
      const auto& x = g.tensors[n.inputs[0].value].shape;
      const auto pads = ints_attr_or(n, "pads", {0, 0, 0, 0});
      const std::uint64_t b =
          static_cast<std::uint64_t>(x[0] * x[1] *
                                     (x[2] + pads[0] + pads[2]) *
                                     (x[3] + pads[1] + pads[3])) *
          sizeof(float);
      pad_bytes = std::max(pad_bytes, b);
    }
    if (n.op == OpKind::MaxPool) {
      const auto& o = g.tensors[n.outputs[0].value].shape;
      index_entries = std::max(
          index_entries,
          static_cast<std::uint64_t>(o[0] * o[1] * o[2] * o[3]));
    }
    if (n.op == OpKind::Gemm) {
      const std::uint64_t ab = numel_of(g.tensors[n.inputs[0].value].shape) *
                               sizeof(float);
      const std::uint64_t bb = numel_of(g.tensors[n.inputs[1].value].shape) *
                               sizeof(float);
      gemm_bytes = std::max(gemm_bytes, ab + bb);
    }
  }
  s.scratch_pad.resize(static_cast<std::size_t>(pad_bytes));
  s.scratch_index.resize(static_cast<std::size_t>(index_entries));
  s.scratch_gemm.resize(static_cast<std::size_t>(gemm_bytes));

  auto type_of = [&](TensorId id) {
    const TensorDesc& t = g.tensors[id.value];
    return TensorType{t.dtype, Shape{t.shape}};
  };
  s.input_ids = g.inputs;
  s.output_ids = g.outputs;
  for (TensorId id : g.inputs) s.input_types.push_back(type_of(id));
  for (TensorId id : g.outputs) s.output_types.push_back(type_of(id));

  // Every node must be one this executor runs, with the shapes and dtypes its
  // kernels read; checked once here so a run never fails for a graph property.
  s.reduce_axes.resize(g.nodes.size());
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const GraphNodeDesc& n = g.nodes[i];
    std::string error;
    if (!validate_node(g, n, &error)) {
      return std::unexpected(
          ExecError{ExecErrorKind::Compile, std::move(error)});
    }
    if (n.op == OpKind::ReduceMean) {
      std::string why;
      if (!reduce_mean_axes(g, n, s.slot_bytes, s.slot_of, &s.reduce_axes[i],
                            &why)) {
        return std::unexpected(
            ExecError{ExecErrorKind::Compile,
                      node_text(n) + ": " + why});
      }
    }
  }

  return std::expected<std::unique_ptr<Executable>, ExecError>(std::move(out));
}

// --- the facade -------------------------------------------------------------

Executable::Executable() : impl_(std::make_unique<Impl>()) {}
Executable::~Executable() = default;
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;

const TensorType& Executable::input_type() const {
  static const TensorType empty{};
  return impl_->input_types.empty() ? empty : impl_->input_types.front();
}

const TensorType& Executable::output_type() const {
  static const TensorType empty{};
  return impl_->output_types.empty() ? empty : impl_->output_types.front();
}

const std::vector<TensorType>& Executable::input_types() const {
  return impl_->input_types;
}

const std::vector<TensorType>& Executable::output_types() const {
  return impl_->output_types;
}

std::expected<void, ExecError> Executable::run(
    const std::vector<Bytes>& inputs, std::vector<Bytes>& outputs) const {
  ExecError error{};
  if (!impl_->run(inputs, outputs, &error))
    return std::unexpected(std::move(error));
  return {};
}

std::expected<void, ExecError> Executable::run(const Bytes& input,
                                               Bytes& output) const {
  std::vector<Bytes> inputs{input};
  std::vector<Bytes> outputs;
  auto result = run(inputs, outputs);
  if (!result) return result;
  if (outputs.size() != 1)
    return std::unexpected(
        ExecError{ExecErrorKind::Runtime,
                  "the graph has " + std::to_string(outputs.size()) +
                      " outputs; the single-output form needs exactly one"});
  output = std::move(outputs.front());
  return {};
}

// --- external weights -------------------------------------------------------

std::expected<std::unordered_map<std::string, Bytes>, ExecError>
load_external_weights(const Document& doc, const std::string& base_dir) {
  std::unordered_map<std::string, Bytes> out;
  for (const Parameter& p : doc.graph.parameters) {
    const auto* external = std::get_if<ExternalData>(&p.data);
    if (external == nullptr) continue;

    std::string path = external->file;
    if (!path.empty() && path.front() != '/')
      path = base_dir.empty() ? path : base_dir + "/" + path;

    const std::int64_t expected =
        numel(p.type) * static_cast<std::int64_t>(dtype_size(p.type.dtype));
    if (external->length != static_cast<std::uint64_t>(expected)) {
      return std::unexpected(
          ExecError{ExecErrorKind::Weight,
                    "parameter '" + p.name + "' declares " +
                        std::to_string(external->length) + " bytes but its "
                        "type needs " + std::to_string(expected)});
    }

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      return std::unexpected(
          ExecError{ExecErrorKind::Weight,
                    "cannot open weight sidecar '" + path + "' for parameter '" +
                        p.name + "'"});
    }
    Bytes bytes(static_cast<std::size_t>(expected));
    bool ok = true;
    if (std::fseek(f, static_cast<long>(external->offset), SEEK_SET) != 0)
      ok = false;
    if (ok && expected > 0 &&
        std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size())
      ok = false;
    std::fclose(f);
    if (!ok) {
      return std::unexpected(
          ExecError{ExecErrorKind::Weight,
                    "cannot read " + std::to_string(expected) + " bytes at " +
                        std::to_string(external->offset) + " from '" + path +
                        "'"});
    }
    out.emplace(p.name, std::move(bytes));
  }
  return out;
}

} // namespace sonicboom::sx
