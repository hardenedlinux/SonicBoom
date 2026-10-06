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

#include <sonicboom/autograd.h>

#include <sonicboom/argument_list.h>
#include <sonicboom/operator_handle.h>
#include <sonicboom/scalar.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/value.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sonicboom {

namespace {

// Dispatch one native operator and return its single tensor result. The
// operator is looked up by (name, overload) and invoked through the Layer 1
// adapter's boxed dispatch; any failure is reported as a std::runtime_error so
// callers (the C API guard, tests) see a clean message, never a c10 exception.
nt::Tensor call1(std::string name, std::string overload, nt::ArgumentList args) {
  try {
    nt::OperatorHandle op = nt::find_operator(nt::OperatorName{name, overload});
    nt::ResultList out = op.call(std::move(args));
    if (out.size() != 1 || !out[0].isTensor())
      throw std::runtime_error("unexpected result arity/kind");
    return out[0].toTensor();
  } catch (const std::exception& e) {
    throw std::runtime_error("autograd op '" + name + "' failed: " + e.what());
  }
}

// --- primitive operators (all float32 CPU, boxed dispatch) -----------------

nt::Tensor transpose(const nt::Tensor& a) {
  return call1("aten::t", "", nt::ArgumentList({nt::Value(a)}));
}

nt::Tensor mm(const nt::Tensor& a, const nt::Tensor& b) {
  return call1("aten::mm", "", nt::ArgumentList({nt::Value(a), nt::Value(b)}));
}

nt::Tensor add(const nt::Tensor& a, const nt::Tensor& b) {
  return call1("aten::add", "Tensor",
               nt::ArgumentList({nt::Value(a), nt::Value(b), nt::Value(nt::Scalar(1.0))}));
}

nt::Tensor sub(const nt::Tensor& a, const nt::Tensor& b) {
  return call1("aten::sub", "Tensor",
               nt::ArgumentList({nt::Value(a), nt::Value(b), nt::Value(nt::Scalar(1.0))}));
}

nt::Tensor mul(const nt::Tensor& a, const nt::Tensor& b) {
  return call1("aten::mul", "Tensor", nt::ArgumentList({nt::Value(a), nt::Value(b)}));
}

nt::Tensor mul_scalar(const nt::Tensor& a, double s) {
  return call1("aten::mul", "Scalar",
               nt::ArgumentList({nt::Value(a), nt::Value(nt::Scalar(s))}));
}

nt::Tensor relu_op(const nt::Tensor& a) {
  return call1("aten::relu", "", nt::ArgumentList({nt::Value(a)}));
}

nt::Tensor gt_scalar(const nt::Tensor& a, double s) {
  return call1("aten::gt", "Scalar",
               nt::ArgumentList({nt::Value(a), nt::Value(nt::Scalar(s))}));
}

nt::Tensor to_float(const nt::Tensor& a) {
  return call1("aten::to", "dtype",
               nt::ArgumentList({nt::Value(a), nt::Value(nt::ScalarType::Float),
                                 nt::Value(false), nt::Value(false), nt::Value()}));
}

nt::Tensor sum_dim0(const nt::Tensor& a) {
  return call1("aten::sum", "dim_IntList",
               nt::ArgumentList({nt::Value(a), nt::Value(std::vector<int64_t>{0}),
                                 nt::Value(false), nt::Value()}));
}

nt::Tensor mean_all(const nt::Tensor& a) {
  return call1("aten::mean", "", nt::ArgumentList({nt::Value(a), nt::Value()}));
}

std::size_t itemsize(nt::ScalarType t) {
  switch (t) {
    case nt::ScalarType::Float:  return 4;
    case nt::ScalarType::Double: return 8;
    case nt::ScalarType::Int:    return 4;
    case nt::ScalarType::Long:   return 8;
    default:                     return 0;
  }
}

} // namespace

void Tape::accumulate(const nt::Tensor& t, const nt::Tensor& g) {
  auto it = grads_.find(t.identity());
  if (it == grads_.end()) {
    grads_.emplace(t.identity(), g);
  } else {
    it->second = add(it->second, g);
  }
}

nt::Tensor linear(const nt::Tensor& x, const nt::Tensor& w, const nt::Tensor& b,
                  Tape& tape) {
  nt::Tensor y = add(mm(x, transpose(w)), b);

  Tape::Node n;
  n.op = Tape::Op::Linear;
  n.inputs = {x, w, b};
  n.output = y;
  tape.record(std::move(n));
  return y;
}

nt::Tensor relu(const nt::Tensor& x, Tape& tape) {
  nt::Tensor y = relu_op(x);

  Tape::Node n;
  n.op = Tape::Op::Relu;
  n.inputs = {x};
  n.output = y;
  tape.record(std::move(n));
  return y;
}

nt::Tensor mse_loss(const nt::Tensor& pred, const nt::Tensor& target,
                    Tape& tape) {
  nt::Tensor diff = sub(pred, target);
  nt::Tensor loss = mean_all(mul(diff, diff));

  Tape::Node n;
  n.op = Tape::Op::MseLoss;
  n.inputs = {pred, target, diff};
  n.output = loss;
  tape.record(std::move(n));
  return loss;
}

void backward(const nt::Tensor& loss, Tape& tape) {
  // Seed d(loss)/d(loss) = 1 (a 0-dim float32 scalar).
  nt::Tensor seed = nt::empty({}, nt::ScalarType::Float);
  *static_cast<float*>(seed.data_ptr()) = 1.0f;
  tape.accumulate(loss, seed);

  for (auto it = tape.nodes_.rbegin(); it != tape.nodes_.rend(); ++it) {
    const Tape::Node& n = *it;
    nt::Tensor g = tape.grad(n.output);
    if (!g.defined())
      continue;  // no gradient flowed to this node's output

    switch (n.op) {
      case Tape::Op::Linear: {
        const nt::Tensor& x = n.inputs[0];
        const nt::Tensor& w = n.inputs[1];
        const nt::Tensor& b = n.inputs[2];
        // y = mm(x, t(w)) + b
        //   dL/db = sum_i dL/dy[i, :]          → [out]
        //   dL/dx = dL/dy @ w                   → [N, in]
        //   dL/dw = t(dL/dy) @ x                → [out, in]
        tape.accumulate(b, sum_dim0(g));
        tape.accumulate(x, mm(g, w));
        tape.accumulate(w, mm(transpose(g), x));
        break;
      }
      case Tape::Op::Relu: {
        const nt::Tensor& x = n.inputs[0];
        // dL/dx = dL/dy * (x > 0)
        nt::Tensor mask = to_float(gt_scalar(x, 0.0));
        tape.accumulate(x, mul(g, mask));
        break;
      }
      case Tape::Op::MseLoss: {
        const nt::Tensor& pred = n.inputs[0];
        const nt::Tensor& target = n.inputs[1];
        const nt::Tensor& diff = n.inputs[2];
        float gl = *static_cast<const float*>(g.data_ptr());
        double scale = 2.0 / static_cast<double>(pred.numel());
        tape.accumulate(pred, mul_scalar(diff, scale * gl));
        tape.accumulate(target, mul_scalar(diff, -scale * gl));
        break;
      }
    }
  }
}

void sgd_step(nt::Tensor& param, const nt::Tensor& grad, double lr) {
  nt::Tensor updated = sub(param, mul_scalar(grad, lr));
  std::size_t nbytes =
      static_cast<std::size_t>(param.numel()) * itemsize(param.dtype());
  if (nbytes == 0)
    throw std::runtime_error("sgd_step: unsupported parameter dtype");
  std::memcpy(param.data_ptr(), updated.data_ptr(), nbytes);
}

} // namespace sonicboom
