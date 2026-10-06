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

// Minimal reverse-mode autograd over nt::Tensor (Layer 2), enough to train a
// small MLP end to end. This is NOT the native-torch / PyTorch autograd engine:
// it records a flat tape of the differentiable ops below and replays vector-
// Jacobian products (VJPs) in reverse topological order. Every numeric
// operation is dispatched through the existing Layer 1 adapter
// (nt::find_operator → boxed dispatch), so the arithmetic runs in native C++,
// never in Guile or Scheme.
//
// v0 coverage is deliberately narrow: linear (x @ Wᵀ + b), relu, and mean
// squared error, plus an SGD step. No full autograd compatibility is promised.

#include <sonicboom/tensor.h>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace sonicboom {

// A tape of differentiable operations for one forward pass. The tape owns the
// recorded dependency graph and the accumulated gradients (keyed by tensor
// identity), so a caller can run forward, backward, then read parameter
// gradients before zeroing and reusing the tape for the next iteration.
class Tape {
 public:
  Tape() = default;
  Tape(const Tape&) = delete;
  Tape& operator=(const Tape&) = delete;
  Tape(Tape&&) = default;
  Tape& operator=(Tape&&) = default;

  // Number of recorded differentiable operations.
  std::size_t size() const noexcept { return nodes_.size(); }

  // True if a gradient has been accumulated for `t`.
  bool has_grad(const nt::Tensor& t) const {
    return grads_.find(t.identity()) != grads_.end();
  }

  // The accumulated gradient for `t`, or an undefined tensor if none.
  nt::Tensor grad(const nt::Tensor& t) const {
    auto it = grads_.find(t.identity());
    return it == grads_.end() ? nt::Tensor() : it->second;
  }

  // Clear recorded operations and accumulated gradients (start a fresh pass).
  void zero_grad() {
    nodes_.clear();
    grads_.clear();
  }

 private:
  enum class Op : uint8_t { Linear, Relu, MseLoss };

  struct Node {
    Op op;
    std::vector<nt::Tensor> inputs;  // tensors saved for the VJP (incl. params)
    nt::Tensor output;               // the forward result (routes the incoming grad)
  };

  std::vector<Node> nodes_;
  std::unordered_map<const void*, nt::Tensor> grads_;

  void record(Node n) { nodes_.push_back(std::move(n)); }
  void accumulate(const nt::Tensor& t, const nt::Tensor& g);

  friend nt::Tensor linear(const nt::Tensor& x, const nt::Tensor& w,
                           const nt::Tensor& b, Tape& tape);
  friend nt::Tensor relu(const nt::Tensor& x, Tape& tape);
  friend nt::Tensor mse_loss(const nt::Tensor& pred, const nt::Tensor& target,
                             Tape& tape);
  friend void backward(const nt::Tensor& loss, Tape& tape);
};

// y = x @ Wᵀ + b, where x is [N, in], W is [out, in], b is [out]; the result is
// [N, out] (b broadcasts along the batch axis). Records a Linear node.
nt::Tensor linear(const nt::Tensor& x, const nt::Tensor& w, const nt::Tensor& b,
                  Tape& tape);

// y = max(x, 0), elementwise. Records a Relu node.
nt::Tensor relu(const nt::Tensor& x, Tape& tape);

// scalar = mean((pred - target)²) over all elements. pred and target must have
// the same shape; the result is a 0-dim (scalar) tensor. Records an MseLoss
// node that backprops into both pred and target.
nt::Tensor mse_loss(const nt::Tensor& pred, const nt::Tensor& target,
                    Tape& tape);

// Reverse-mode backprop: seeds the loss gradient with 1.0 and replays the tape
// in reverse topological order, accumulating gradients into the tape. After
// this, Tape::grad(param) yields the parameter's gradient.
void backward(const nt::Tensor& loss, Tape& tape);

// In-place SGD update: param <- param - lr * grad. `param` keeps its identity
// (and underlying storage) so gradient accumulation stays valid across a
// training loop; `grad` must have the same shape/dtype as `param`.
void sgd_step(nt::Tensor& param, const nt::Tensor& grad, double lr);

} // namespace sonicboom
