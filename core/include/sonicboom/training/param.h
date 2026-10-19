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

// A trainable float32 parameter and its optimizer state.
//
// A Param owns four equally sized flat buffers: the parameter itself, its
// gradient, and AdamW's first and second moment estimates. When a CUDA device
// is present they stay in device memory across updates (the update is a fused
// device kernel and the buffers are never moved); otherwise they are host
// float32 vectors and `step()` runs the host reference. The device handles are
// opaque uint64_t values — the same representation the planner's device buffer
// pool uses — so no CUDA type crosses this header.
//
// The public interface is deliberately flat: training graphs consume the raw
// buffers, and the shape is only needed to size them (the kernels are
// elementwise over `size()` floats).

#include <sonicboom/training/adamw.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

namespace sonicboom::training {

class Param {
public:
  // Create a parameter from its initial values. The gradient and both moment
  // buffers start at zero. Fails when `init` is empty.
  static std::expected<std::unique_ptr<Param>, std::string> create(
      std::span<const float> init);

  ~Param();
  Param(Param&&) noexcept;
  Param& operator=(Param&&) noexcept;
  Param(const Param&) = delete;
  Param& operator=(const Param&) = delete;

  // Number of float elements.
  std::size_t size() const noexcept;

  // True when the buffers are device-resident (a CUDA device was available at
  // creation).
  bool on_device() const noexcept;

  // Zero the gradient buffer (the start of a training step).
  void zero_grad();

  // Replace the gradient buffer. `grad.size()` must equal size().
  bool set_grad(std::span<const float> grad);

  // One AdamW update: `cfg` supplies the hyperparameters and the Param's own
  // 1-based counter supplies the bias-correction step. Returns false when no
  // update could be performed (sizes or device submission failed).
  bool step(const AdamWConfig& cfg);

  // Read a buffer back into host floats. Each requires `out.size() == size()`.
  bool read_param(std::span<float> out) const;
  bool read_grad(std::span<float> out) const;
  bool read_m1(std::span<float> out) const;
  bool read_m2(std::span<float> out) const;

  // Device buffer handles (0 when the Param is host-resident). The parameter
  // handle is what a training graph binds as its parameter input; the gradient
  // handle is where the backward graph writes.
  std::uint64_t param_handle() const noexcept;
  std::uint64_t grad_handle() const noexcept;
  std::uint64_t m1_handle() const noexcept;
  std::uint64_t m2_handle() const noexcept;

  // Number of updates applied so far (0 before the first step).
  std::uint64_t step_count() const noexcept;

private:
  Param();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sonicboom::training
