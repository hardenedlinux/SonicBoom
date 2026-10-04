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

// SonicBoom S-Expr v0.1 → CPU execution (public facade).
//
// This header is deliberately MLIR-free. The MLIR pass pipeline and LLVM JIT
// live behind this boundary in core/src/sx/exec.cpp: `Executable::compile`
// lowers a validated Document into an MLIR module, runs the CPU lowering
// pipeline (one-shot bufferize → linalg-to-loops → … → LLVM dialect), and JITs
// it into a callable native function. `run` executes that function over raw
// little-endian tensor buffers.
//
// v0 execution scope: exactly one float32 graph input and one float32 graph
// output (the ResNet-18 example and the Add/Relu smoke graphs both satisfy
// this). Compiling a graph outside this scope is an explicit error.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sonicboom/sx/ir.h>

namespace sonicboom::sx {

// Raw little-endian byte buffer for one tensor (row-major, matching its dtype).
using Bytes = std::vector<std::byte>;

// Broad classes of execution failure.
enum class ExecErrorKind : uint8_t {
  Weight,   // a weight sidecar could not be read / resolved
  Compile,  // lowering, pass pipeline, or JIT compilation failed
  Binding,  // input/output buffer size or type mismatch
  Runtime,  // the JITted function failed or was unavailable
};

struct ExecError {
  ExecErrorKind kind;
  std::string message;
};

// Resolve every `:external` parameter in `doc` into raw little-endian bytes,
// reading (file, offset, length) from sidecars. A `file` that is not an
// absolute path is resolved relative to `base_dir`. Each returned buffer's
// byte count is validated against the parameter's declared shape and dtype.
std::expected<std::unordered_map<std::string, Bytes>, ExecError>
load_external_weights(const Document& doc, const std::string& base_dir);

// A CPU-compiled, executable S-Expr graph.
class Executable {
public:
  // Compile `doc` for CPU execution. `weights` must provide a buffer for every
  // `:external` parameter (see load_external_weights); `:values` parameters are
  // taken from `doc` itself. Requires exactly one float32 graph input and one
  // float32 graph output (v0 execution scope).
  static std::expected<std::unique_ptr<Executable>, ExecError> compile(
      const Document& doc, const std::unordered_map<std::string, Bytes>& weights);

  Executable(Executable&&) noexcept;
  Executable& operator=(Executable&&) noexcept;
  Executable(const Executable&) = delete;
  Executable& operator=(const Executable&) = delete;
  ~Executable();

  // The graph's single input and output tensor types (binding contract).
  const TensorType& input_type() const;
  const TensorType& output_type() const;

  // Run the graph. `input` is the raw little-endian buffer for the single input
  // tensor (byte count must equal numel(input_type()) * dtype_size); `output`
  // is resized to the single output tensor and filled. Returns an error on
  // size/type mismatch or runtime failure.
  std::expected<void, ExecError> run(const Bytes& input, Bytes& output) const;

private:
  Executable();  // constructed by compile()

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sonicboom::sx
