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

// PRIVATE header shared by core/src/sx/lowering.cpp and core/src/sx/exec.cpp.
// It is NOT installed and never included by a public <sonicboom/...> header,
// so the MLIR types below do not leak into the public Layer 2 / C API surface.

#include <sonicboom/sx/ir.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sonicboom::sx {

// The v0 MLIR/linalg CPU pipeline is deprecated. Boom IR carries the same
// seven-op vocabulary (BoomOps.td: conv, relu, add, max_pool, reduce_mean,
// reshape, gemm), but the Blade -> Boom -> graph route executes only on CUDA
// (DeviceTrainingGraph), no CPU Boom executor exists, and `sb_compile`'s CPU
// execution is part of the stable C ABI — so the path cannot be deleted yet.
// SONICBOOM_SX_LINALG=0 retires it explicitly (compilation fails with a clear
// error) for a build or run that only uses the device route; it is the switch
// the removal will flip.
inline bool linalg_cpu_path_enabled() {
  const char* raw = std::getenv("SONICBOOM_SX_LINALG");
  if (!raw) return true;
  const std::string_view v{raw};
  return !(v == "0" || v == "off" || v == "none");
}

// Raw little-endian bytes for one external weight tensor, keyed by parameter
// name. The byte count must equal numel(type) * dtype_size(type).
using WeightMap = std::unordered_map<std::string, std::vector<std::byte>>;

// Lower `doc` into `module`, which must already be created inside `ctx` with
// the func/arith/tensor/linalg dialects loaded.
//
// If `weights` is non-null, every `:external` parameter is materialized as a
// dense constant from `weights` (each must be present, with a byte count that
// matches the parameter's declared shape/dtype). If `weights` is null, `:external`
// parameters become `tensor.empty` placeholders and their sidecar metadata is
// recorded on the module (the serialization view used by lower_to_mlir).
//
// Returns "" on success, or a non-empty error string on failure.
std::string lower_into_module(mlir::MLIRContext& ctx, mlir::ModuleOp module,
                              const Document& doc, const WeightMap* weights);

} // namespace sonicboom::sx
