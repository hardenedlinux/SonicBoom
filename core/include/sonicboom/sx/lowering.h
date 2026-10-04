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

// SonicBoom S-Expr v0.1 → MLIR lowering (public facade).
//
// This header is deliberately MLIR-free: the actual MLIR types, dialect
// registration and module construction live behind this boundary in
// core/src/sx/lowering.cpp. The public surface returns the textual form of a
// verified MLIR module (or a structured error), so consumers — the C API,
// Guile, or any other binding — never see MLIR types.
//
// Lowering pipeline (internal): MLIRContext → dialect registration →
// S-Expr IR → one func.func per graph → full MLIR verification → print.

#include <cstdint>
#include <expected>
#include <string>

#include <sonicboom/sx/ir.h>

namespace sonicboom::sx {

// Broad classes of lowering failure.
enum class LoweringErrorKind : uint8_t {
  Type,          // a dtype/shape could not be mapped to an MLIR type
  Operator,      // an operator is unsupported or could not be lowered
  Verification,  // the produced module failed MLIR verification
};

struct LoweringError {
  LoweringErrorKind kind;
  std::string message;
};

// Lower a validated S-Expr Document into a verified MLIR module and return its
// textual form.
//
// Internally this creates an MLIRContext, registers the func/arith/tensor/
// linalg dialects, lowers the graph into a single func.func, runs full MLIR
// verification, and prints the module. A successful return therefore implies
// that both lowering and MLIR verification succeeded; a Verification error is
// returned if the produced module does not verify.
std::expected<std::string, LoweringError> lower_to_mlir(const Document& doc);

} // namespace sonicboom::sx
