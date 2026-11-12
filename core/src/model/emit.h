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

// The emitter: a recipe (model/recipe.h) written as Boom IR for one phase.
//
// Internal to the model layer: the IR it returns is an MLIR module, which does
// not cross into the public headers.

#include <sonicboom/model/plan.h>
#include <sonicboom/model/recipe.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <memory>

namespace sonicboom::model {

// An emitted module with the context that owns it. The context is declared
// first so it is destroyed last.
struct GraphIr {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

// One decode step over `spec.rows` query positions (more than one row is the
// verify step: the same graph dealt position-major, each node marked
// `verify`). The graph ends at the model's logits and declares `spec.consumer`
// as what reads them; which graph that consumer needs is the compiler's
// derivation (compiler/BoomPasses.h resolveConsumer), not the emitter's.
GraphIr emit_decode_ir(const Recipe& recipe, const DecodeGraphSpec& spec);

// One prefill segment of `n` positions; over a dynamic extent when
// `dynamic_extent`, with `n` the extent the graph is built at. The graph ends
// at the residual stream and appends every owned cache's K and V after the
// last block.
GraphIr emit_prefill_ir(const Recipe& recipe, uint64_t n,
                        bool dynamic_extent = false);

// Whether a decode or verify matmul can take the mmq lowering in this process:
// the q8_1 dispatcher and the mmq kernel are both available. Which projections
// do is the per-shape family measured at load.
bool decode_mmq_available();

} // namespace sonicboom::model
