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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <sonicboom/sx/ir.h>

// Blade (sx::Document) -> Boom IR (design/mlir-migration.md §6.4, §7d).
//
// The lowering reads the frozen Blade v0.1 document directly: tensor ids are
// assigned in the same document order adapt_graph assigns them (inputs,
// parameters, node outputs), every tensor no node produces becomes a block
// argument in ascending id order, and each node becomes one `boom.<op>` with
// the node record both IR constructors share (`createBoomNode`). The IR is
// therefore byte-identical to `graphToBoom(adapt_graph(doc))` — that equality is
// the acceptance test — but no planner::Graph is built in between.
//
// Returns an empty module, with a diagnostic on `context`, when the document
// cannot be lowered under the same rules adapt_graph enforces: an unknown
// operator, a duplicate value definition, a reference to an undefined value, a
// byte size that overflows, a dtype or attribute value Boom IR cannot spell, or
// IR that fails verification.
namespace sonicboom::compiler {

mlir::OwningOpRef<mlir::ModuleOp> bladeToBoom(const sx::Document &doc,
                                              mlir::MLIRContext *context);

} // namespace sonicboom::compiler
