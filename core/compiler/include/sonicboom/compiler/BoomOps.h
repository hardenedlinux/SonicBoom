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

// Boom IR operations: the graph container and one op per planner::OpKind.

#pragma once

// What the generated op classes are built out of: mlir::Op, the trait bases,
// and the bytecode interface they carry (which is also how a generically built
// op is round-tripped through properties). BytecodeOpInterface.h brings
// OpDefinition.h with it.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "sonicboom/compiler/BoomTypes.h"

// The generated op classes. The define is consumed by the include, so a
// translation unit that also includes BoomOps.cpp.inc (the definitions)
// re-defines it itself.
#define GET_OP_CLASSES
#include "sonicboom/compiler/BoomOps.h.inc"

namespace sonicboom::compiler::boom {

// The attribute names that every Boom op declares as a typed attribute, in the
// order ODS declares them. This is the set the typed view and `node_attrs` are
// checked against: `node_attrs` is the authoritative record (it is what the
// round trip reproduces, in order), and these are the same values reachable
// through a typed accessor, which is what a pass wants to read.
llvm::ArrayRef<llvm::StringRef> boomTypedAttrNames();

// True when `name` is one of boomTypedAttrNames().
bool isBoomTypedAttrName(llvm::StringRef name);

// The typed attributes `op` actually carries, as (name, value) pairs.
llvm::SmallVector<mlir::NamedAttribute> typedNodeAttributes(mlir::Operation *op);

} // namespace sonicboom::compiler::boom
