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

// The Boom dialect (Boom IR), the node-parameter attributes and the KV-cache
// effect resource. Compiler-only header: it lives under
// core/compiler/include/, never under core/include/, so no MLIR type can reach
// the Layer 2 boundary (docs/boom-ir.md).

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sonicboom/compiler/BoomDialect.h.inc"

// The generated attribute classes, and the generated
// BoomDialect::parseAttribute/printAttribute along with them. The define is
// consumed by the include (`#undef` in the generated file), so a translation
// unit that also includes BoomAttrs.cpp.inc re-defines it itself.
#define GET_ATTRDEF_CLASSES
#include "sonicboom/compiler/BoomAttrs.h.inc"

namespace sonicboom::compiler::boom {

// The per-layer KV cache as a memory-effect resource (§6.3). The IR names a
// layer's cache with a `!boom.kv<layer>` handle (BoomTypes.td) and states the
// access as an effect on this resource; the cache's storage stays in the
// backend, keyed by layer. Stating the access is what lets MLIR order the
// decode step's write against the next step's read instead of relying on "the
// executor walks the nodes in order".
struct KVCacheResource
    : public mlir::SideEffects::Resource::Base<KVCacheResource> {
  llvm::StringRef getName() const final { return "boom KV cache"; }
};

// `custom<DimList>` — the shape syntax shared by every Boom type and by the
// tensor-table attribute: `[4096, 2048]` (an empty shape, `[]`, is a scalar).
// Declared here because the generated parse/print bodies refer to it
// unqualified from inside this namespace.
mlir::LogicalResult parseDimList(mlir::AsmParser &parser,
                                 llvm::SmallVector<int64_t> &dims);
void printDimList(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> dims);

} // namespace sonicboom::compiler::boom
