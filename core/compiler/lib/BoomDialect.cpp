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

// The Boom dialect's registration and assembly: the generated definitions all
// live in this one translation unit (the standard MLIR arrangement — the
// dialect's generated parseType/printType/parseAttribute/printAttribute bodies
// each need the generated dispatch tables, and initialize() needs the op, type
// and attribute lists). The hand-written verifier lives in BoomVerify.cpp.

#include "sonicboom/compiler/BoomDialect.h"

#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"  // interleaveComma (printDimList)
#include "llvm/ADT/TypeSwitch.h"

using namespace sonicboom::compiler::boom;

//--- generated: dialect, node attributes, types, ops -------------------------

#include "sonicboom/compiler/BoomDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "sonicboom/compiler/BoomAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "sonicboom/compiler/BoomTypes.cpp.inc"

#define GET_OP_CLASSES
#include "sonicboom/compiler/BoomOps.cpp.inc"

//--- registration -----------------------------------------------------------

void BoomDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "sonicboom/compiler/BoomAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "sonicboom/compiler/BoomTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "sonicboom/compiler/BoomOps.cpp.inc"
      >();
}

//--- shared shape syntax ----------------------------------------------------

namespace sonicboom::compiler::boom {

// `[4096, 2048]`, `[?, 2048]` for a dynamic leading extent, or `[]` for a
// scalar. Printed and parsed by hand rather than through a parameter's default,
// because ArrayRef<int64_t> has no FieldParser/printer in MLIR (the EmitC array
// type writes the same helper). Defined inside the namespace the generated
// parse/print bodies call them from — the file's `using namespace` is not a
// declaration, so a definition at file scope would declare a different, global
// function.
mlir::LogicalResult parseDimList(mlir::AsmParser &parser,
                                 llvm::SmallVector<int64_t> &dims) {
  return parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                        [&]() -> mlir::ParseResult {
                                          if (parser.parseOptionalQuestion()
                                                  .succeeded()) {
                                            dims.push_back(
                                                mlir::ShapedType::kDynamic);
                                            return mlir::success();
                                          }
                                          int64_t dim = 0;
                                          if (parser.parseInteger(dim))
                                            return mlir::failure();
                                          dims.push_back(dim);
                                          return mlir::success();
                                        });
}

void printDimList(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> dims) {
  printer << '[';
  llvm::interleaveComma(dims, printer, [&](int64_t dim) {
    if (dim == mlir::ShapedType::kDynamic)
      printer << '?';
    else
      printer << dim;
  });
  printer << ']';
}

} // namespace sonicboom::compiler::boom
