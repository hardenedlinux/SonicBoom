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

// Boom IR value types: !boom.tensor, the quantized formats, and !boom.q8_1.

#pragma once

#include "sonicboom/compiler/BoomDialect.h"

// sx::DType, for boomDTypeFromName. The compiler library sits above Layer 2, so
// the planner's and sx's types are the vocabulary it works in.
#include <sonicboom/sx/ir.h>

#include <optional>

// The generated type classes (and BoomDialect::parseType/printType with them);
// the define is consumed by the include.
#define GET_TYPEDEF_CLASSES
#include "sonicboom/compiler/BoomTypes.h.inc"

namespace sonicboom::compiler::boom {

// True for every type this dialect uses as an SSA value type — i.e. what
// Boom_ValueType admits as an operand/result. The generated ops already reject
// anything else through their operand constraints; this is the same predicate
// available to code that has a bare mlir::Type.
inline bool isBoomValueType(mlir::Type type) {
  return llvm::isa<TensorType, Q8_1Type, Q4KType, Q5KType, Q6KType, Q5_0Type,
                   Q8_0Type>(type);
}

// The quantized *weight* formats, in the order quant::QuantType lists them
// (quant/quantized_tensor.h). `!boom.q8_1` is deliberately not here: it is an
// activation layout, not a stored weight format.
bool isBoomWeightQuantType(mlir::Type type);

// The number of elements per serialized block for a quantized Boom type, or 0
// when `type` is not a quantized Boom type. Mirrors QuantTypeInfo::block_size
// for the weight formats; for !boom.q8_1 it is the 32-element block the
// decode-side activation quantizer packs.
int boomQuantBlockSize(mlir::Type type);

// The quant::QuantTypeInfo::name of a quantized Boom type ("q4_K", "q8_1", …),
// or an empty StringRef when `type` is not a quantized Boom type.
llvm::StringRef boomQuantFormatName(mlir::Type type);

// sx::dtype_name's inverse: the sx::DType that prints as `name`, or nullopt when
// `name` is outside the frozen ten-name set. The names are matched against
// sx::dtype_name() rather than a second copy of the strings, so `!boom.tensor`
// and `TensorDesc::dtype` cannot disagree about what "bfloat16" is spelled.
std::optional<sx::DType> boomDTypeFromName(llvm::StringRef name);

} // namespace sonicboom::compiler::boom
