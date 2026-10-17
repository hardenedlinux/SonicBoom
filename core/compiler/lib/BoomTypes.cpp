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

// Boom IR types: the construction-time verifiers (ODS generated the get/getChecked
// declarations, not the bodies) and the quant-format helpers.

#include "sonicboom/compiler/BoomTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstddef>

#include <sonicboom/quant/quantized_tensor.h>
#include <sonicboom/sx/ir.h>

namespace sonicboom::compiler::boom {

//--- verified shape/dtype invariants ----------------------------------------

namespace {

// Every dimension is a static, non-negative extent, or MLIR's `?` dynamic
// marker (ShapedType::kDynamic). The dynamic axis is the graph's *extent* — the
// prefill segment length — and it belongs to the vocabulary because the plan a
// segment produces must be reusable across the lengths that share its kernel
// selections. The runtime's Shape is static (sx/ir.h: "Static, non-negative
// dimensions"); a dynamic dimension is bound to a concrete extent when the IR is
// lowered (BoomConversion.cpp, boomToGraph).
llvm::LogicalResult verifyDims(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  for (int64_t d : shape)
    if (d < 0 && d != mlir::ShapedType::kDynamic)
      return emitError() << "dimension " << d
                         << " is negative and is not the dynamic marker";
  return llvm::success();
}

// A quantized format serializes the block-contiguous (fastest-moving, i.e.
// last) dimension in whole blocks — the precondition QuantizedTensor
// documents for the gguf payload. Stating it here means an IR that violates it
// cannot be built, instead of failing inside a kernel.
template <typename QuantType>
llvm::LogicalResult verifyQuantShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  if (llvm::failed(verifyDims(emitError, shape)))
    return llvm::failure();
  if (shape.empty())
    return emitError() << "a " << QuantType::name
                       << " value must have at least one dimension";
  const int64_t block = QuantType::blockSize();
  // The block-contiguous dimension is a property of the format, not of the
  // graph's extent, so it can never be the dynamic axis.
  if (shape.back() == mlir::ShapedType::kDynamic)
    return emitError() << "the block-contiguous dimension of a "
                       << QuantType::name << " is dynamic, but a block size "
                       << "is a property of the format";
  if (shape.back() % block != 0)
    return emitError() << "the block-contiguous dimension of a "
                       << QuantType::name << " (" << shape.back()
                       << ") is not a multiple of the " << block
                       << "-element block of " << QuantType::formatName();
  return llvm::success();
}

} // namespace

// The frozen ten-name dtype set (sx/ir.h:39-50).
std::optional<sx::DType> boomDTypeFromName(llvm::StringRef name) {
  constexpr sx::DType kAll[] = {
      sx::DType::Float32, sx::DType::Float16, sx::DType::BFloat16,
      sx::DType::Float64, sx::DType::Int8,    sx::DType::UInt8,
      sx::DType::Int16,   sx::DType::Int32,   sx::DType::Int64,
      sx::DType::Bool,
  };
  for (sx::DType d : kAll)
    if (name == sx::dtype_name(d))
      return d;
  return std::nullopt;
}

llvm::LogicalResult TensorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::StringRef dtype) {
  if (llvm::failed(verifyDims(emitError, shape)))
    return llvm::failure();
  if (!boomDTypeFromName(dtype)) {
    return emitError() << "'" << dtype
                       << "' is not one of the frozen sx::DType names";
  }
  return llvm::success();
}

llvm::LogicalResult Q3KType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q3KType>(emitError, shape);
}

llvm::LogicalResult Q4KType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q4KType>(emitError, shape);
}

llvm::LogicalResult Q5KType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q5KType>(emitError, shape);
}

llvm::LogicalResult Q6KType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q6KType>(emitError, shape);
}

llvm::LogicalResult Q5_0Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q5_0Type>(emitError, shape);
}

llvm::LogicalResult Q8_0Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q8_0Type>(emitError, shape);
}

llvm::LogicalResult Q8_1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape) {
  return verifyQuantShape<Q8_1Type>(emitError, shape);
}

//--- quant-format helpers ---------------------------------------------------

bool isBoomWeightQuantType(mlir::Type type) {
  return llvm::isa<Q3KType, Q4KType, Q5KType, Q6KType, Q5_0Type, Q8_0Type>(
      type);
}

int boomQuantBlockSize(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, int>(type)
      .Case<Q3KType>([](auto) { return Q3KType::blockSize(); })
      .Case<Q4KType>([](auto) { return Q4KType::blockSize(); })
      .Case<Q5KType>([](auto) { return Q5KType::blockSize(); })
      .Case<Q6KType>([](auto) { return Q6KType::blockSize(); })
      .Case<Q5_0Type>([](auto) { return Q5_0Type::blockSize(); })
      .Case<Q8_0Type>([](auto) { return Q8_0Type::blockSize(); })
      .Case<Q8_1Type>([](auto) { return Q8_1Type::blockSize(); })
      .Default([](auto) { return 0; });
}

llvm::StringRef boomQuantFormatName(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, llvm::StringRef>(type)
      .Case<Q3KType>([](auto) { return Q3KType::formatName(); })
      .Case<Q4KType>([](auto) { return Q4KType::formatName(); })
      .Case<Q5KType>([](auto) { return Q5KType::formatName(); })
      .Case<Q6KType>([](auto) { return Q6KType::formatName(); })
      .Case<Q5_0Type>([](auto) { return Q5_0Type::formatName(); })
      .Case<Q8_0Type>([](auto) { return Q8_0Type::formatName(); })
      .Case<Q8_1Type>([](auto) { return Q8_1Type::formatName(); })
      .Default([](auto) { return llvm::StringRef(); });
}

} // namespace sonicboom::compiler::boom
