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

// SonicBoom S-Expr v0.1 → MLIR lowering.
//
// Turns a validated sx::Document (see ir.h / parser.h) into a verified MLIR
// module: one func.func per graph, graph inputs as arguments, graph outputs as
// return values, parameters as arith.constant (:values) or tensor.empty
// placeholders (:external, with sidecar metadata recorded on the module).
//
// Dialect set: builtin + func + arith + tensor + linalg. This is the minimum
// needed to lower the frozen v0 operator subset (conv/relu/add/max_pool/
// reduce_mean/reshape/gemm). No custom SonicBoom dialect is introduced.

#include "sonicboom/sx/lowering.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

namespace sonicboom::sx {
namespace {

using namespace mlir;

// ---------------------------------------------------------------------------
// Type lowering
// ---------------------------------------------------------------------------

Type lower_elem_type(MLIRContext& ctx, DType d) {
  switch (d) {
  case DType::Float32: return Float32Type::get(&ctx);
  case DType::Float16: return Float16Type::get(&ctx);
  case DType::BFloat16: return BFloat16Type::get(&ctx);
  case DType::Float64: return Float64Type::get(&ctx);
  // MLIR signless integer types. The frozen dtype set distinguishes signed
  // (Int8/16/32/64) from unsigned (UInt8), but no v0 operator depends on the
  // distinction, so signedness is not materialized here (documented limit).
  case DType::Int8:  return IntegerType::get(&ctx, 8);
  case DType::UInt8: return IntegerType::get(&ctx, 8);
  case DType::Int16: return IntegerType::get(&ctx, 16);
  case DType::Int32: return IntegerType::get(&ctx, 32);
  case DType::Int64: return IntegerType::get(&ctx, 64);
  case DType::Bool:  return IntegerType::get(&ctx, 1);
  }
  return Type();
}

RankedTensorType lower_tensor_type(MLIRContext& ctx, const TensorType& t) {
  return RankedTensorType::get(llvm::ArrayRef<int64_t>(t.shape.dims),
                               lower_elem_type(ctx, t.dtype));
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

const Attribute* find_attr(const Node& n, llvm::StringRef name) {
  for (const auto& a : n.attributes)
    if (a.name == name.str())
      return &a;
  return nullptr;
}

int64_t get_int_attr(const Node& n, llvm::StringRef name, int64_t def) {
  const Attribute* a = find_attr(n, name);
  if (a)
    if (const auto* v = std::get_if<int64_t>(&a->value))
      return *v;
  return def;
}

double get_float_attr(const Node& n, llvm::StringRef name, double def) {
  const Attribute* a = find_attr(n, name);
  if (a)
    if (const auto* v = std::get_if<double>(&a->value))
      return *v;
  return def;
}

std::vector<int64_t> get_ints_attr(const Node& n, llvm::StringRef name,
                                   std::vector<int64_t> def) {
  const Attribute* a = find_attr(n, name);
  if (a)
    if (const auto* v = std::get_if<std::vector<int64_t>>(&a->value))
      return *v;
  return def;
}

// ---------------------------------------------------------------------------
// Lowering context
// ---------------------------------------------------------------------------

struct Ctx {
  MLIRContext& ctx;
  const Graph& graph;
  OpBuilder& builder;
  Location loc;
  ModuleOp module;
  func::FuncOp func;
  llvm::DenseMap<llvm::StringRef, Value> values;          // SSA name → value
  llvm::DenseMap<llvm::StringRef, const Parameter*> params; // name → parameter

  Ctx(MLIRContext& c, const Graph& g, OpBuilder& b, Location l, ModuleOp m)
      : ctx(c), graph(g), builder(b), loc(l), module(m) {}
};

// ---------------------------------------------------------------------------
// Value builders
// ---------------------------------------------------------------------------

// A splat constant of `value` across the whole tensor type.
Value splat_constant(OpBuilder& b, Location loc, RankedTensorType t,
                     double value) {
  Type e = t.getElementType();
  TypedAttr dense;
  if (e.isF32())
    dense = DenseElementsAttr::get(t, static_cast<float>(value));
  else if (e.isF64())
    dense = DenseElementsAttr::get(t, value);
  else if (e.isInteger(64))
    dense = DenseElementsAttr::get(t, static_cast<int64_t>(value));
  else if (e.isInteger(32))
    dense = DenseElementsAttr::get(t, static_cast<int32_t>(value));
  else if (e.isInteger(16))
    dense = DenseElementsAttr::get(t, static_cast<int16_t>(value));
  else if (e.isInteger(8))
    dense = DenseElementsAttr::get(t, static_cast<int8_t>(value));
  else {
    APFloat v(value);
    bool ignored;
    v.convert(mlir::cast<FloatType>(e).getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &ignored);
    dense = DenseElementsAttr::get(t, llvm::ArrayRef<APFloat>{v});
  }
  return b.create<arith::ConstantOp>(loc, t, dense);
}

// A dense constant from a :values literal list (element type from `t`).
Value dense_constant(OpBuilder& b, Location loc, RankedTensorType t,
                     const ValuesData& data) {
  TypedAttr dense;
  if (mlir::isa<IntegerType>(t.getElementType())) {
    SmallVector<int64_t> vals;
    for (const auto& v : data.values) {
      if (const auto* i = std::get_if<int64_t>(&v))
        vals.push_back(*i);
      else
        vals.push_back(static_cast<int64_t>(std::get<double>(v)));
    }
    dense = DenseElementsAttr::get(t, llvm::ArrayRef<int64_t>(vals));
  } else {
    SmallVector<double> vals;
    for (const auto& v : data.values) {
      if (const auto* d = std::get_if<double>(&v))
        vals.push_back(*d);
      else
        vals.push_back(static_cast<double>(std::get<int64_t>(v)));
    }
    dense = DenseElementsAttr::get(t, llvm::ArrayRef<double>(vals));
  }
  return b.create<arith::ConstantOp>(loc, t, dense);
}

// An uninitialized tensor of the given type (used for :external placeholders
// and for linalg structured-op output init tensors).
Value empty_tensor(Ctx& c, RankedTensorType t) {
  return c.builder.create<tensor::EmptyOp>(c.loc, t.getShape(),
                                           t.getElementType());
}

// A scalar zero of element type `e` (for tensor.pad's constant region).
Value scalar_zero(OpBuilder& b, Location loc, Type e) {
  TypedAttr a = mlir::isa<FloatType>(e) ? TypedAttr(FloatAttr::get(e, 0.0))
                                   : TypedAttr(IntegerAttr::get(e, 0));
  return b.create<arith::ConstantOp>(loc, e, a);
}

// ---------------------------------------------------------------------------
// Reusable structured-op helpers
// ---------------------------------------------------------------------------

// 2-D transpose (used by gemm when transA/transB are set).
Value transpose_2d(Ctx& c, Value v) {
  auto t = mlir::cast<RankedTensorType>(v.getType());
  auto shp = t.getShape();
  auto outT = RankedTensorType::get({shp[1], shp[0]}, t.getElementType());
  Value init = empty_tensor(c, outT);
  return c.builder
      .create<linalg::TransposeOp>(c.loc, v, init,
                                   llvm::ArrayRef<int64_t>{1, 0})
      .getResults()[0];
}

// Element-wise add broadcasting a rank-1 `rhs` along `axis` of `lhs`.
// Used for conv bias ([F] → [N,F,OH,OW], axis 1) and gemm bias ([N] → [M,N],
// axis 1).
Value broadcast_add_axis(Ctx& c, Value lhs, Value rhs, unsigned axis,
                         RankedTensorType outType) {
  unsigned rank = outType.getRank();
  SmallVector<AffineExpr, 4> identity;
  for (unsigned i = 0; i < rank; ++i)
    identity.push_back(getAffineDimExpr(i, &c.ctx));
  AffineMap lhsMap = AffineMap::get(rank, 0, identity, &c.ctx);
  AffineMap rhsMap =
      AffineMap::get(rank, 0, getAffineDimExpr(axis, &c.ctx), &c.ctx);
  AffineMap outMap = lhsMap;
  SmallVector<utils::IteratorType, 4> iters(rank, utils::IteratorType::parallel);
  Value init = empty_tensor(c, outType);
  auto op = c.builder.create<linalg::GenericOp>(
      c.loc, TypeRange{outType}, ValueRange{lhs, rhs}, ValueRange{init},
      llvm::ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, iters,
      [](OpBuilder& b, Location loc, ValueRange args) {
        Value s = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, s);
      });
  return op.getResult(0);
}

// Zero-pad a tensor along the given per-dim low/high amounts.
Value pad_zeros(Ctx& c, Value src, RankedTensorType resultType,
                llvm::ArrayRef<int64_t> low, llvm::ArrayRef<int64_t> high) {
  SmallVector<OpFoldResult> lowOfr, highOfr;
  for (int64_t p : low)
    lowOfr.push_back(c.builder.getI64IntegerAttr(p));
  for (int64_t p : high)
    highOfr.push_back(c.builder.getI64IntegerAttr(p));
  Value padVal = scalar_zero(c.builder, c.loc, resultType.getElementType());
  return c.builder
      .create<tensor::PadOp>(c.loc, resultType, src, lowOfr, highOfr, padVal)
      .getResult();
}

// ---------------------------------------------------------------------------
// Operator lowering
// ---------------------------------------------------------------------------

// relu(x) = maximumf(x, 0.0)
std::string lower_relu(Ctx& c, const Node& node, llvm::ArrayRef<Value> inputs) {
  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);
  Value zero = splat_constant(c.builder, c.loc, outType, 0.0);
  Value r = c.builder.create<arith::MaximumFOp>(c.loc, inputs[0], zero);
  c.values[node.outputs[0].name] = r;
  return "";
}

// add(a, b) = addf(a, b)
std::string lower_add(Ctx& c, const Node& node, llvm::ArrayRef<Value> inputs) {
  Value r = c.builder.create<arith::AddFOp>(c.loc, inputs[0], inputs[1]);
  c.values[node.outputs[0].name] = r;
  return "";
}

// reshape(x, shape) = tensor.reshape using the output type's static shape.
std::string lower_reshape(Ctx& c, const Node& node,
                          llvm::ArrayRef<Value> inputs) {
  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);
  const auto& dims = node.outputs[0].type.shape.dims;
  auto shapeType = RankedTensorType::get({static_cast<int64_t>(dims.size())},
                                         IntegerType::get(&c.ctx, 64));
  auto shapeAttr =
      DenseElementsAttr::get(shapeType, llvm::ArrayRef<int64_t>(dims));
  Value shapeVal = c.builder.create<arith::ConstantOp>(c.loc, shapeType,
                                                       shapeAttr);
  Value r = c.builder.create<tensor::ReshapeOp>(c.loc, outType, inputs[0],
                                                shapeVal);
  c.values[node.outputs[0].name] = r;
  return "";
}

// conv(x, w, bias): linalg.conv_2d_nchw_fchw (+ explicit zero padding) then a
// channel-axis bias add. group == 1 and dilations == 1 only (v0).
std::string lower_conv(Ctx& c, const Node& node,
                       llvm::ArrayRef<Value> inputs) {
  if (inputs.size() != 3)
    return "conv expects 3 inputs (x, weight, bias)";

  if (get_int_attr(node, "group", 1) != 1)
    return "conv with group != 1 is unsupported in v0";
  auto dilations = get_ints_attr(node, "dilations", {1, 1});
  if (dilations[0] != 1 || dilations[1] != 1)
    return "conv with dilations != 1 is unsupported in v0";
  auto pads = get_ints_attr(node, "pads", {0, 0, 0, 0});
  if (pads.size() != 4)
    return "conv pads must have 4 entries";
  auto strides = get_ints_attr(node, "strides", {1, 1});

  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);
  auto inType = mlir::cast<RankedTensorType>(inputs[0].getType());
  auto inShape = inType.getShape();   // [N, C, IH, IW]

  int64_t sh = strides[0], sw = strides[1];

  // Pad by the exact ONNX pads [top, left, bottom, right], then run linalg's
  // valid convolution; its own floor semantics reproduce ONNX's floor output.
  Value x = inputs[0];
  if (pads[0] || pads[1] || pads[2] || pads[3]) {
    auto paddedType = RankedTensorType::get(
        {inShape[0], inShape[1], inShape[2] + pads[0] + pads[2],
         inShape[3] + pads[1] + pads[3]},
        inType.getElementType());
    x = pad_zeros(c, x, paddedType, {0, 0, pads[0], pads[1]},
                  {0, 0, pads[2], pads[3]});
  }

  Value init = empty_tensor(c, outType);
  auto stridesAttr = c.builder.getI64VectorAttr({sh, sw});
  auto dilationsAttr = c.builder.getI64VectorAttr({1, 1});
  Value conv = c.builder
                   .create<linalg::Conv2DNchwFchwOp>(
                       c.loc, TypeRange{outType}, ValueRange{x, inputs[1]},
                       ValueRange{init}, stridesAttr, dilationsAttr)
                   .getResult(0);

  c.values[node.outputs[0].name] =
      broadcast_add_axis(c, conv, inputs[2], /*axis=*/1, outType);
  return "";
}

// max_pool(x): linalg.pooling_nchw_max with explicit zero padding (the kernel
// is carried as a [KH,KW] operand, unused by the max reduction).
std::string lower_max_pool(Ctx& c, const Node& node,
                           llvm::ArrayRef<Value> inputs) {
  if (get_int_attr(node, "ceil_mode", 0) != 0)
    return "max_pool with ceil_mode != 0 is unsupported in v0";
  auto dilations = get_ints_attr(node, "dilations", {1, 1});
  if (dilations[0] != 1 || dilations[1] != 1)
    return "max_pool with dilations != 1 is unsupported in v0";
  auto kernel = get_ints_attr(node, "kernel_shape", {});
  if (kernel.size() != 2)
    return "max_pool requires a 2-entry kernel_shape";
  auto pads = get_ints_attr(node, "pads", {0, 0, 0, 0});
  if (pads.size() != 4)
    return "max_pool pads must have 4 entries";
  auto strides = get_ints_attr(node, "strides", {1, 1});

  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);
  auto inType = mlir::cast<RankedTensorType>(inputs[0].getType());
  auto inShape = inType.getShape();   // [N, C, IH, IW]

  int64_t sh = strides[0], sw = strides[1];
  int64_t kh = kernel[0], kw = kernel[1];

  // Pad by the exact ONNX pads [top, left, bottom, right], then run linalg's
  // valid pooling; its own floor semantics reproduce ONNX's floor output.
  Value x = inputs[0];
  if (pads[0] || pads[1] || pads[2] || pads[3]) {
    auto paddedType = RankedTensorType::get(
        {inShape[0], inShape[1], inShape[2] + pads[0] + pads[2],
         inShape[3] + pads[1] + pads[3]},
        inType.getElementType());
    x = pad_zeros(c, x, paddedType, {0, 0, pads[0], pads[1]},
                  {0, 0, pads[2], pads[3]});
  }

  // Dummy [KH,KW] kernel operand (shape drives the reduction; unused by max).
  Value kernelInit = empty_tensor(
      c, RankedTensorType::get({kh, kw}, inType.getElementType()));

  Value init = empty_tensor(c, outType);
  auto stridesAttr = c.builder.getI64VectorAttr({sh, sw});
  auto dilationsAttr = c.builder.getI64VectorAttr({1, 1});
  Value pool = c.builder
                   .create<linalg::PoolingNchwMaxOp>(
                       c.loc, TypeRange{outType}, ValueRange{x, kernelInit},
                       ValueRange{init}, stridesAttr, dilationsAttr)
                   .getResult(0);

  c.values[node.outputs[0].name] = pool;
  return "";
}

// reduce_mean(x, axes): sum over the given axes via linalg.generic, then divide
// by the number of reduced elements. axes come from the literal :values int64
// parameter named by inputs[1]; keepdims is preserved.
std::string lower_reduce_mean(Ctx& c, const Node& node,
                              llvm::ArrayRef<Value> inputs) {
  if (inputs.size() < 2)
    return "reduce_mean requires an explicit axes input in v0";

  std::vector<int64_t> axes;
  auto pit = c.params.find(llvm::StringRef(node.inputs[1]));
  if (pit == c.params.end())
    return "reduce_mean axes must reference a parameter";
  const auto* vd = std::get_if<ValuesData>(&pit->second->data);
  if (!vd)
    return "reduce_mean axes must be a :values int64 parameter";
  for (const auto& v : vd->values) {
    if (const auto* i = std::get_if<int64_t>(&v))
      axes.push_back(*i);
    else
      return "reduce_mean axes must be integer";
  }

  auto inType = mlir::cast<RankedTensorType>(inputs[0].getType());
  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);
  unsigned rank = inType.getRank();

  std::vector<bool> reduced(rank, false);
  for (int64_t a : axes) {
    if (a < 0)
      a += static_cast<int64_t>(rank);
    if (a < 0 || a >= static_cast<int64_t>(rank))
      return "reduce_mean axis out of range";
    reduced[static_cast<unsigned>(a)] = true;
  }

  SmallVector<utils::IteratorType, 4> iters;
  SmallVector<AffineExpr, 4> inExprs, outExprs;
  int64_t count = 1;
  for (unsigned i = 0; i < rank; ++i) {
    inExprs.push_back(getAffineDimExpr(i, &c.ctx));
    if (reduced[i]) {
      iters.push_back(utils::IteratorType::reduction);
      outExprs.push_back(getAffineConstantExpr(0, &c.ctx));
      count *= inType.getShape()[i];
    } else {
      iters.push_back(utils::IteratorType::parallel);
      outExprs.push_back(getAffineDimExpr(i, &c.ctx));
    }
  }
  AffineMap inMap = AffineMap::get(rank, 0, inExprs, &c.ctx);
  AffineMap outMap = AffineMap::get(rank, 0, outExprs, &c.ctx);

  Value init = empty_tensor(c, outType);
  auto op = c.builder.create<linalg::GenericOp>(
      c.loc, TypeRange{outType}, ValueRange{inputs[0]}, ValueRange{init},
      llvm::ArrayRef<AffineMap>{inMap, outMap}, iters,
      [](OpBuilder& b, Location loc, ValueRange args) {
        Value s = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, s);
      });
  Value sum = op.getResult(0);

  if (count != 1) {
    Value cst = splat_constant(c.builder, c.loc, outType,
                               static_cast<double>(count));
    sum = c.builder.create<arith::DivFOp>(c.loc, sum, cst);
  }

  c.values[node.outputs[0].name] = sum;
  return "";
}

// gemm(A, B[, C]) = alpha * A' @ B' + beta * C, with linalg.transpose for the
// transA/transB flags and a trailing-axis broadcast add for the bias.
std::string lower_gemm(Ctx& c, const Node& node,
                       llvm::ArrayRef<Value> inputs) {
  if (inputs.size() < 2 || inputs.size() > 3)
    return "gemm expects 2 or 3 inputs";

  Value a = inputs[0], b = inputs[1];
  bool hasBias = inputs.size() == 3;
  double alpha = get_float_attr(node, "alpha", 1.0);
  double beta = get_float_attr(node, "beta", 1.0);
  bool transA = get_int_attr(node, "transA", 0) != 0;
  bool transB = get_int_attr(node, "transB", 0) != 0;

  auto outType = lower_tensor_type(c.ctx, node.outputs[0].type);

  if (transA)
    a = transpose_2d(c, a);
  if (transB)
    b = transpose_2d(c, b);

  Value init = empty_tensor(c, outType);
  Value mm = c.builder
                 .create<linalg::MatmulOp>(c.loc, TypeRange{outType},
                                           ValueRange{a, b}, ValueRange{init})
                 .getResult(0);

  if (alpha != 1.0) {
    Value s = splat_constant(c.builder, c.loc, outType, alpha);
    mm = c.builder.create<arith::MulFOp>(c.loc, mm, s);
  }

  Value result = mm;
  if (hasBias) {
    Value bias = inputs[2];
    if (beta != 1.0) {
      auto biasType = mlir::cast<RankedTensorType>(bias.getType());
      Value s = splat_constant(c.builder, c.loc, biasType, beta);
      bias = c.builder.create<arith::MulFOp>(c.loc, bias, s);
    }
    result = broadcast_add_axis(c, mm, bias, /*axis=*/1, outType);
  }

  c.values[node.outputs[0].name] = result;
  return "";
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

using LowerFn = std::string (*)(Ctx&, const Node&, llvm::ArrayRef<Value>);

const std::unordered_map<std::string, LowerFn>& lowerers() {
  static const std::unordered_map<std::string, LowerFn> map = {
      {"relu", lower_relu},
      {"add", lower_add},
      {"reshape", lower_reshape},
      {"conv", lower_conv},
      {"max_pool", lower_max_pool},
      {"reduce_mean", lower_reduce_mean},
      {"gemm", lower_gemm},
  };
  return map;
}

std::string lower_node(Ctx& c, const Node& node) {
  if (node.outputs.size() != 1)
    return "operator '" + node.op_name + "' must have exactly one output in v0";

  SmallVector<Value, 4> inputs;
  for (const auto& in : node.inputs) {
    Value v = c.values.lookup(llvm::StringRef(in));
    if (!v)
      return "reference to undefined value '" + in + "'";
    inputs.push_back(v);
  }

  auto it = lowerers().find(node.op_name);
  if (it == lowerers().end())
    return "unsupported operator '" + node.op_name + "'";

  return it->second(c, node, inputs);
}

// ---------------------------------------------------------------------------
// Graph lowering
// ---------------------------------------------------------------------------

// Record :external parameter sidecar metadata on the module as a dictionary
// name → {file, offset, length}. Identity/type stay on the tensor.empty value.
void record_external_data(Ctx& c) {
  SmallVector<NamedAttribute> entries;
  auto i64 = IntegerType::get(&c.ctx, 64);
  for (const auto& p : c.graph.parameters) {
    const auto* ext = std::get_if<ExternalData>(&p.data);
    if (!ext)
      continue;
    auto dict = DictionaryAttr::get(
        &c.ctx,
        {NamedAttribute(StringAttr::get(&c.ctx, "file"),
                        StringAttr::get(&c.ctx, ext->file)),
         NamedAttribute(StringAttr::get(&c.ctx, "offset"),
                        IntegerAttr::get(i64, ext->offset)),
         NamedAttribute(StringAttr::get(&c.ctx, "length"),
                        IntegerAttr::get(i64, ext->length))});
    entries.push_back(
        NamedAttribute(StringAttr::get(&c.ctx, p.name), dict));
  }
  if (!entries.empty())
    c.module->setAttr("sonicboom.external_data",
                      DictionaryAttr::get(&c.ctx, entries));
}

// Lower one graph into a func.func inside the module. Returns "" on success or
// an error message.
std::string lower_graph(Ctx& c) {
  const Graph& g = c.graph;

  // name → type, from every SSA definition (inputs, params, node outputs).
  llvm::DenseMap<llvm::StringRef, const TensorType*> defTypes;
  for (const auto& in : g.inputs)
    defTypes[in.name] = &in.type;
  for (const auto& p : g.parameters)
    defTypes[p.name] = &p.type;
  for (const auto& n : g.nodes)
    for (const auto& o : n.outputs)
      defTypes[o.name] = &o.type;

  SmallVector<Type> inputTypes;
  for (const auto& in : g.inputs)
    inputTypes.push_back(lower_tensor_type(c.ctx, in.type));

  SmallVector<Type> resultTypes;
  for (const auto& out : g.outputs) {
    auto it = defTypes.find(out);
    if (it == defTypes.end())
      return "graph output references undefined value '" + out + "'";
    resultTypes.push_back(lower_tensor_type(c.ctx, *it->second));
  }

  auto funcType = FunctionType::get(&c.ctx, inputTypes, resultTypes);
  c.func = c.builder.create<func::FuncOp>(c.loc, g.name, funcType);
  Block* entry = c.func.addEntryBlock();
  c.builder.setInsertionPointToStart(entry);

  for (size_t i = 0; i < g.inputs.size(); ++i)
    c.values[g.inputs[i].name] = entry->getArgument(i);

  for (const auto& p : g.parameters) {
    auto t = lower_tensor_type(c.ctx, p.type);
    if (std::holds_alternative<ExternalData>(p.data)) {
      // Placeholder: uninitialized tensor of the right type; sidecar metadata
      // is recorded on the module (record_external_data).
      c.values[p.name] = empty_tensor(c, t);
    } else {
      c.values[p.name] =
          dense_constant(c.builder, c.loc, t, std::get<ValuesData>(p.data));
    }
  }

  for (const auto& node : g.nodes) {
    std::string err = lower_node(c, node);
    if (!err.empty())
      return err;
  }

  SmallVector<Value> retVals;
  for (const auto& out : g.outputs) {
    Value v = c.values.lookup(llvm::StringRef(out));
    if (!v)
      return "graph output references undefined value '" + out + "'";
    retVals.push_back(v);
  }
  c.builder.create<func::ReturnOp>(c.loc, retVals);
  return "";
}

} // namespace

std::expected<std::string, LoweringError> lower_to_mlir(const Document& doc) {
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();

  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);

  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module->getBody());

  Ctx c(context, doc.graph, builder, loc, *module);
  for (const auto& p : doc.graph.parameters)
    c.params[p.name] = &p;

  std::string err = lower_graph(c);
  if (!err.empty())
    return std::unexpected(LoweringError{LoweringErrorKind::Operator, err});

  record_external_data(c);

  std::string diagMsg;
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& d) {
      llvm::raw_string_ostream os(diagMsg);
      os << d;
      return success();
    });
    if (failed(verify(*module)))
      return std::unexpected(LoweringError{
          LoweringErrorKind::Verification,
          diagMsg.empty() ? "module failed MLIR verification" : diagMsg});
  }

  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os);
  return out;
}

} // namespace sonicboom::sx
