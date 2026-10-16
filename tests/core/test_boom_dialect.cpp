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

// The Boom IR dialect, checked on its own (compiler-side test: it links
// sonicboom_compiler, not the runtime path):
//
//   * every planner::OpKind has an op, named after op_kind_name(), and the
//     mnemonic maps back to the same OpKind — the drift check between the
//     planner's operator vocabulary and the IR that must represent it;
//   * the generated assembly formats round-trip: print → parse → the same
//     type/attribute;
//   * the construction-time verifiers reject what the runtime cannot execute (a
//     negative dimension, an unknown dtype, a quantized shape that is not a
//     whole number of blocks);
//   * the quant-format set is quant::QuantType's table (quant/quantized_tensor.h)
//     — same names, same block sizes, no second list to drift from it.
//
// The graph op's verifier is exercised by the conversion test, which builds
// real graphs from real models; here there is nothing but the dialect.

#include "sonicboom/compiler/BoomOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "sonicboom/planner/op_kind.h"
#include "sonicboom/quant/quantized_tensor.h"

#include <cctype>
#include <iostream>
#include <string>

namespace boom = sonicboom::compiler::boom;
namespace planner = sonicboom::planner;
namespace quant = sonicboom::quant;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::cout << "FAIL: " << what << "\n";
  }
}

// The text form, as the dialect's own printer writes it. Through the base
// class: the generated classes declare print(AsmPrinter&), which hides
// Type/Attribute's print(raw_ostream&). Nested attribute types are elided, the
// way the IR prints them.
std::string printed(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return os.str();
}

std::string printed(mlir::Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os, /*elideType=*/true);
  return os.str();
}

} // namespace

int main() {
  mlir::MLIRContext context;

  // Some of the checks below diagnose on purpose; count the errors instead of
  // letting the default handler print them.
  int errors = 0;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    if (d.getSeverity() == mlir::DiagnosticSeverity::Error)
      ++errors;
  });

  auto *dialect = context.getOrLoadDialect<boom::BoomDialect>();
  check(dialect != nullptr, "the boom dialect registers");

  //--- one op per OpKind, named after it -----------------------------------

  // The enum is contiguous (Conv = 0 .. KvAppend), so looping the integers
  // covers the vocabulary; the reverse mapping is the one the conversion uses
  // to rebuild a planner node from an op.
  const int kinds = int(planner::OpKind::KvAppend) + 1;
  for (int i = 0; i < kinds; ++i) {
    const auto kind = planner::OpKind(i);
    const std::string name = std::string("boom.") + planner::op_kind_name(kind);
    check(mlir::OperationName(name, &context).isRegistered(),
          name + " is registered");
    // The reverse map covers the node vocabulary. WholeGraph is a
    // task-granularity marker rather than a node operator (planner/op_kind.h)
    // and is deliberately not nameable back — a graph node is never one, so the
    // conversion back to a planner node has nothing to special-case.
    const auto back = planner::op_kind_from_name(name.substr(5));
    if (kind == planner::OpKind::WholeGraph)
      check(!back.has_value(), name + " stays a task marker, not a node op");
    else
      check(back.has_value() && *back == kind,
            name + " maps back to its OpKind");
  }
  check(mlir::OperationName("boom.graph", &context).isRegistered(),
        "boom.graph is registered");

  //--- types: print, parse, identical --------------------------------------

  const auto tensor = boom::TensorType::get(&context, {4096, 2048}, "float32");
  check(bool(tensor), "a !boom.tensor builds");
  check(mlir::parseType(printed(tensor), &context) == tensor,
        "!boom.tensor round-trips through its assembly format: " +
            printed(tensor));

  const auto q4k = boom::Q4KType::get(&context, {2560, 256});
  check(bool(q4k), "a !boom.q4_k builds");
  check(mlir::parseType(printed(q4k), &context) == q4k,
        "!boom.q4_k round-trips through its assembly format: " + printed(q4k));

  const auto q8_1 = boom::Q8_1Type::get(&context, {2560});
  check(bool(q8_1), "a !boom.q8_1 builds");
  check(mlir::parseType(printed(q8_1), &context) == q8_1,
        "!boom.q8_1 round-trips through its assembly format: " + printed(q8_1));

  // 32-element formats and the K-quant 256-element ones share the syntax.
  check(mlir::parseType(printed(boom::Q5_0Type::get(&context, {32})), &context) ==
            boom::Q5_0Type::get(&context, {32}),
        "!boom.q5_0 round-trips through its assembly format");

  // The format set is quant::QuantType's, not a second list that can drift from
  // it (quant/quantized_tensor.h). For every format that table lists, the
  // dialect must have a weight type whose mnemonic is the format name
  // lowercased, which spells that same name and uses the table's block size.
  // q3_K is in the table and was missing here; this is the check that catches
  // the next one.
  for (int i = int(quant::QuantType::Q3_K); i <= int(quant::QuantType::Q8_0);
       ++i) {
    const quant::QuantTypeInfo* info = quant::quant_type_info(quant::QuantType(i));
    check(info != nullptr, "the QuantType table has an entry for index " +
                               std::to_string(i));
    if (!info) continue;
    std::string mnemonic = info->name;
    for (char& c : mnemonic) c = char(std::tolower(c));
    // One block is a legal logical shape in every format.
    const std::string text =
        "!boom." + mnemonic + "<[" + std::to_string(info->block_size) + "]>";
    mlir::Type parsed = mlir::parseType(text, &context);
    check(bool(parsed), text + " is a type the dialect knows");
    if (!parsed) continue;
    check(boom::isBoomWeightQuantType(parsed), text + " is a weight quant type");
    check(boom::boomQuantFormatName(parsed) == info->name,
          text + " spells the format '" + std::string(info->name) + "'");
    check(boom::boomQuantBlockSize(parsed) == int(info->block_size),
          text + " uses the table's block size " +
              std::to_string(info->block_size));
  }

  //--- attributes: print, parse, identical ---------------------------------

  const auto desc = boom::TensorDescAttr::get(
      &context, /*id=*/7, {2560, 4096}, "q4_K", /*byte_size=*/1441792,
      /*is_input=*/false, /*is_output=*/false, /*is_constant=*/true,
      /*is_external=*/false, "/model/layers.0/mlp/gate_proj");
  check(bool(desc), "a #boom.tensor_desc builds");
  check(mlir::parseAttribute(printed(desc), &context) == desc,
        "#boom.tensor_desc round-trips: " + printed(desc));

  const auto node_attr = boom::NodeAttr::get(
      &context, "layer", mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 3));
  check(bool(node_attr), "a #boom.attr builds");
  check(mlir::parseAttribute(printed(node_attr), &context) == node_attr,
        "#boom.attr round-trips: " + printed(node_attr));

  // The typed-attribute names are the vocabulary a node's `node_attrs` and its
  // typed accessors must agree on.
  for (llvm::StringRef name : boom::boomTypedAttrNames())
    check(boom::isBoomTypedAttrName(name), name.str() + " is a typed attribute");
  check(!boom::isBoomTypedAttrName("not_an_attribute"),
        "an unknown name is not a typed attribute");

  // Nothing above should have diagnosed: a round trip that emits an error has
  // already been caught by the equality checks, but this names the cause.
  check(errors == 0, "the round trips emitted no diagnostics");

  //--- the verifiers reject what the runtime cannot execute ----------------

  const auto emit_error = [&] {
    return mlir::emitError(mlir::UnknownLoc::get(&context));
  };

  // getChecked, not get: a failed verification must be observable here rather
  // than fatal.
  check(!boom::TensorType::getChecked(emit_error, &context, {4096, -1}, "float32"),
        "a negative dimension is rejected");
  check(!boom::TensorType::getChecked(emit_error, &context, {4096, 2048}, "f32"),
        "a dtype outside the frozen sx::DType set is rejected");
  check(bool(boom::TensorType::getChecked(emit_error, &context, {4096, 2048}, "float32")),
        "and float32 is one of them");

  // The block-contiguous dimension is the fastest-moving (last) one, so it is
  // the one that must be a whole number of blocks.
  check(!boom::Q3KType::getChecked(emit_error, &context, {4096, 200}),
        "a q3_K shape whose last dimension is not a multiple of 256 is rejected");
  check(!boom::Q4KType::getChecked(emit_error, &context, {2560, 200}),
        "a q4_K shape whose last dimension is not a multiple of 256 is rejected");
  check(!boom::Q4KType::getChecked(emit_error, &context, {}),
        "a q4_K shape with no dimensions is rejected");
  check(!boom::Q5KType::getChecked(emit_error, &context, {2560, 200}),
        "a q5_K shape whose last dimension is not a multiple of 256 is rejected");
  check(!boom::Q6KType::getChecked(emit_error, &context, {100}),
        "a q6_K shape whose last dimension is not a multiple of 256 is rejected");
  check(!boom::Q8_0Type::getChecked(emit_error, &context, {4096, 30}),
        "a q8_0 shape whose last dimension is not a multiple of 32 is rejected");
  check(!boom::Q8_1Type::getChecked(emit_error, &context, {4096, 30}),
        "a q8_1 shape whose last dimension is not a multiple of 32 is rejected");
  check(!boom::Q5_0Type::getChecked(emit_error, &context, {4096, 30}),
        "a q5_0 shape whose last dimension is not a multiple of 32 is rejected");

  if (failures == 0)
    std::cout << "test_boom_dialect: OK (" << kinds << " OpKinds + boom.graph)\n";
  else
    std::cout << "test_boom_dialect: " << failures << " failure(s)\n";
  return failures == 0 ? 0 : 1;
}
