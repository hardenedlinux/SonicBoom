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

// SonicBoom S-Expr v0.1 → CPU JIT execution.
//
// `Executable::compile` lowers a validated Document into an MLIR module (see
// lowering.cpp), runs the CPU lowering pipeline below, and JITs the result with
// MLIR's ExecutionEngine into a callable native function:
//
//   convert-elementwise-to-linalg → one-shot-bufferize (function boundaries)
//     → convert-bufferization-to-memref → convert-linalg-to-loops
//     → convert-scf-to-cf → expand-strided-metadata → lower-affine
//     → convert-arith-to-llvm → convert-scf-to-cf (residual)
//     → finalize-memref-to-llvm → convert-func-to-llvm (bare-ptr call conv)
//     → convert-cf-to-llvm → reconcile-unrealized-casts
//
// The function is invoked through the bare-pointer calling convention: a
// `func.func @main(tensor<...xf32>, ..., tensor<...xf32>) -> tensor<...xf32>`
// (identity layout) becomes `float *main(float *, ..., float *)`; the returned
// pointer is malloc'd by the JIT runtime and freed by the caller. v0 execution
// scope is one or more float32 inputs and exactly one float32 output.

#include <sonicboom/sx/exec.h>

#include "lowering_internal.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <variant>

namespace sonicboom::sx {
namespace {

using namespace mlir;

ExecError make_error(ExecErrorKind kind, std::string message) {
  return ExecError{kind, std::move(message)};
}

// Resolve a graph output name to its declared TensorType by scanning every SSA
// definition site (inputs, parameters, node outputs) — the same lookup the
// lowering performs for result types.
const TensorType* find_def_type(const Graph& g, const std::string& name) {
  for (const auto& in : g.inputs)
    if (in.name == name)
      return &in.type;
  for (const auto& p : g.parameters)
    if (p.name == name)
      return &p.type;
  for (const auto& n : g.nodes)
    for (const auto& o : n.outputs)
      if (o.name == name)
        return &o.type;
  return nullptr;
}

// Run the full CPU lowering pipeline over `module`. Returns "" on success or a
// non-empty diagnostic string on failure.
std::string run_cpu_pipeline(MLIRContext& ctx, ModuleOp module) {
  PassManager pm(&ctx);

  // The lowering emits tensor-level arith elementwise ops (addf/maximumf/mulf/
  // divf) rather than linalg.generic; bufferization only knows scalar arith
  // inside linalg regions, so map those to linalg.generic first.
  pm.addPass(createConvertElementwiseToLinalgPass());

  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  bufferizeOpts.bufferizeFunctionBoundaries = true;
  // Identity layout on the function signature keeps the memref as a plain
  // `memref<...xf32>` (no dynamic offset/stride), which is what the bare-ptr
  // call convention needs to produce `float*(float*)`.
  bufferizeOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  // Force out-of-place bufferization so the input buffer is never written and
  // the returned tensor is a freshly allocated buffer the caller can free
  // (the default in-place analysis would alias the result to the input).
  bufferizeOpts.copyBeforeWrite = true;
  // Alignment of 1 keeps `memref.alloc`'s allocated pointer identical to its
  // aligned (data) pointer. The bare-ptr return convention exposes the
  // *allocated* pointer, so a larger alignment (the 64-byte default) would
  // return the raw malloc base while the data sits at the aligned offset —
  // producing garbage in the caller.
  bufferizeOpts.bufferAlignment = 1;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createSCFToControlFlowPass());  // residual scf from expansions
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  ConvertFuncToLLVMPassOptions funcOpts;
  funcOpts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(funcOpts));

  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  std::string diag;
  {
    ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic& d) {
      llvm::raw_string_ostream os(diag);
      os << d;
      return success();
    });
    if (failed(pm.run(module)))
      return diag.empty() ? "MLIR CPU lowering pipeline failed" : diag;
  }
  return "";
}

} // namespace

// ---------------------------------------------------------------------------
// Executable
// ---------------------------------------------------------------------------

struct Executable::Impl {
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void* entry;  // raw JIT symbol; the number of input pointers is fixed by the graph
  std::vector<TensorType> inputs;
  std::vector<TensorType> outputs;
  std::vector<std::size_t> input_bytes;
  std::vector<std::size_t> output_bytes;
};

// Invoke the JIT entry with one bare pointer per graph input. The bare-ptr call
// convention turns `func @main(memref<..>, ..., memref<..>) -> memref<..>`
// (identity layout) into `float* main(float*, ..., float*)`, so the entry takes
// one pointer per input and returns the single output pointer. Arity is fixed
// at compile time by the graph, so dispatch on it. Returns nullptr for an
// unsupported (larger) arity.
void* call_entry(void* entry, const std::vector<const void*>& ptrs) {
  switch (ptrs.size()) {
    case 1:
      return reinterpret_cast<void* (*)(const void*)>(entry)(ptrs[0]);
    case 2:
      return reinterpret_cast<void* (*)(const void*, const void*)>(entry)(
          ptrs[0], ptrs[1]);
    case 3:
      return reinterpret_cast<void* (*)(const void*, const void*,
                                       const void*)>(entry)(
          ptrs[0], ptrs[1], ptrs[2]);
    case 4:
      return reinterpret_cast<void* (*)(const void*, const void*, const void*,
                                       const void*)>(entry)(
          ptrs[0], ptrs[1], ptrs[2], ptrs[3]);
    default:
      return nullptr;
  }
}

Executable::Executable() : impl_(std::make_unique<Impl>()) {}
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;
Executable::~Executable() = default;

std::expected<std::unique_ptr<Executable>, ExecError> Executable::compile(
    const Document& doc, const std::unordered_map<std::string, Bytes>& weights) {
  const Graph& g = doc.graph;

  // v0 execution scope: one or more float32 inputs, exactly one float32 output.
  if (g.inputs.empty())
    return std::unexpected(make_error(
        ExecErrorKind::Compile, "execution requires at least one graph input"));
  if (g.outputs.size() != 1)
    return std::unexpected(make_error(
        ExecErrorKind::Compile,
        "execution requires exactly one graph output (got " +
            std::to_string(g.outputs.size()) + ")"));

  // Reject overflowing byte sizes up front, before any MLIR lowering or JIT,
  // so a wrapped size can never reach an allocation, file read, or memcpy.
  std::vector<TensorType> in_types;
  std::vector<std::size_t> in_bytes;
  for (const auto& in : g.inputs) {
    if (in.type.dtype != DType::Float32)
      return std::unexpected(make_error(
          ExecErrorKind::Compile,
          "execution input '" + in.name + "' must be float32 in v0"));
    auto b = tensor_byte_size(in.type);
    if (!b)
      return std::unexpected(make_error(
          ExecErrorKind::Compile, "input tensor byte size overflows"));
    in_types.push_back(in.type);
    in_bytes.push_back(static_cast<std::size_t>(*b));
  }

  const TensorType* out_type = find_def_type(g, g.outputs[0]);
  if (!out_type)
    return std::unexpected(make_error(
        ExecErrorKind::Compile,
        "graph output references undefined value '" + g.outputs[0] + "'"));
  if (out_type->dtype != DType::Float32)
    return std::unexpected(make_error(
        ExecErrorKind::Compile, "execution output must be float32 in v0"));
  auto out_bytes = tensor_byte_size(*out_type);
  if (!out_bytes)
    return std::unexpected(make_error(
        ExecErrorKind::Compile, "output tensor byte size overflows"));

  // Dialect set: the lowering emits func/arith/tensor/linalg ops, and the
  // pipeline introduces bufferization/scf/cf/memref/affine/LLVM ops. The
  // bufferization external models (for arith/tensor/linalg constants and the
  // func.func boundary) are registered explicitly: `getOrLoadDialect` only
  // "promises" those interfaces, and one-shot-bufferize cannot bufferize any
  // op without the concrete model.
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, tensor::TensorDialect,
                  linalg::LinalgDialect, bufferization::BufferizationDialect,
                  scf::SCFDialect, cf::ControlFlowDialect,
                  memref::MemRefDialect, affine::AffineDialect,
                  LLVM::LLVMDialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  MLIRContext ctx(registry);

  // Load every dialect the lowering and pipeline can emit. (The registry only
  // registers dialects and their external models; `getOrLoadDialect` is what
  // actually instantiates them, which is also what applies the bufferization
  // external models registered above.)
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<tensor::TensorDialect>();
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  ctx.getOrLoadDialect<bufferization::BufferizationDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<affine::AffineDialect>();
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();

  // LLVM IR translation for the builtin and LLVM dialects (only these two are
  // present in the lowered module, so the full registerAllToLLVMIRTranslations
  // set — and its additional GPU/NVVM/… link deps — is not needed).
  registerBuiltinDialectTranslation(ctx);
  registerLLVMDialectTranslation(ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&ctx));

  // Lower with :external parameters baked from `weights` (which is exactly the
  // WeightMap the lowering consumes).
  std::string err = lower_into_module(ctx, *module, doc, &weights);
  if (!err.empty())
    return std::unexpected(make_error(ExecErrorKind::Compile, std::move(err)));

  err = run_cpu_pipeline(ctx, *module);
  if (!err.empty())
    return std::unexpected(make_error(ExecErrorKind::Compile, std::move(err)));

  // Native target registration, mirroring mlir-runner. Required before the
  // ExecutionEngine can JIT for the host (x86-64 here).
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto engine = ExecutionEngine::create(*module);
  if (!engine)
    return std::unexpected(make_error(
        ExecErrorKind::Compile,
        "failed to create ExecutionEngine: " +
            llvm::toString(engine.takeError())));

  std::unique_ptr<mlir::ExecutionEngine> eng = std::move(*engine);

  // Bare-ptr ABI: each tensor argument/result is a raw data pointer. The
  // lowering names the entry function after the graph (`g.name`), so look that
  // symbol up (its LLVM symbol name is unchanged by the conversion).
  auto sym = eng->lookup(g.name);
  if (!sym)
    return std::unexpected(make_error(
        ExecErrorKind::Compile,
        "JIT entry symbol '" + g.name + "' not found: " +
            llvm::toString(sym.takeError())));

  auto exe = std::unique_ptr<Executable>(new Executable());
  exe->impl_->engine = std::move(eng);
  exe->impl_->entry = *sym;
  exe->impl_->inputs = std::move(in_types);
  exe->impl_->outputs = {*out_type};
  exe->impl_->input_bytes = std::move(in_bytes);
  exe->impl_->output_bytes = {static_cast<std::size_t>(*out_bytes)};
  return exe;
}

const TensorType& Executable::input_type() const { return impl_->inputs.front(); }
const TensorType& Executable::output_type() const { return impl_->outputs.front(); }

std::expected<void, ExecError> Executable::run(const std::vector<Bytes>& inputs,
                                               std::vector<Bytes>& outputs) const {
  if (inputs.size() != impl_->inputs.size())
    return std::unexpected(make_error(
        ExecErrorKind::Binding,
        "expected " + std::to_string(impl_->inputs.size()) +
            " input buffers, got " + std::to_string(inputs.size())));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].size() != impl_->input_bytes[i])
      return std::unexpected(make_error(
          ExecErrorKind::Binding,
          "input buffer " + std::to_string(i) + " is " +
              std::to_string(inputs[i].size()) + " bytes, expected " +
              std::to_string(impl_->input_bytes[i])));
  if (impl_->output_bytes.front() == 0)
    return std::unexpected(make_error(ExecErrorKind::Binding,
                                      "output tensor has zero elements"));

  if (inputs.size() > 4)
    return std::unexpected(make_error(
        ExecErrorKind::Binding,
        "execution supports at most 4 graph inputs, got " +
            std::to_string(inputs.size())));

  std::vector<const void*> ptrs;
  ptrs.reserve(inputs.size());
  for (const auto& b : inputs)
    ptrs.push_back(b.data());

  void* result = call_entry(impl_->entry, ptrs);
  if (!result)
    return std::unexpected(make_error(
        ExecErrorKind::Runtime, "JIT entry returned a null pointer"));

  outputs.assign(impl_->outputs.size(), Bytes{});
  outputs.front().resize(impl_->output_bytes.front());
  std::memcpy(outputs.front().data(), result, impl_->output_bytes.front());
  std::free(result);  // the bare-ptr ABI returns a malloc'd buffer
  return {};
}

std::expected<void, ExecError> Executable::run(const Bytes& input,
                                               Bytes& output) const {
  if (impl_->inputs.size() != 1)
    return std::unexpected(make_error(
        ExecErrorKind::Binding,
        "single-input run() requires exactly one graph input, got " +
            std::to_string(impl_->inputs.size())));
  std::vector<Bytes> ins{input};
  std::vector<Bytes> outs;
  auto r = run(ins, outs);
  if (!r)
    return std::unexpected(r.error());
  output = std::move(outs.front());
  return {};
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

std::expected<std::unordered_map<std::string, Bytes>, ExecError>
load_external_weights(const Document& doc, const std::string& base_dir) {
  std::unordered_map<std::string, Bytes> out;
  for (const auto& p : doc.graph.parameters) {
    const auto* ext = std::get_if<ExternalData>(&p.data);
    if (!ext)
      continue;

    auto byte_size = tensor_byte_size(p.type);
    if (!byte_size)
      return std::unexpected(make_error(
          ExecErrorKind::Weight,
          "external parameter '" + p.name + "' has an overflowing byte size"));
    std::int64_t expected = *byte_size;
    if (ext->length != expected)
      return std::unexpected(make_error(
          ExecErrorKind::Weight,
          "external parameter '" + p.name + "' declares " +
              std::to_string(ext->length) + " bytes but its shape/dtype need " +
              std::to_string(expected)));

    std::filesystem::path path(ext->file);
    if (path.is_relative())
      path = std::filesystem::path(base_dir) / path;

    std::ifstream f(path, std::ios::binary);
    if (!f)
      return std::unexpected(make_error(
          ExecErrorKind::Weight,
          "cannot open weight sidecar '" + path.string() + "'"));
    f.seekg(static_cast<std::streamoff>(ext->offset), std::ios::beg);
    if (!f)
      return std::unexpected(make_error(
          ExecErrorKind::Weight,
          "cannot seek to offset " + std::to_string(ext->offset) + " in '" +
              path.string() + "'"));

    Bytes bytes(static_cast<std::size_t>(ext->length));
    f.read(reinterpret_cast<char*>(bytes.data()),
           static_cast<std::streamsize>(ext->length));
    if (f.gcount() != static_cast<std::streamsize>(ext->length))
      return std::unexpected(make_error(
          ExecErrorKind::Weight,
          "short read on weight sidecar '" + path.string() + "' for '" +
              p.name + "'"));

    out[p.name] = std::move(bytes);
  }
  return out;
}

} // namespace sonicboom::sx
