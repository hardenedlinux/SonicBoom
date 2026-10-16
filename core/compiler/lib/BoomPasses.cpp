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

// The planner's fusion passes, as MLIR passes over Boom IR
// (design/mlir-migration.md §7c, step (c) of the migration).
//
// Each pass here is the port of one pass in planner/rewrites.cpp, in that
// pass's place in its phase's order, with the same preconditions. What is
// *not* copied is how it reads the graph: a pass here works through Boom IR —
// ops, SSA values, the tensor table — and not through planner::Graph, which is
// why it can be a pass at all. What is copied exactly is the result: the graph
// this pipeline produces is the graph the Graph rewrites produce, tensor for
// tensor and attribute for attribute, which is the whole acceptance criterion
// (BoomConversion.h) and what tests/data/plans/*.plan pin.
//
// The split between the two implementations of a phase's pipeline is at the
// boundary in addPortedPasses()/runUnportedPasses() below: everything before it
// runs as Boom IR passes, everything after it still runs as the Graph rewrites
// they were. A pass moves across with its port, in its own commit, and the
// second list empties as the migration completes.

#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomOps.h>
#include <sonicboom/compiler/BoomTypes.h>
#include <sonicboom/planner/ir_passes.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/planner/rewrites.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sonicboom::compiler {
namespace {

//--- the IR view a pass works through ---------------------------------------

// A graph, as a pass sees it. The graph *is* one op (BoomVerify.cpp:22), and
// what a pass needs from it is the two directions of one relation: a tensor id
// — what the graph names and what `result_ids` records — to its SSA value, and
// back. The value is a block argument when the tensor is a free variable and a
// node result otherwise, which is the only difference the relation carries.
//
// The tensor table is the other half, and it is ordered: the order is meaning
// (boomToGraph reads it in order, and so does the planner's tensor list), so a
// pass adds at the end and never sorts. Both the table and the value relation
// are edited only here, so a pass cannot restate the encoding and get it
// subtly wrong — the rules are stated once, in BoomConversion.cpp.
class GraphEditor {
public:
  explicit GraphEditor(boom::GraphOp graph) : graph_(graph), ctx_(graph.getContext()) {
    for (mlir::Attribute entry : graph.getTensors()) {
      auto desc = mlir::cast<boom::TensorDescAttr>(entry);
      ids_.push_back(desc.getId());
      descs_.insert({desc.getId(), desc});
      maxTensorId_ = std::max(maxTensorId_, desc.getId());
    }

    mlir::Block &body = graph_.getBody().front();
    llvm::DenseSet<int64_t> produced;
    for (mlir::Operation &node : body) {
      maxNodeId_ = std::max(maxNodeId_, nodeId(node));
      for (auto [index, id] : llvm::enumerate(resultIds(node))) {
        produced.insert(id);
        note(id, node.getResult(index));
      }
    }

    // The block arguments are the free variables: every tensor the table holds
    // that no node produces, in ascending id order (BoomVerify.cpp:276-282).
    // graphToBoom built them that way, and a pass that keeps the table and the
    // nodes consistent keeps them that way.
    llvm::SmallVector<int64_t> freeIds;
    for (int64_t id : ids_)
      if (!produced.count(id))
        freeIds.push_back(id);
    llvm::sort(freeIds);
    for (auto [index, id] : llvm::enumerate(freeIds))
      note(id, body.getArgument(index));
  }

  boom::GraphOp graph() const { return graph_; }
  mlir::MLIRContext *ctx() const { return ctx_; }
  // Not const: the generated region accessors are not (a region is editable
  // through its op), and a pass's whole job is to edit this one.
  mlir::Block &body() { return graph_.getBody().front(); }

  // The SSA value of tensor `id`, and the id of a value.
  mlir::Value value(int64_t id) const { return valueOf_.lookup(id); }
  int64_t idOf(mlir::Value v) const { return idOfValue_.lookup(v); }
  bool hasTensor(int64_t id) const { return descs_.count(id) != 0; }

  const boom::TensorDescAttr desc(int64_t id) const { return descs_.lookup(id); }

  // Ids above every id in use, handed out one per call: the Graph passes size
  // their new ids from max(id)+1 for the same reason (rewrites.cpp:93-94).
  int64_t nextTensorId() { return ++maxTensorId_; }
  int64_t nextNodeId() { return ++maxNodeId_; }

  // Append a tensor to the table (the end, so the order the planner reads is
  // the order the Graph passes' g.tensors.push_back produced).
  void appendTensor(boom::TensorDescAttr desc) {
    ids_.push_back(desc.getId());
    descs_.insert({desc.getId(), desc});
    maxTensorId_ = std::max(maxTensorId_, desc.getId());
  }

  // Drop a tensor from the table. Only for a tensor nothing references any more:
  // a table entry with no producer that is still read would turn into a block
  // argument, and the graph's arity would change under it (the verifier catches
  // exactly that, per pass — rewrites.cpp drops the same tensors at the same
  // points).
  void dropTensor(int64_t id) {
    descs_.erase(id);
    ids_.erase(std::remove(ids_.begin(), ids_.end(), id), ids_.end());
    valueOf_.erase(id);
  }

  // `oldNode`'s results now come from `newNode`: the id → value direction has to
  // follow the op before the old one is erased, or a later node reading the
  // tensor would resolve to a value that no longer exists.
  void noteReplaced(mlir::Operation *oldNode, mlir::Operation *newNode) {
    for (auto [index, id] : llvm::enumerate(resultIds(*oldNode)))
      note(id, newNode->getResult(index));
  }

  // Write the table back. Until this is called the op still holds the one it
  // came with, so a pass that fails part-way leaves the IR alone.
  void commitTable() {
    llvm::SmallVector<mlir::Attribute> table;
    table.reserve(ids_.size());
    for (int64_t id : ids_)
      table.push_back(descs_.lookup(id));
    graph_.setTensorsAttr(mlir::ArrayAttr::get(ctx_, table));
  }

  // The generated attribute accessors are not const, so neither are these.
  static std::optional<int64_t> nodeIdOpt(mlir::Operation &node) {
    if (auto id = node.getAttrOfType<mlir::IntegerAttr>("node_id"))
      return id.getInt();
    return std::nullopt;
  }
  static int64_t nodeId(mlir::Operation &node) {
    return nodeIdOpt(node).value_or(0);
  }
  static llvm::ArrayRef<int64_t> resultIds(mlir::Operation &node) {
    return node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids").asArrayRef();
  }

private:
  void note(int64_t id, mlir::Value v) {
    valueOf_[id] = v;
    idOfValue_[v] = id;
  }

  boom::GraphOp graph_;
  mlir::MLIRContext *ctx_;
  llvm::SmallVector<int64_t> ids_;                       // table order
  llvm::DenseMap<int64_t, boom::TensorDescAttr> descs_;  // by id
  llvm::DenseMap<int64_t, mlir::Value> valueOf_;
  llvm::DenseMap<mlir::Value, int64_t> idOfValue_;
  int64_t maxTensorId_ = 0;
  int64_t maxNodeId_ = 0;
};

//--- building a node op -----------------------------------------------------

// Build one Boom node op the way the conversion builds one
// (BoomConversion.cpp:365-393): the identity attributes, the authoritative
// attribute record, the optional backend, and the typed attributes ODS declares
// as a view of that record — which the verifier requires to be its int/float
// subset (BoomVerify.cpp:362-366). A pass creates nodes through here for the
// same reason the conversion does: so that what it hands over is an op the
// verifier accepts, and so that the rule has one implementation.
mlir::Operation *buildNode(mlir::OpBuilder &builder, mlir::Location loc,
                           llvm::StringRef opName, int64_t nodeId,
                           llvm::StringRef nodeName, mlir::ArrayAttr nodeAttrs,
                           llvm::ArrayRef<int64_t> resultIds,
                           mlir::ValueRange operands, mlir::TypeRange results,
                           mlir::StringAttr backend) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.emplace_back("node_name", builder.getStringAttr(nodeName));
  attrs.emplace_back("node_id", builder.getI64IntegerAttr(nodeId));
  attrs.emplace_back("result_ids", builder.getDenseI64ArrayAttr(resultIds));
  attrs.emplace_back("node_attrs", nodeAttrs);
  if (backend)
    attrs.emplace_back("backend", backend);

  for (llvm::StringRef name : boom::boomTypedAttrNames()) {
    for (mlir::Attribute entry : nodeAttrs) {
      auto attr = mlir::cast<boom::NodeAttr>(entry);
      if (attr.getName() != name)
        continue;
      // The graph was verified on the way in, so a name in this vocabulary
      // already carries an integer or a float (BoomVerify.cpp:340-357).
      attrs.emplace_back(name, attr.getValue());
      break;
    }
  }

  mlir::OperationState state(loc, ("boom." + opName).str());
  state.addOperands(operands);
  state.addTypes(results);
  state.propertiesAttr = mlir::DictionaryAttr::get(builder.getContext(), attrs);
  return builder.create(state);
}

// The integer a node's attribute record carries for `name`, or nullopt. This is
// the IR-side counterpart of attr_i() (rewrites.cpp:49): the passes read the
// same attributes the Graph passes read, out of the same record.
std::optional<int64_t> nodeAttrInt(mlir::ArrayAttr nodeAttrs, llvm::StringRef name) {
  for (mlir::Attribute entry : nodeAttrs) {
    auto attr = mlir::cast<boom::NodeAttr>(entry);
    if (attr.getName() != name)
      continue;
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
      return value.getInt();
    return std::nullopt;
  }
  return std::nullopt;
}

//--- the passes -------------------------------------------------------------

// quantize_once_pass (rewrites.cpp:79), as a Boom IR pass.
//
// The rewrite, unchanged: a Quantize node is inserted immediately before the
// first QuantizedMatmul consumer of an activation, that matmul becomes the
// q8_1-consuming QuantizedMatmulQ8, and every later matmul over the same
// activation shares the one Quantize node.
class QuantizeOncePass
    : public mlir::PassWrapper<QuantizeOncePass, mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeOncePass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    // The activations already quantized, by tensor id: two Quantize nodes are
    // identical iff their input tensor is — the quantization is a pure function
    // of the activation — so one q8 tensor per distinct input is enough
    // (rewrites.cpp:87-91).
    llvm::DenseMap<int64_t, mlir::Value> q8Of;

    // The nodes as they came in. The walk rewrites them, and a rewrite must not
    // be matched again: the replacement is a quantized_matmul_q8, which is not
    // what this pass matches, but fixing the list up front says so.
    llvm::SmallVector<mlir::Operation *> nodes;
    for (mlir::Operation &node : view.body())
      nodes.push_back(&node);

    for (mlir::Operation *node : nodes) {
      if (node->getName().getStringRef() != "boom.quantized_matmul" ||
          node->getNumOperands() != 1)
        continue;

      const int64_t x = view.idOf(node->getOperand(0));
      mlir::Value q8;
      auto seen = q8Of.find(x);
      if (seen == q8Of.end()) {
        // First use of this activation: the shared Quantize node. Its output
        // mirrors the input's f32 shape and size — a superset of the q8_1 bytes,
        // so the executor and the planner treat it as an ordinary f32
        // intermediate; only the CUDA kernels read the buffer as q8_1
        // (rewrites.cpp:105-120).
        const boom::TensorDescAttr xt = view.desc(x);
        const int64_t qid = view.nextTensorId();
        view.appendTensor(boom::TensorDescAttr::get(
            view.ctx(), qid, xt.getShape(), xt.getDtype(), xt.getByteSize(),
            /*is_input=*/false, /*is_output=*/false, /*is_constant=*/false,
            /*is_external=*/false, ("q8_" + xt.getName()).str()));

        // The value type of the buffer the kernels read as q8_1: the Q8_1Type
        // of its (f32, block-multiple) shape, which is also what the graph
        // verifier will recompute for it once a matmul consumes it.
        auto q8Type = boom::Q8_1Type::getChecked(
            mlir::detail::getDefaultDiagnosticEmitFn(loc), view.ctx(),
            xt.getShape());
        if (!q8Type) {
          signalPassFailure();
          return;
        }

        builder.setInsertionPoint(node);
        mlir::Operation *quantize =
            buildNode(builder, loc, planner::op_kind_name(planner::OpKind::Quantize),
                      view.nextNodeId(),
                      planner::op_kind_name(planner::OpKind::Quantize),
                      builder.getArrayAttr({}), {qid}, {node->getOperand(0)},
                      {q8Type},
                      builder.getStringAttr(
                          planner::backend_tag_name(planner::BackendTag::Sonic)));
        q8 = quantize->getResult(0);
        q8Of.insert({x, q8});
      } else {
        q8 = seen->second;
      }

      // The matmul now consumes the q8_1 activation: the same node — the same
      // id, the same attributes, the same results — under a different operator
      // (rewrites.cpp:134-137). It takes the old node's place in the body, which
      // is after the Quantize node just inserted before it.
      builder.setInsertionPoint(node);
      mlir::Operation *q8node = buildNode(
          builder, loc,
          planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
          GraphEditor::nodeId(*node),
          planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
          node->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
          GraphEditor::resultIds(*node), {q8}, node->getResultTypes(),
          node->getAttrOfType<mlir::StringAttr>("backend"));
      node->replaceAllUsesWith(q8node);
      view.noteReplaced(node, q8node);
      node->erase();
    }

    view.commitTable();
  }
};

//--- the pipeline -----------------------------------------------------------

// The phase's passes whose port has landed, in the phase's execution order
// (design/mlir-migration.md §2). A pass arrives here from runUnportedPasses()
// below in its own commit; the boundary between the two is the migration's only
// state, and the two lists together are always the whole phase.
void addPortedPasses(mlir::PassManager &pm, planner::IrPassPhase phase) {
  switch (phase) {
  case planner::IrPassPhase::Decode:
    pm.addPass(std::make_unique<QuantizeOncePass>());
    return;
  case planner::IrPassPhase::Prefill:
    return;
  }
}

// The same pipeline's other half: the passes still implemented as the Graph
// rewrites they were. They run on the graph the IR passes produced, so the
// boundary is a graph either side can hold, and their order here is the order
// the emitter used to call them in.
void runUnportedPasses(planner::Graph &g, planner::IrPassPhase phase) {
  switch (phase) {
  case planner::IrPassPhase::Decode:
    planner::fuse_rms_norm_quantize_pass(g);
    planner::fuse_gelu_mul_quantize_pass(g);
    planner::fuse_silu_mul_quantize_pass(g);
    planner::fuse_bias_into_gemv_pass(g);
    planner::fuse_residual_add_into_gemv_pass(g);
    return;
  case planner::IrPassPhase::Prefill:
    planner::fuse_rms_norm_mmq_pass(g);
    planner::fuse_gelu_mul_mmq_pass(g);
    planner::fuse_silu_mul_mmq_pass(g);
    return;
  }
}

const char *phaseName(planner::IrPassPhase phase) {
  switch (phase) {
  case planner::IrPassPhase::Decode:
    return "decode";
  case planner::IrPassPhase::Prefill:
    return "prefill";
  }
  return "?";
}

} // namespace

// The pipeline itself: graph -> Boom IR -> the phase's passes -> graph -> the
// passes still implemented as Graph rewrites.
void runBoomIrPipeline(planner::Graph &graph, planner::IrPassPhase phase) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<boom::BoomDialect>();

  auto module = graphToBoom(graph, &context);
  if (!module) {
    std::fprintf(stderr,
                 "ir_pass_pipeline(%s): graph -> Boom IR failed (diagnostic "
                 "above); the graph is left unfused\n",
                 phaseName(phase));
    return;
  }

  // The unit of meaning in this IR is the graph, not the module that holds it
  // (BoomVerify.cpp:21-24), and that is where the passes run: the graph op is
  // IsolatedFromAbove, which is what makes it a pass anchor at all.
  boom::GraphOp graphOp = *module->getOps<boom::GraphOp>().begin();

  // The pass manager verifies after every pass, so a pass that leaves the IR in
  // a state the dialect rejects fails here, naming that pass, rather than three
  // passes later.
  mlir::PassManager pm(&context);
  addPortedPasses(pm, phase);
  if (mlir::failed(pm.run(graphOp))) {
    std::fprintf(stderr,
                 "ir_pass_pipeline(%s): a Boom IR pass failed (diagnostic "
                 "above); the graph is left unfused\n",
                 phaseName(phase));
    return;
  }

  auto back = boomToGraph(*module);
  if (!back) {
    std::fprintf(stderr,
                 "ir_pass_pipeline(%s): Boom IR -> graph failed (diagnostic "
                 "above); the graph is left unfused\n",
                 phaseName(phase));
    return;
  }

  graph = std::move(*back);
  runUnportedPasses(graph, phase);
}

} // namespace sonicboom::compiler

//--- the switch's entry points ----------------------------------------------
//
// What the runtime names (planner/ir_passes.h). They live here, with the
// pipeline they run, rather than in the planner: there is nothing on the
// planner side of them but the choice of which phase to run.

namespace sonicboom::planner {

bool ir_passes_available() { return true; }

void ir_pass_pipeline(Graph &graph, IrPassPhase phase) {
  compiler::runBoomIrPipeline(graph, phase);
}

} // namespace sonicboom::planner
