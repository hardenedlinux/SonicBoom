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
// Every pass here is the port of one pass in the planner's deleted Graph
// rewrites (core/src/planner/rewrites.cpp), in that pass's place in its phase's
// order, with the same preconditions; the `rewrites.cpp:NNN` references below
// are to that file and are kept as the port's provenance. What was never copied
// is how a pass reads the graph: a pass here works through Boom IR — ops, SSA
// values, the tensor table — and not through planner::Graph, which is why it can
// be a pass at all. What was copied exactly is the result: the graph this
// pipeline produces is the graph the Graph rewrites produced, tensor for tensor
// and attribute for attribute, which was the whole acceptance criterion for the
// port (BoomConversion.h) and what tests/data/plans/*.plan pin. The rewrites and
// the switch between the two paths were removed once the port was complete; this
// pipeline is now the only rewrite path.

#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomOps.h>
#include <sonicboom/compiler/BoomPasses.h>
#include <sonicboom/compiler/BoomTypes.h>
#include <sonicboom/planner/ir_passes.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
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

// The IR name of an op kind's node. `buildNode` adds the dialect prefix (an op
// is `boom.quantize`, not `quantize`), so a pass that matches a node by name
// spells it through here rather than by hand — the Graph passes match on the
// op kind, and the one thing a name match can get wrong is its spelling.
std::string boomOpName(planner::OpKind kind) {
  std::string name = "boom.";
  name += planner::op_kind_name(kind);
  return name;
}

// block_q8_1_mmq size in bytes (144: a 16-byte scale union + 128 int8 quants)
// and the mmq matmul's J_max (= I == 128) tail over-read guard — the two
// constants mmq_q8_fits is stated over (rewrites.cpp:58-63).
constexpr uint64_t kMmqBlockBytes = 144;
constexpr uint64_t kMmqTailGuardBlocks = 128;

// Whether the f32 buffer a norm output occupies covers the mmq activation
// buffer for a [n, dim] position-major input: (dim/128, rounded up to an even
// plane count)*n blocks of data + the J_max tail guard, 144 bytes each. The
// even rounding is the zero pad plane the kernel's last 256-value K-iteration
// reads when dim is not a multiple of 256 (mmq::mmq_q8_planes; a no-op for
// K-quant weights). dim must be a multiple of 128 (the kernels enforce it).
// Same arithmetic as rewrites.cpp:71, over the byte size the tensor table
// records.
bool mmqQ8Fits(int64_t dim, int64_t n, uint64_t sizeBytes) {
  const uint64_t planes = uint64_t(dim) / 128 + ((uint64_t(dim) / 128) & 1);
  const uint64_t blocks = planes * uint64_t(n) + kMmqTailGuardBlocks;
  return sizeBytes >= blocks * kMmqBlockBytes;
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
      if (node->getName().getStringRef() !=
              boomOpName(planner::OpKind::QuantizedMatmul) ||
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

// fuse_rms_norm_quantize_pass (rewrites.cpp:143), as a Boom IR pass.
//
// The rewrite, unchanged: an RmsNorm whose normalized output has exactly one
// reader, a Quantize node, becomes one rms_norm_quantize act — the Quantize
// folds into its producer, and the f32 activation the pair carried between them
// stops being a tensor of the graph. The fused node keeps the RmsNorm's
// identity (its id, its attributes, its operands, its backend) and takes the
// Quantize's output, so the matmuls that read the q8_1 activation read on.
class RmsNormQuantizePass
    : public mlir::PassWrapper<RmsNormQuantizePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RmsNormQuantizePass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    // The producer of each tensor, and how many operands read it: the two
    // relations the match is stated in (rewrites.cpp:146-157). Every tensor in
    // the graph has at most one producer; a graph input has none.
    llvm::DenseMap<int64_t, mlir::Operation *> producer;
    llvm::DenseMap<int64_t, unsigned> consumers;
    for (mlir::Operation &node : view.body()) {
      for (int64_t id : GraphEditor::resultIds(node))
        producer.insert({id, &node});
      for (mlir::Value operand : node.getOperands())
        ++consumers[view.idOf(operand)];
    }

    // The (RmsNorm, Quantize) pairs. A match needs the reader count on the
    // RmsNorm's output to be exactly one: the fusion removes that f32 activation
    // from the graph, so a second reader would need it to still exist
    // (rewrites.cpp:159-176). Two Quantize nodes on one output cannot both match
    // for the same reason.
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> pairs;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::Quantize) ||
          node.getNumOperands() != 1)
        continue;
      const int64_t x = view.idOf(node.getOperand(0));
      auto producedBy = producer.find(x);
      if (producedBy == producer.end())
        continue;
      if (producedBy->second->getName().getStringRef() !=
          boomOpName(planner::OpKind::RmsNorm))
        continue;
      if (consumers[x] != 1)
        continue;
      pairs.emplace_back(producedBy->second, &node);
    }

    if (pairs.empty())
      return;

    for (auto [rmsNorm, quantize] : pairs) {
      // The fused node, built where the RmsNorm was: the same node id, the same
      // attribute record, the same operands and backend, and the Quantize's
      // result — the q8_1 activation, whose type and ids carry over
      // (rewrites.cpp:198-205).
      builder.setInsertionPoint(rmsNorm);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::RmsNormQuantize),
          GraphEditor::nodeId(*rmsNorm),
          planner::op_kind_name(planner::OpKind::RmsNormQuantize),
          rmsNorm->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
          GraphEditor::resultIds(*quantize), rmsNorm->getOperands(),
          quantize->getResultTypes(),
          rmsNorm->getAttrOfType<mlir::StringAttr>("backend"));

      // The activation's readers — the matmuls — now read the fused node, and
      // the Quantize node goes away, folded into its producer. The id → value
      // direction follows before the old node is erased.
      quantize->replaceAllUsesWith(fused);
      view.noteReplaced(quantize, fused);
      quantize->erase();

      // The f32 activation the pair carried is no longer a tensor of the graph:
      // nothing reads it any more (its reader count was one, and that reader was
      // the Quantize node just erased), and a table entry with no producer would
      // turn it into a graph input (rewrites.cpp:212-217).
      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*rmsNorm));
      for (int64_t id : dead)
        view.dropTensor(id);
      rmsNorm->erase();
    }

    view.commitTable();
  }
};

// fuse_act_mul_quantize_pass (rewrites.cpp:224), as a Boom IR pass: the shared
// body of the gated-FFN activation×up → q8_1 fusions. The two activation
// flavours (GeluFp16 for Gemma, Silu for SwiGLU models) differ only in which
// activation op is matched and which fused op is emitted; the rewrite is
// otherwise identical, so it is written once and parameterized — as it is on the
// Graph side.
//
// The rewrite, unchanged: where a Quantize reads the product of a Mul whose
// first operand is a single-reader activation node, the activation node becomes
// one fused act_mul_quantize(gate, up) taking over the Quantize's output; the
// activation's output and the act×up product stop being tensors of the graph.
void fuseActMulQuantize(boom::GraphOp graph, planner::OpKind act,
                        planner::OpKind fused) {
  GraphEditor view(graph);
  mlir::OpBuilder builder(view.ctx());
  const mlir::Location loc = graph.getLoc();

  llvm::DenseMap<int64_t, mlir::Operation *> producer;
  llvm::DenseMap<int64_t, unsigned> consumers;
  for (mlir::Operation &node : view.body()) {
    for (int64_t id : GraphEditor::resultIds(node))
      producer.insert({id, &node});
    for (mlir::Value operand : node.getOperands())
      ++consumers[view.idOf(operand)];
  }

  // The op kind's node name and its IR name differ by the dialect prefix, which
  // is exactly the difference a name match has to get right: match on the
  // latter, hand the former to buildNode (which adds the prefix itself).
  const std::string actNode = boomOpName(act);
  const llvm::StringRef fusedName = planner::op_kind_name(fused);

  // The matches, keyed by the activation op: both reader counts have to be one,
  // because the fusion removes both intermediate tensors (rewrites.cpp:235-269).
  llvm::SmallVector<mlir::Operation *> acts;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> mulOfAct;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> quantizeOfAct;
  for (mlir::Operation &node : view.body()) {
    if (node.getName().getStringRef() != boomOpName(planner::OpKind::Quantize) ||
        node.getNumOperands() != 1)
      continue;
    const int64_t fh = view.idOf(node.getOperand(0));
    auto producedBy = producer.find(fh);
    if (producedBy == producer.end())
      continue;
    mlir::Operation *mul = producedBy->second;
    if (mul->getName().getStringRef() != boomOpName(planner::OpKind::Mul) ||
        mul->getNumOperands() != 2)
      continue;
    if (consumers[fh] != 1)
      continue; // the act*up product has another reader
    const int64_t actT = view.idOf(mul->getOperand(0));
    auto actProducer = producer.find(actT);
    if (actProducer == producer.end())
      continue;
    mlir::Operation *a = actProducer->second;
    if (a->getName().getStringRef() != actNode || a->getNumOperands() != 1)
      continue;
    if (consumers[actT] != 1)
      continue; // the activation output has another reader
    acts.push_back(a);
    mulOfAct.insert({a, mul});
    quantizeOfAct.insert({a, &node});
  }

  if (acts.empty())
    return;

  for (mlir::Operation *a : acts) {
    mlir::Operation *mul = mulOfAct.lookup(a);
    mlir::Operation *quantize = quantizeOfAct.lookup(a);

    // The fused node, built where the activation was: the activation's id,
    // attribute record and backend, the gate plus the Mul's up operand as
    // inputs, and the Quantize's output (rewrites.cpp:296-301).
    builder.setInsertionPoint(a);
    mlir::Operation *node = buildNode(
        builder, loc, fusedName, GraphEditor::nodeId(*a), fusedName,
        a->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
        GraphEditor::resultIds(*quantize),
        {a->getOperand(0), mul->getOperand(1)}, quantize->getResultTypes(),
        a->getAttrOfType<mlir::StringAttr>("backend"));

    // The tensors folded away: the activation's output and the act*up product.
    // Their ids are read before their ops go.
    llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*a));
    for (int64_t id : GraphEditor::resultIds(*mul))
      dead.push_back(id);

    // The product's only reader, the Quantize, is gone; then the Mul that made
    // it, then the activation, whose output the Mul was the last reader of.
    quantize->replaceAllUsesWith(node);
    view.noteReplaced(quantize, node);
    quantize->erase();
    mul->erase();
    for (int64_t id : dead)
      view.dropTensor(id);
    a->erase();
  }

  view.commitTable();
}

// fuse_gelu_mul_quantize_pass (rewrites.cpp:315), as a Boom IR pass.
class GeluMulQuantizePass
    : public mlir::PassWrapper<GeluMulQuantizePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeluMulQuantizePass)

  void runOnOperation() override {
    fuseActMulQuantize(getOperation(), planner::OpKind::GeluFp16,
                       planner::OpKind::GeluMulQuantize);
  }
};

// fuse_silu_mul_quantize_pass (rewrites.cpp:319), as a Boom IR pass: the same
// rewrite over the SwiGLU activation.
class SiluMulQuantizePass
    : public mlir::PassWrapper<SiluMulQuantizePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SiluMulQuantizePass)

  void runOnOperation() override {
    fuseActMulQuantize(getOperation(), planner::OpKind::Silu,
                       planner::OpKind::SiluMulQuantize);
  }
};

// fuse_bias_into_gemv_pass (rewrites.cpp:539), as a Boom IR pass.
//
// The rewrite, unchanged: where an AddBias reads the single output of a
// QuantizedMatmulQ8 and is itself the only reader of it, the projection takes
// the AddBias's output tensor over and gains a `bias_slot` attribute naming the
// bias to apply — the bias becomes an epilogue of the gemv instead of a
// separate elementwise pass. The decode form only: a `batch` attribute on
// either node means the batched (prefill) shape, and the mmq path has no
// epilogue for it.
class BiasIntoGemvPass
    : public mlir::PassWrapper<BiasIntoGemvPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BiasIntoGemvPass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    // A projection whose output feeds anything besides the AddBias cannot take
    // the bias over: that reader would read the fused node's output instead
    // (rewrites.cpp:540-551).
    llvm::DenseMap<int64_t, mlir::Operation *> producer;
    llvm::DenseMap<int64_t, unsigned> consumers;
    for (mlir::Operation &node : view.body()) {
      for (int64_t id : GraphEditor::resultIds(node))
        producer.insert({id, &node});
      for (mlir::Value operand : node.getOperands())
        ++consumers[view.idOf(operand)];
    }

    // AddBias op -> (matmul op, bias slot). These AddBias ops are exactly the
    // nodes the rewrite removes.
    llvm::SmallVector<mlir::Operation *> biases;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> matmulOfBias;
    llvm::DenseMap<mlir::Operation *, int64_t> slotOfBias;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::AddBias) ||
          node.getNumOperands() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      auto biasSlot = nodeAttrInt(attrs, "slot");
      if (!biasSlot)
        continue;
      if (nodeAttrInt(attrs, "batch"))
        continue; // batched (prefill) form
      const int64_t proj = view.idOf(node.getOperand(0));
      auto producedBy = producer.find(proj);
      if (producedBy == producer.end())
        continue;
      mlir::Operation *m = producedBy->second;
      if (m->getName().getStringRef() !=
              boomOpName(planner::OpKind::QuantizedMatmulQ8) ||
          m->getNumResults() != 1)
        continue;
      if (nodeAttrInt(m->getAttrOfType<mlir::ArrayAttr>("node_attrs"), "batch"))
        continue; // no epilogue on the mmq path
      if (consumers[proj] != 1)
        continue; // the projection has another reader
      biases.push_back(&node);
      matmulOfBias.insert({&node, m});
      slotOfBias.insert({&node, *biasSlot});
    }

    if (biases.empty())
      return;

    for (mlir::Operation *addBias : biases) {
      mlir::Operation *m = matmulOfBias.lookup(addBias);

      // The matmul's attributes plus the bias it now applies, appended — the
      // Graph pass push_backs onto the same ordered record (rewrites.cpp:601).
      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          m->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      nodeAttrs.push_back(boom::NodeAttr::get(
          view.ctx(), "bias_slot",
          builder.getI64IntegerAttr(slotOfBias.lookup(addBias))));

      // The matmul keeps its identity and place and takes the AddBias's output;
      // its own f32 output tensor — same shape, one buffer either way — stops
      // being a tensor of the graph (rewrites.cpp:580-585).
      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*m));
      builder.setInsertionPoint(m);
      mlir::Operation *node = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
          GraphEditor::nodeId(*m),
          m->getAttrOfType<mlir::StringAttr>("node_name").getValue(),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*addBias),
          m->getOperands(), addBias->getResultTypes(),
          m->getAttrOfType<mlir::StringAttr>("backend"));

      addBias->replaceAllUsesWith(node);
      view.noteReplaced(addBias, node);
      addBias->erase();
      for (int64_t id : dead)
        view.dropTensor(id);
      m->erase();
    }

    view.commitTable();
  }
};

// fuse_residual_add_into_gemv_pass (rewrites.cpp:614), as a Boom IR pass.
//
// The rewrite, unchanged: `Add(proj, residual)` where `proj` is the single-reader
// output of a QuantizedMatmulQ8 becomes a projection of two inputs — the
// residual appended as the last one, so the executor hands it to the backend —
// taking over the Add's output. The residual keeps its buffer (the graph reads
// it one node earlier, at the projection), so liveness only shortens. Either
// operand may be the projection. Decode form only, as above.
class ResidualAddIntoGemvPass
    : public mlir::PassWrapper<ResidualAddIntoGemvPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResidualAddIntoGemvPass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    llvm::DenseMap<int64_t, mlir::Operation *> producer;
    llvm::DenseMap<int64_t, unsigned> consumers;
    for (mlir::Operation &node : view.body()) {
      for (int64_t id : GraphEditor::resultIds(node))
        producer.insert({id, &node});
      for (mlir::Value operand : node.getOperands())
        ++consumers[view.idOf(operand)];
    }

    // Add op -> (matmul op, the operand index the residual sits at). These Add
    // ops are exactly the nodes the rewrite removes.
    llvm::SmallVector<mlir::Operation *> adds;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> matmulOfAdd;
    llvm::DenseMap<mlir::Operation *, unsigned> residualOfAdd;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::Add) ||
          node.getNumOperands() != 2)
        continue;
      if (nodeAttrInt(node.getAttrOfType<mlir::ArrayAttr>("node_attrs"), "batch"))
        continue; // batched (prefill) form
      for (unsigned s = 0; s < 2; ++s) {
        const int64_t proj = view.idOf(node.getOperand(s));
        auto producedBy = producer.find(proj);
        if (producedBy == producer.end())
          continue;
        mlir::Operation *m = producedBy->second;
        if (m->getName().getStringRef() !=
                boomOpName(planner::OpKind::QuantizedMatmulQ8) ||
            m->getNumResults() != 1)
          continue;
        if (nodeAttrInt(m->getAttrOfType<mlir::ArrayAttr>("node_attrs"), "batch"))
          continue; // no epilogue on the mmq path
        if (consumers[proj] != 1)
          continue; // the projection has another reader
        adds.push_back(&node);
        matmulOfAdd.insert({&node, m});
        residualOfAdd.insert({&node, 1 - s});
        break;
      }
    }

    if (adds.empty())
      return;

    for (mlir::Operation *add : adds) {
      mlir::Operation *m = matmulOfAdd.lookup(add);
      const unsigned residual = residualOfAdd.lookup(add);

      // The residual becomes the last input, so the executor hands it to the
      // backend; the attribute names its index for backends that want it
      // explicitly (rewrites.cpp:682-687).
      llvm::SmallVector<mlir::Value> operands;
      for (mlir::Value operand : m->getOperands())
        operands.push_back(operand);
      operands.push_back(add->getOperand(residual));

      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          m->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      nodeAttrs.push_back(boom::NodeAttr::get(view.ctx(), "add_input",
                                              builder.getI64IntegerAttr(
                                                  m->getNumOperands())));

      // The matmul keeps its identity and place and takes the Add's output; its
      // own f32 output tensor becomes dead (rewrites.cpp:660-665).
      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*m));
      builder.setInsertionPoint(m);
      mlir::Operation *node = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
          GraphEditor::nodeId(*m),
          m->getAttrOfType<mlir::StringAttr>("node_name").getValue(),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*add), operands,
          add->getResultTypes(),
          m->getAttrOfType<mlir::StringAttr>("backend"));

      add->replaceAllUsesWith(node);
      view.noteReplaced(add, node);
      add->erase();
      for (int64_t id : dead)
        view.dropTensor(id);
      m->erase();
    }

    view.commitTable();
  }
};

// fuse_rms_norm_mmq_pass (rewrites.cpp:323), as a Boom IR pass — the first pass
// of the prefill pipeline.
//
// The rewrite, unchanged, and the one prefill pass that adds no node and drops
// no tensor: a batched RmsNorm whose every consumer is a batched QuantizedMatmul
// sharing one `ds` becomes a rms_norm_quantize carrying that `ds`, and those
// matmuls become the q8_1-consuming quantized_matmul_q8 — the norm writes its
// q8_1 activation straight into its own f32 output buffer, which is why the
// tensor table (and the buffer it names) is untouched.
//
// A mixed-precision norm — say attn_norm feeding q/k/D4 and v/DS4 — is left
// alone, as is a weight type with no mmq activation layout at all.
//
// TODO(weight-format): `ds` is an emitter-baked integer, the format predicate
// is a private one here as it is in rewrites.cpp, and a `ds < 0` weight type is
// a hard-coded "no mmq layout" answer. When weight formats enter the type
// system these become type properties and the predicate goes away (see the
// weight-format design note, design/weight-quant-format.md).
class RmsNormMmqPass
    : public mlir::PassWrapper<RmsNormMmqPass, mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RmsNormMmqPass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    // The consumers of each tensor, as ops (rewrites.cpp:324-328).
    llvm::DenseMap<int64_t, llvm::SmallVector<mlir::Operation *>> consumers;
    for (mlir::Operation &node : view.body())
      for (mlir::Value operand : node.getOperands())
        consumers[view.idOf(operand)].push_back(&node);

    // The norm ops that fuse, with the shared `ds` and the matmuls they fuse
    // with.
    llvm::SmallVector<mlir::Operation *> norms;
    llvm::DenseMap<mlir::Operation *, int64_t> dsOfNorm;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
        matmulsOfNorm;

    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::RmsNorm) ||
          node.getNumOperands() != 1 || node.getNumResults() == 0)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      auto nb = nodeAttrInt(attrs, "batch");
      auto dim = nodeAttrInt(attrs, "dim");
      if (!nb || !dim || *nb <= 0 || *dim <= 0)
        continue;
      const int64_t out = GraphEditor::resultIds(node)[0];
      auto found = consumers.find(out);
      if (found == consumers.end() || found->second.empty())
        continue;

      int64_t ds = -1;
      llvm::SmallVector<mlir::Operation *> matmuls;
      bool ok = true;
      for (mlir::Operation *consumer : found->second) {
        if (consumer->getName().getStringRef() !=
                boomOpName(planner::OpKind::QuantizedMatmul) ||
            consumer->getNumOperands() != 1) {
          ok = false;
          break;
        }
        auto cAttrs = consumer->getAttrOfType<mlir::ArrayAttr>("node_attrs");
        auto cnb = nodeAttrInt(cAttrs, "batch");
        auto cds = nodeAttrInt(cAttrs, "ds");
        if (!cnb || *cnb != *nb || !cds) {
          ok = false;
          break;
        }
        // ds < 0 marks a weight type with no mmq activation layout (Q6_K, whose
        // mmq weight kernel is disabled); such a matmul never fuses into mmq.
        if (*cds < 0) {
          ok = false;
          break;
        }
        if (ds == -1)
          ds = *cds;
        else if (ds != *cds) {
          ok = false;
          break;
        }
        matmuls.push_back(consumer);
      }
      if (!ok)
        continue;

      // The f32 norm output has to cover the mmq activation buffer, tail guard
      // included (rewrites.cpp:383-385).
      const boom::TensorDescAttr desc = view.desc(out);
      if (!mmqQ8Fits(*dim, *nb, desc.getByteSize()))
        continue;

      norms.push_back(&node);
      dsOfNorm.insert({&node, ds});
      matmulsOfNorm.insert({&node, std::move(matmuls)});
    }

    if (norms.empty())
      return;

    // The norms first, so the matmuls below read the retyped activation.
    for (mlir::Operation *norm : norms) {
      const auto outType =
          mlir::cast<boom::TensorType>(norm->getResult(0).getType());

      // The `ds` the mmq activation layout is selected by, appended to the
      // attribute record (rewrites.cpp:407).
      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          norm->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      nodeAttrs.push_back(boom::NodeAttr::get(
          view.ctx(), "ds",
          builder.getI64IntegerAttr(dsOfNorm.lookup(norm))));

      // The norm's output is the q8_1 activation now: the same buffer, the same
      // shape, read as q8_1 — so the value is retyped with it.
      auto q8Type = boom::Q8_1Type::getChecked(
          mlir::detail::getDefaultDiagnosticEmitFn(loc), view.ctx(),
          outType.getShape());
      if (!q8Type) {
        signalPassFailure();
        return;
      }

      builder.setInsertionPoint(norm);
      mlir::Operation *fused = buildNode(
          builder, loc,
          planner::op_kind_name(planner::OpKind::RmsNormQuantize),
          GraphEditor::nodeId(*norm),
          planner::op_kind_name(planner::OpKind::RmsNormQuantize),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*norm),
          norm->getOperands(), {q8Type},
          norm->getAttrOfType<mlir::StringAttr>("backend"));

      norm->replaceAllUsesWith(fused);
      view.noteReplaced(norm, fused);
      norm->erase();
    }

    // Then the consumers: the same nodes under the operator that reads the
    // activation as q8_1 (rewrites.cpp:408-411).
    for (auto &entry : matmulsOfNorm) {
      for (mlir::Operation *consumer : entry.second) {
        builder.setInsertionPoint(consumer);
        mlir::Operation *q8 = buildNode(
            builder, loc,
            planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
            GraphEditor::nodeId(*consumer),
            planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
            consumer->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
            GraphEditor::resultIds(*consumer), consumer->getOperands(),
            consumer->getResultTypes(),
            consumer->getAttrOfType<mlir::StringAttr>("backend"));
        consumer->replaceAllUsesWith(q8);
        view.noteReplaced(consumer, q8);
        consumer->erase();
      }
    }

    // No tensor changed: no append, no drop, and the table is written back
    // unchanged. The pass changes what the graph *says*, not what it holds.
    view.commitTable();
  }
};

// fuse_act_mul_mmq_pass (rewrites.cpp:421), as a Boom IR pass: the shared body
// of the C.4 prefill activation×up → mmq fusions, parameterized by (activation
// op, fused op) as it is on the Graph side.
//
// The rewrite, unchanged: where a batched QuantizedMatmul (the FFN down
// projection) reads an fh produced by Mul(act, up) with a single-use activation
// node, the activation becomes `fused`(gate, up) — taking over the Mul's f32
// output buffer, which the kernel reads as q8_1 — carrying the batch/dim/ds the
// matmul needs to select its kernel, and the matmul becomes the q8_1-consuming
// quantized_matmul_q8. The activation's own output stops being a tensor of the
// graph; the Mul's output stays, reused.
//
// TODO(weight-format): the `ds < 0` predicate, as in the rms_norm mmq pass.
//
// Returns false when a value cannot be retyped, which is a failure to report
// rather than something to work around: the shared body has no pass to signal
// through, so the wrapper does (an empty shape should not be reachable — it
// would mean the graph cannot be represented as Boom IR at all).
bool fuseActMulMmq(boom::GraphOp graph, planner::OpKind act,
                   planner::OpKind fused) {
  GraphEditor view(graph);
  mlir::OpBuilder builder(view.ctx());
  const mlir::Location loc = graph.getLoc();

  const std::string actNode = boomOpName(act);
  const llvm::StringRef fusedName = planner::op_kind_name(fused);

  llvm::DenseMap<int64_t, mlir::Operation *> producer;
  llvm::DenseMap<int64_t, unsigned> consumers;
  for (mlir::Operation &node : view.body()) {
    for (int64_t id : GraphEditor::resultIds(node))
      producer.insert({id, &node});
    for (mlir::Value operand : node.getOperands())
      ++consumers[view.idOf(operand)];
  }

  // The matches, keyed by the activation op. The matmul is the batched
  // QuantizedMatmul the fusion turns into its q8_1 form.
  llvm::SmallVector<mlir::Operation *> acts;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> matmulOfAct;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> mulOfAct;
  llvm::DenseMap<mlir::Operation *, int64_t> batchOfAct;
  llvm::DenseMap<mlir::Operation *, int64_t> dimOfAct;
  llvm::DenseMap<mlir::Operation *, int64_t> dsOfAct;

  for (mlir::Operation &node : view.body()) {
    if (node.getName().getStringRef() !=
            boomOpName(planner::OpKind::QuantizedMatmul) ||
        node.getNumOperands() != 1)
      continue;
    auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
    auto nb = nodeAttrInt(attrs, "batch");
    auto dim = nodeAttrInt(attrs, "dim");
    auto ds = nodeAttrInt(attrs, "ds");
    if (!nb || !dim || !ds || *nb <= 0 || *dim <= 0)
      continue;
    // ds < 0 marks a weight type with no mmq activation layout (Q6_K, whose mmq
    // weight kernel is disabled); its activation must not be quantized, so the
    // matmul keeps its internal (dequant + cuBLAS) path and never fuses.
    if (*ds < 0)
      continue;
    const int64_t fh = view.idOf(node.getOperand(0));
    auto mulOf = producer.find(fh);
    if (mulOf == producer.end())
      continue;
    mlir::Operation *mul = mulOf->second;
    if (mul->getName().getStringRef() != boomOpName(planner::OpKind::Mul) ||
        mul->getNumOperands() != 2)
      continue;
    if (consumers[fh] != 1)
      continue; // the act*up product has another reader
    const int64_t actT = view.idOf(mul->getOperand(0));
    auto actProducer = producer.find(actT);
    if (actProducer == producer.end())
      continue;
    mlir::Operation *a = actProducer->second;
    if (a->getName().getStringRef() != actNode || a->getNumOperands() != 1)
      continue;
    if (consumers[actT] != 1)
      continue; // the activation output has another reader
    // The Mul's f32 output is the buffer the fused node reuses as the q8_1
    // activation, so it is that buffer the fit is checked against
    // (rewrites.cpp:475-477).
    const boom::TensorDescAttr ft = view.desc(GraphEditor::resultIds(*mul)[0]);
    if (!mmqQ8Fits(*dim, *nb, ft.getByteSize()))
      continue;

    acts.push_back(a);
    matmulOfAct.insert({a, &node});
    mulOfAct.insert({a, mul});
    batchOfAct.insert({a, *nb});
    dimOfAct.insert({a, *dim});
    dsOfAct.insert({a, *ds});
  }

  if (acts.empty())
    return true;

  llvm::SmallVector<mlir::Operation *> matmuls;
  for (mlir::Operation *a : acts) {
    mlir::Operation *mul = mulOfAct.lookup(a);

    // The fused node, built where the activation was: its id and backend, the
    // gate plus the Mul's up operand as inputs, and the Mul's output tensor —
    // reused as the q8_1 activation buffer, so its shape is the value's and its
    // table entry (f32, that shape) is what the verifier wants behind a q8_1
    // value. The batch/dim/ds the matmul's kernel selection reads are appended
    // in that order (rewrites.cpp:509-515).
    llvm::SmallVector<mlir::Attribute> nodeAttrs(
        a->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
    nodeAttrs.push_back(boom::NodeAttr::get(
        view.ctx(), "batch", builder.getI64IntegerAttr(batchOfAct.lookup(a))));
    nodeAttrs.push_back(boom::NodeAttr::get(
        view.ctx(), "dim", builder.getI64IntegerAttr(dimOfAct.lookup(a))));
    nodeAttrs.push_back(boom::NodeAttr::get(
        view.ctx(), "ds", builder.getI64IntegerAttr(dsOfAct.lookup(a))));

    const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*a));
    const auto mulType = mlir::cast<boom::TensorType>(mul->getResult(0).getType());
    auto q8Type = boom::Q8_1Type::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(loc), view.ctx(),
        mulType.getShape());
    if (!q8Type)
      return false;

    builder.setInsertionPoint(a);
    mlir::Operation *fusedNode = buildNode(
        builder, loc, fusedName, GraphEditor::nodeId(*a), fusedName,
        builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*mul),
        {a->getOperand(0), mul->getOperand(1)}, {q8Type},
        a->getAttrOfType<mlir::StringAttr>("backend"));

    // The Mul goes away: its readers — the matmul below — read the fused node,
    // and its output tensor is that node's now. Then the activation goes too:
    // the Mul was its only reader.
    mul->replaceAllUsesWith(fusedNode);
    view.noteReplaced(mul, fusedNode);
    mul->erase();
    for (int64_t id : dead)
      view.dropTensor(id);
    a->erase();

    matmuls.push_back(matmulOfAct.lookup(a));
  }

  // The consumers last, so they read the retyped activation: the same nodes
  // under the operator that reads it as q8_1 (rewrites.cpp:516-519).
  for (mlir::Operation *consumer : matmuls) {
    builder.setInsertionPoint(consumer);
    mlir::Operation *q8 = buildNode(
        builder, loc,
        planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
        GraphEditor::nodeId(*consumer),
        planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
        consumer->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
        GraphEditor::resultIds(*consumer), consumer->getOperands(),
        consumer->getResultTypes(),
        consumer->getAttrOfType<mlir::StringAttr>("backend"));
    consumer->replaceAllUsesWith(q8);
    view.noteReplaced(consumer, q8);
    consumer->erase();
  }

  view.commitTable();
  return true;
}

// fuse_gelu_mul_mmq_pass (rewrites.cpp:531), as a Boom IR pass.
class GeluMulMmqPass
    : public mlir::PassWrapper<GeluMulMmqPass, mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeluMulMmqPass)

  void runOnOperation() override {
    if (!fuseActMulMmq(getOperation(), planner::OpKind::GeluFp16,
                       planner::OpKind::GeluMulQuantize))
      signalPassFailure();
  }
};

// fuse_silu_mul_mmq_pass (rewrites.cpp:535), as a Boom IR pass: the same
// rewrite over the SwiGLU activation.
class SiluMulMmqPass
    : public mlir::PassWrapper<SiluMulMmqPass, mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SiluMulMmqPass)

  void runOnOperation() override {
    if (!fuseActMulMmq(getOperation(), planner::OpKind::Silu,
                       planner::OpKind::SiluMulQuantize))
      signalPassFailure();
  }
};

//--- the per-pass disable switch --------------------------------------------

// Ablation for one fusion at a time: the passes a run skips, named on the
// command line (SONICBOOM_IR_PASS_DISABLE, documented on the runtime side beside
// the pipeline entry point in planner/ir_passes.h, and read here, where the
// passes and their names live).
//
// Skipping a pass leaves the graph as the pass before it would, so a later pass
// sees what it is written to see. That is not the same as the later pass being
// unaffected: the pipeline is a sequence over one graph, and QuantizeOncePass
// inserts the Quantize nodes RmsNormQuantizePass and GeluMulQuantizePass fold
// into their producer, so skipping it also removes those two folds (measured —
// see the plan deltas in the run that landed this). A disabled pass is read as
// "this pass and the folds it feeds", which is what the plan diff records.
struct DisabledPasses {
  bool quantize_once = false;
  bool rms_norm_quantize = false;
  bool gelu_mul_quantize = false;
  bool silu_mul_quantize = false;
  bool bias_into_gemv = false;
  bool residual_add_into_gemv = false;
  bool rms_norm_mmq = false;
  bool gelu_mul_mmq = false;
  bool silu_mul_mmq = false;
};

// The names the switch accepts: the pass class names, one table so the parser
// and the error message cannot disagree about them.
struct PassName {
  const char *name;
  bool DisabledPasses::*flag;
};
constexpr PassName kPassNames[] = {
    {"QuantizeOncePass", &DisabledPasses::quantize_once},
    {"RmsNormQuantizePass", &DisabledPasses::rms_norm_quantize},
    {"GeluMulQuantizePass", &DisabledPasses::gelu_mul_quantize},
    {"SiluMulQuantizePass", &DisabledPasses::silu_mul_quantize},
    {"BiasIntoGemvPass", &DisabledPasses::bias_into_gemv},
    {"ResidualAddIntoGemvPass", &DisabledPasses::residual_add_into_gemv},
    {"RmsNormMmqPass", &DisabledPasses::rms_norm_mmq},
    {"GeluMulMmqPass", &DisabledPasses::gelu_mul_mmq},
    {"SiluMulMmqPass", &DisabledPasses::silu_mul_mmq},
};

// Read SONICBOOM_IR_PASS_DISABLE into `out`. Returns false, having named both
// the offending entry and the valid names, when the list is not a list of
// passes — an unknown name is reported rather than ignored so a typo cannot
// pass for an ablation.
bool parseDisabledPasses(DisabledPasses &out) {
  const char *env = std::getenv("SONICBOOM_IR_PASS_DISABLE");
  if (env == nullptr || *env == '\0') return true;
  const std::string_view list(env);
  std::string unknown;
  for (std::size_t pos = 0;;) {
    const std::size_t comma = list.find(',', pos);
    std::string_view name =
        list.substr(pos, comma == std::string_view::npos ? comma : comma - pos);
    while (!name.empty() && (name.front() == ' ' || name.front() == '\t'))
      name.remove_prefix(1);
    while (!name.empty() && (name.back() == ' ' || name.back() == '\t'))
      name.remove_suffix(1);
    if (!name.empty()) {
      bool matched = false;
      for (const PassName &p : kPassNames) {
        if (name == p.name) {
          out.*(p.flag) = true;
          matched = true;
          break;
        }
      }
      if (!matched && unknown.empty()) unknown = std::string(name);
    }
    if (comma == std::string_view::npos) break;
    pos = comma + 1;
  }
  if (unknown.empty()) return true;
  std::fprintf(stderr,
               "SONICBOOM_IR_PASS_DISABLE: unknown pass '%s'; valid names are:",
               unknown.c_str());
  for (const PassName &p : kPassNames) std::fprintf(stderr, " %s", p.name);
  std::fprintf(stderr, "\n");
  return false;
}

//--- the pipeline -----------------------------------------------------------

// A phase's passes, in the phase's execution order (design/mlir-migration.md
// §2): the port of each Graph rewrite in planner/rewrites.cpp, in that pass's
// place in the order and with its preconditions. Both phases run entirely here,
// and the Graph rewrites are gone, so this is the emitter's only rewrite path.
void addPasses(mlir::PassManager &pm, planner::IrPassPhase phase) {
  DisabledPasses off;
  // A rejected list disables nothing: running the phase with no passes would
  // leave the graph unfused, and an ablation run would then report that as the
  // effect of the pass it meant to skip.
  if (!parseDisabledPasses(off)) off = DisabledPasses{};
  switch (phase) {
  case planner::IrPassPhase::Decode:
    if (!off.quantize_once) pm.addPass(std::make_unique<QuantizeOncePass>());
    if (!off.rms_norm_quantize) pm.addPass(std::make_unique<RmsNormQuantizePass>());
    if (!off.gelu_mul_quantize) pm.addPass(std::make_unique<GeluMulQuantizePass>());
    if (!off.silu_mul_quantize) pm.addPass(std::make_unique<SiluMulQuantizePass>());
    if (!off.bias_into_gemv) pm.addPass(std::make_unique<BiasIntoGemvPass>());
    if (!off.residual_add_into_gemv)
      pm.addPass(std::make_unique<ResidualAddIntoGemvPass>());
    return;
  case planner::IrPassPhase::Prefill:
    if (!off.rms_norm_mmq) pm.addPass(std::make_unique<RmsNormMmqPass>());
    if (!off.gelu_mul_mmq) pm.addPass(std::make_unique<GeluMulMmqPass>());
    if (!off.silu_mul_mmq) pm.addPass(std::make_unique<SiluMulMmqPass>());
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

// The phase's passes over the IR a caller built (compiler/BoomPasses.h).
bool runBoomPasses(mlir::ModuleOp module, planner::IrPassPhase phase) {
  // The unit of meaning in this IR is the graph, not the module that holds it
  // (BoomVerify.cpp:21-24), and that is where the passes run: the graph op is
  // IsolatedFromAbove, which is what makes it a pass anchor at all.
  auto graphs = module.getOps<boom::GraphOp>();
  if (graphs.empty()) {
    std::fprintf(stderr, "ir_pass_pipeline(%s): the module holds no graph\n",
                 phaseName(phase));
    return false;
  }
  boom::GraphOp graphOp = *graphs.begin();

  // The pass manager verifies after every pass, so a pass that leaves the IR in
  // a state the dialect rejects fails here, naming that pass, rather than three
  // passes later.
  mlir::PassManager pm(module.getContext());
  addPasses(pm, phase);
  if (mlir::failed(pm.run(graphOp))) {
    std::fprintf(stderr,
                 "ir_pass_pipeline(%s): a Boom IR pass failed (diagnostic "
                 "above); the graph is left unfused\n",
                 phaseName(phase));
    return false;
  }
  return true;
}

// The pipeline itself: graph -> Boom IR -> the phase's passes -> graph.
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

  if (!runBoomPasses(*module, phase))
    return;

  auto back = boomToGraph(*module);
  if (!back) {
    std::fprintf(stderr,
                 "ir_pass_pipeline(%s): Boom IR -> graph failed (diagnostic "
                 "above); the graph is left unfused\n",
                 phaseName(phase));
    return;
  }

  graph = std::move(*back);
}

} // namespace sonicboom::compiler

//--- the runtime's entry point ----------------------------------------------
//
// What the runtime names (planner/ir_passes.h). It lives here, with the
// pipeline it runs, rather than in the planner: there is nothing on the planner
// side of it but the choice of which phase to run.

namespace sonicboom::planner {

void ir_pass_pipeline(Graph &graph, IrPassPhase phase) {
  compiler::runBoomIrPipeline(graph, phase);
}

} // namespace sonicboom::planner
