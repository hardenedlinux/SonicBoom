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

// The planner's fusion passes, as MLIR passes over Boom IR (docs/boom-ir.md).
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

#include <sonicboom/compiler/BoomOps.h>
#include <sonicboom/compiler/BoomPasses.h>
#include <sonicboom/compiler/BoomTypes.h>
#include <sonicboom/planner/ir_passes.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/runtime_features.h>

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

  // Replace a tensor's declared byte size (and the fixed part of its affine
  // size in the graph's dynamic extent), keeping every other field. A fused op
  // writes a compact quantized activation into a buffer the emitter declared as
  // the f32 tensor it was derived from; once every reader of that buffer takes
  // the quantized view, the declaration can be the quantized size. `sizeBytes`
  // is the size at the extent this pass binds.
  void setTensorSize(int64_t id, uint64_t sizeBytes,
                     uint64_t dynamicBaseBytes = 0) {
    const boom::TensorDescAttr d = descs_.lookup(id);
    descs_[id] = boom::TensorDescAttr::get(
        ctx_, id, d.getShape(), d.getDtype(), sizeBytes, dynamicBaseBytes,
        d.getIsInput(), d.getIsOutput(), d.getIsConstant(), d.getIsExternal(),
        d.getName());
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

  // Point one tensor id at `v`. For a fusion whose results are not the old
  // node's in order (a region that keeps the producer's output as a second
  // result, say), where noteReplaced's index-wise mapping would be wrong.
  void noteValue(int64_t id, mlir::Value v) { note(id, v); }

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

// The float a node's attribute record carries for `name`, or nullopt. Attribute
// records come from the planner as doubles, which the Boom dialect carries as
// f64 attributes.
std::optional<double> nodeAttrFloat(mlir::ArrayAttr nodeAttrs,
                                    llvm::StringRef name) {
  for (mlir::Attribute entry : nodeAttrs) {
    auto attr = mlir::cast<boom::NodeAttr>(entry);
    if (attr.getName() != name)
      continue;
    if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attr.getValue()))
      return value.getValueAsDouble();
    return std::nullopt;
  }
  return std::nullopt;
}

// The string a node's attribute record carries for `name`, or nullopt. The
// string-valued counterpart of nodeAttrInt/nodeAttrFloat (the region name and
// the dense layout are the strings passes read).
std::optional<llvm::StringRef> nodeAttrString(mlir::ArrayAttr nodeAttrs,
                                             llvm::StringRef name) {
  for (mlir::Attribute entry : nodeAttrs) {
    auto attr = mlir::cast<boom::NodeAttr>(entry);
    if (attr.getName() != name)
      continue;
    if (auto value = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
      return value.getValue();
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

// The one q8_1 block the decode-side activation quantizer writes: an fp32
// scale and 32 int8 quants (quant/cuda_resident.h BlockQ8_1; the compiler does
// not include the CUDA headers, so the two constants are restated here).
constexpr uint64_t kQ8_1BlockBytes = 36;

// The bytes a q8_1 activation of `shape` occupies: 32-element blocks, 36 bytes
// each. Every dimension the emitters produce is a multiple of 32.
uint64_t q8_1ActivationBytes(llvm::ArrayRef<int64_t> shape) {
  uint64_t elements = 1;
  for (int64_t d : shape)
    elements *= uint64_t(d);
  return (elements / 32) * kQ8_1BlockBytes;
}

// The bytes an mmq q8_1 activation for [n, dim] occupies: the same blocks
// mmqQ8Fits requires, guard included.
uint64_t mmqActivationBytes(int64_t dim, int64_t n) {
  const uint64_t planes = uint64_t(dim) / 128 + ((uint64_t(dim) / 128) & 1);
  return (planes * uint64_t(n) + kMmqTailGuardBlocks) * kMmqBlockBytes;
}

// The fixed part of mmqActivationBytes: the tail guard does not scale with the
// batch, so a symbolic (extent-parameterized) plan records it separately.
constexpr uint64_t mmqActivationBaseBytes() {
  return kMmqTailGuardBlocks * kMmqBlockBytes;
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
        // keeps the input's shape and dtype (the CUDA kernels read the buffer
        // as q8_1; nothing else reads it), and is sized as the q8_1 blocks it
        // holds rather than as its f32 superset (rewrites.cpp:105-120).
        const boom::TensorDescAttr xt = view.desc(x);
        const int64_t qid = view.nextTensorId();
        view.appendTensor(boom::TensorDescAttr::get(
            view.ctx(), qid, xt.getShape(), xt.getDtype(),
            q8_1ActivationBytes(xt.getShape()),
            /*dynamic_base_bytes=*/0, /*is_input=*/false, /*is_output=*/false,
            /*is_constant=*/false, /*is_external=*/false,
            ("q8_" + xt.getName()).str()));

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

// FuseResidualEpiloguePass — the decode residual add whose producer is a norm,
// folded into the norm's apply loop (the gemv form is
// ResidualAddIntoGemvPass above; between them, a decode graph has no
// standalone residual add left).
//
// This is the shape the Gemma sandwich norm leaves in the decode graph:
// `Add(rms_norm(proj), residual)` for the attn and FFN post-norms, and
// `Add(rms_norm_heads(x), scaled_embedding)` for the PLE path — 127 add_kernel
// launches per decode step at 42 layers. The norm writes the add's tensor and
// takes the residual as an appended input, marked `add_input` (the same
// attribute and index convention the gemv epilogue uses). The kernel applies
// `y = norm(x) + residual` in one launch; the f32 value is the one the pair
// produced (the add of the stored norm value, with no multiply contraction).
//
// Decode form only, like the gemv residual fold: a `batch` (prefill) or
// `verify` node is left alone (prefill has the generated rms_norm_add region;
// verify runs the per-row decode norm).
class FuseResidualEpiloguePass
    : public mlir::PassWrapper<FuseResidualEpiloguePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseResidualEpiloguePass)

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

    // Add op -> (norm op, operand index the residual sits at).
    llvm::SmallVector<mlir::Operation *> adds;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> normOfAdd;
    llvm::DenseMap<mlir::Operation *, unsigned> residualOfAdd;
    // std::string, not StringRef: boomOpName returns by value, and a StringRef
    // to the temporary would dangle.
    const std::string rmsNorm = boomOpName(planner::OpKind::RmsNorm);
    const std::string rmsHeads = boomOpName(planner::OpKind::RmsNormHeads);
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::Add) ||
          node.getNumOperands() != 2 || node.getNumResults() != 1)
        continue;
      // A decode Add may carry no node_attrs at all (the PLE post-norm add);
      // only the batched/verify forms are excluded.
      auto addAttrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (addAttrs && (nodeAttrInt(addAttrs, "batch") ||
                       nodeAttrInt(addAttrs, "verify")))
        continue;
      for (unsigned s = 0; s < 2; ++s) {
        const int64_t proj = view.idOf(node.getOperand(s));
        auto producedBy = producer.find(proj);
        if (producedBy == producer.end())
          continue;
        mlir::Operation *m = producedBy->second;
        const llvm::StringRef name = m->getName().getStringRef();
        if ((name != rmsNorm && name != rmsHeads) || m->getNumResults() != 1 ||
            m->getNumOperands() != 1)
          continue;
        auto attrs = m->getAttrOfType<mlir::ArrayAttr>("node_attrs");
        if (!attrs || nodeAttrInt(attrs, "batch") || nodeAttrInt(attrs, "verify"))
          continue; // batched/verify form: no residual epilogue
        if (consumers[proj] != 1)
          continue; // the norm has another reader
        adds.push_back(&node);
        normOfAdd.insert({&node, m});
        residualOfAdd.insert({&node, 1 - s});
        break;
      }
    }

    if (adds.empty())
      return;

    for (mlir::Operation *add : adds) {
      mlir::Operation *m = normOfAdd.lookup(add);
      const unsigned residual = residualOfAdd.lookup(add);

      llvm::SmallVector<mlir::Value> operands;
      for (mlir::Value operand : m->getOperands())
        operands.push_back(operand);
      operands.push_back(add->getOperand(residual));

      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          m->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      // The residual's input index, as for the gemv epilogue.
      nodeAttrs.push_back(boom::NodeAttr::get(view.ctx(), "add_input",
                                              builder.getI64IntegerAttr(
                                                  m->getNumOperands())));

      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*m));
      // buildNode takes the bare op name and adds the dialect prefix.
      const char *opName =
          m->getName().getStringRef() == rmsNorm
              ? planner::op_kind_name(planner::OpKind::RmsNorm)
              : planner::op_kind_name(planner::OpKind::RmsNormHeads);
      // At the Add's place, not the norm's: the residual is only guaranteed to
      // dominate the Add (the PLE chain's scale runs between the norm and the
      // add), and the norm's output has no reader but this add.
      builder.setInsertionPoint(add);
      mlir::Operation *node = buildNode(
          builder, loc, opName,
          GraphEditor::nodeId(*add),
          add->getAttrOfType<mlir::StringAttr>("node_name").getValue(),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*add),
          operands, add->getResultTypes(),
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

// fuse_sibling_gemv_pass (D1): projections that read the same quantized
// activation — the q/k/v trio of one attention, the gate/up pair of one FFN —
// become one node, and the CUDA backend runs them as one launch.
//
// The decode graph quantizes an activation once and feeds every projection that
// consumes it, so those projections are separate nodes and separate kernel
// launches. At decode shape each launch is small enough that its ramp/tail
// dominates its memory time (nsys on Qwen2.5 0.5B: the same q5_0 gemv kernel
// reaches 203 GB/s at 48640 rows, 181 at 4864, 132 at 896), so one launch over
// the members' rows laid end to end pays that cost once. Every row keeps its own
// lane-to-block walk and 32-lane reduction in the merged kernel, so the outputs
// are bit-identical to the separate launches.
//
// Attribute form: the first projection keeps the node and its identity, gains
// `s1_slot` (`s2_slot`) naming the members' weights and `s1_bias_slot`
// (`s2_bias_slot`) for their folded biases, and carries the members' output
// tensors as extra results. A backend that cannot run the merged form falls
// back to the members' separate launches in order, so this is a scheduling
// change only. Decode form only: a `batch` (mmq prefill) or `verify` node is
// left alone, as is one carrying `add_input` (the merged kernel has no residual
// epilogue).
class FuseSiblingGemvPass
    : public mlir::PassWrapper<FuseSiblingGemvPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseSiblingGemvPass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    // Projections grouped by the activation tensor they read (operand 0 — the
    // only operand of a decode gemv), in body order.
    llvm::DenseMap<int64_t, llvm::SmallVector<mlir::Operation *>> groups;
    llvm::SmallVector<int64_t> order;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() !=
          boomOpName(planner::OpKind::QuantizedMatmulQ8))
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (nodeAttrInt(attrs, "batch") || nodeAttrInt(attrs, "verify")) continue;
      if (nodeAttrInt(attrs, "add_input")) continue;
      if (!nodeAttrInt(attrs, "slot")) continue;
      if (node.getNumResults() != 1 || node.getNumOperands() != 1) continue;
      const int64_t act = view.idOf(node.getOperand(0));
      auto [it, inserted] = groups.try_emplace(act);
      if (inserted) order.push_back(act);
      it->second.push_back(&node);
    }

    for (int64_t act : order) {
      llvm::SmallVector<mlir::Operation *> &g = groups.find(act)->second;
      if (g.size() < 2) continue;
      const std::size_t members = std::min<std::size_t>(g.size(), 3);
      mlir::Operation *head = g[0];

      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          head->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      llvm::SmallVector<int64_t> resultIds;
      llvm::SmallVector<mlir::Type> resultTypes;
      for (std::size_t k = 0; k < members; ++k) {
        mlir::Operation *m = g[k];
        auto attrs = m->getAttrOfType<mlir::ArrayAttr>("node_attrs");
        if (k > 0) {
          // The member's weight and its folded bias (when it has one), named
          // after its position in the merged launch.
          const std::string prefix = k == 1 ? "s1_" : "s2_";
          const std::string slotName = prefix + "slot";
          nodeAttrs.push_back(boom::NodeAttr::get(
              view.ctx(), slotName, builder.getI64IntegerAttr(*nodeAttrInt(attrs, "slot"))));
          if (auto bias = nodeAttrInt(attrs, "bias_slot")) {
            const std::string biasName = prefix + "bias_slot";
            nodeAttrs.push_back(boom::NodeAttr::get(
                view.ctx(), biasName, builder.getI64IntegerAttr(*bias)));
          }
        }
        resultIds.push_back(GraphEditor::resultIds(*m).front());
        resultTypes.push_back(m->getResult(0).getType());
      }

      // The head keeps its id, name, place and activation; the members' outputs
      // become the node's additional results.
      builder.setInsertionPoint(head);
      mlir::Operation *node = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::QuantizedMatmulQ8),
          GraphEditor::nodeId(*head),
          head->getAttrOfType<mlir::StringAttr>("node_name").getValue(),
          builder.getArrayAttr(nodeAttrs), resultIds, head->getOperands(),
          resultTypes, head->getAttrOfType<mlir::StringAttr>("backend"));

      for (std::size_t k = 0; k < members; ++k) {
        view.noteValue(resultIds[k], node->getResult(k));
        g[k]->getResult(0).replaceAllUsesWith(node->getResult(k));
      }
      for (std::size_t k = 0; k < members; ++k) g[k]->erase();
    }
    view.commitTable();
  }
};

// fuse_glue_regions_pass (Stage 5): a maximal glue region becomes one node the
// compiler lowers to a device kernel through MLIR (linalg/gpu -> NVVM -> PTX;
// see compiler/GlueCodegen.h).
//
// The earlier decode passes have already done the region formation the design
// calls for: the residual Add is folded into the gemv epilogue and rms_norm +
// q8_1 quantize are one node. So the region this pass recognizes is the decode
// `rms_norm_quantize` node itself (85 per Gemma decode step: the two norms of
// each of the 42 layers plus the final norm; 49 for Qwen2.5 0.5B). The node is rewritten in place as
// `boom.fused_region`, with the region's name in the `fused_region` attribute;
// node id, name, attributes, operands, results and backend all carry over, so
// the resulting plan differs only in the op kind (and in the kernel the CUDA
// backend dispatches).
//
// The switch is the GlueCodegen runtime feature: with it off the pass does not
// run, so the ablation arm plans exactly the hand-written kernel's graph.
// SONICBOOM_IR_PASS_DISABLE=FuseGlueRegionsPass does the same from the pass side.
class FuseGlueRegionsPass
    : public mlir::PassWrapper<FuseGlueRegionsPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGlueRegionsPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    llvm::SmallVector<mlir::Operation *> regions;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() !=
          boomOpName(planner::OpKind::RmsNormQuantize))
        continue;
      // The prefill (batched, mmq-layout) form is Stage 5b; this pass runs in
      // the decode pipeline, where the absence of `batch` is what selects the
      // q8_1 (32-element block) region shape.
      if (node.hasAttr("batch"))
        continue;
      regions.push_back(&node);
    }

    for (mlir::Operation *node : regions) {
      auto attrs =
          llvm::to_vector(node->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "fused_region",
          builder.getStringAttr("rms_norm_quantize_q8_1")));
      auto name = node->getAttrOfType<mlir::StringAttr>("node_name");

      builder.setInsertionPoint(node);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*node),
          name ? name.getValue() : llvm::StringRef("fused_region"),
          builder.getArrayAttr(attrs), GraphEditor::resultIds(*node),
          node->getOperands(), node->getResultTypes(),
          node->getAttrOfType<mlir::StringAttr>("backend"));

      node->replaceAllUsesWith(fused);
      view.noteReplaced(node, fused);
      node->erase();
    }

    // The region keeps its result tensor ids, so the table is unchanged; the
    // commit is still the one place the (unchanged) view is written back.
    view.commitTable();
  }
};

// FuseAddRmsNormMmqPass — the prefill glue region (Stage 5b): a residual add
// whose output a batched RmsNormMmq quantizer reads, and the quantizer itself,
// become one GPU-dialect region. `add -> rms_norm_quantize` is the prefill
// twin of the decode pair FuseGlueRegionsPass fuses; unlike that one, the sum
// cannot be dropped.
//
// The add's output has two readers — this norm and the next residual add — so
// the fused region keeps it: its second result is the residual sum, written by
// the same kernel that normalizes it. The norm's reader then sees the add's
// tensor id and the next add sees the same id; only the two nodes disappear.
// The kernel is emitted per [dim] row with a 256-thread block, phase A writing
// the sum and computing inv, phase B quantizing in the mmq geometry (see
// GlueCodegen.cpp). Runs after RmsNormMmqPass: that is what creates the
// batched quantizer this matches.
//
// The region is still a glue region in the codegen sense, so it is gated by the
// same `glue_codegen` feature and has its own entry in the pass-disable list.
class FuseAddRmsNormMmqPass
    : public mlir::PassWrapper<FuseAddRmsNormMmqPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAddRmsNormMmqPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    llvm::DenseMap<int64_t, mlir::Operation *> producer;
    for (mlir::Operation &node : view.body())
      for (int64_t id : GraphEditor::resultIds(node))
        producer.insert({id, &node});

    struct Match {
      mlir::Operation *add;
      mlir::Operation *norm;
    };
    llvm::SmallVector<Match> matches;
    llvm::DenseSet<mlir::Operation *> usedAdds;
    const std::string normName = boomOpName(planner::OpKind::RmsNormQuantize);
    const std::string addName = boomOpName(planner::OpKind::Add);
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != normName || node.getNumOperands() != 1 ||
          node.getNumResults() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs || !nodeAttrInt(attrs, "batch"))
        continue; // the decode form is not this region
      const int64_t ds = nodeAttrInt(attrs, "ds").value_or(-1);
      if (ds != 0 && ds != 1)
        continue; // D4_Q8_1 (-1/2) is not in this region's quantize
      const int64_t dim = nodeAttrInt(attrs, "dim").value_or(0);
      if (dim <= 0 || dim % 128 != 0)
        continue; // the reference quantizer requires whole 128-element blocks
      auto it = producer.find(view.idOf(node.getOperand(0)));
      if (it == producer.end())
        continue;
      mlir::Operation *add = it->second;
      if (add->getName().getStringRef() != addName ||
          add->getNumOperands() != 2 || add->getNumResults() != 1)
        continue;
      if (!usedAdds.insert(add).second)
        continue;
      matches.push_back({add, &node});
    }

    for (const Match &m : matches) {
      mlir::Operation *add = m.add;
      mlir::Operation *norm = m.norm;
      auto attrs =
          llvm::to_vector(norm->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "fused_region",
          builder.getStringAttr("add_rms_norm_quantize_mmq_q8_1")));
      auto name = add->getAttrOfType<mlir::StringAttr>("node_name");

      // Result order is the norm's first, so the ordinary index-wise id mapping
      // below stays right for it, and the add's second.
      llvm::SmallVector<int64_t> resultIds{
          GraphEditor::resultIds(*norm)[0], GraphEditor::resultIds(*add)[0]};
      llvm::SmallVector<mlir::Type> types{norm->getResult(0).getType(),
                                         add->getResult(0).getType()};
      const int64_t addId = GraphEditor::resultIds(*add)[0];
      const int64_t normId = GraphEditor::resultIds(*norm)[0];

      builder.setInsertionPoint(add);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*add),
          name ? name.getValue() : llvm::StringRef("add_rms_norm_quantize"),
          builder.getArrayAttr(attrs), resultIds, add->getOperands(), types,
          add->getAttrOfType<mlir::StringAttr>("backend"));

      // Index-wise, not norm->replaceAllUsesWith(fused): the fused node has two
      // results and the norm's is only the first.
      norm->getResult(0).replaceAllUsesWith(fused->getResult(0));
      add->getResult(0).replaceAllUsesWith(fused->getResult(1));
      view.noteValue(normId, fused->getResult(0));
      view.noteValue(addId, fused->getResult(1));
      norm->erase();
      add->erase();
    }

    view.commitTable();
  }
};

// FuseRmsNormAddPass — the prefill glue region (Stage 5c): a batched RMS-norm
// whose only reader is a residual add, and that add, become one GPU-dialect
// region `rms_norm_add`. The kernel computes the row's inverse RMS with
// rms_heads_fused_kernel's reduction, then writes `residual + x * inv * w` —
// the pair's exact fp32 sequence.
//
// Unlike FuseAddRmsNormMmqPass the norm's result is private to the region (it
// has one reader, the add), so the region has one result, under the add's
// tensor id, and the norm's tensor leaves the table. The norm's attrs
// (batch/dim/layer/slot) are the region's: the kernel needs the row width and
// the weight; only the node id and name come from the add.
//
// Runs after FuseAddRmsNormMmqPass. That pass owns the `norm -> add ->
// norm+quantize` chain; by the time this one runs, the adds it did not take
// (the post-FF norm and the PLE post-norm) are the ones left with a
// single-reader norm in front of them.
//
// Gated by the `glue_codegen` feature like the other regions, with its own
// entry in the pass-disable list.
class FuseRmsNormAddPass
    : public mlir::PassWrapper<FuseRmsNormAddPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseRmsNormAddPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    struct Match {
      mlir::Operation *norm;
      mlir::Operation *add;
    };
    llvm::SmallVector<Match> matches;
    llvm::DenseSet<mlir::Operation *> usedAdds;
    const std::string normName = boomOpName(planner::OpKind::RmsNorm);
    const std::string addName = boomOpName(planner::OpKind::Add);
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != normName || node.getNumOperands() != 1 ||
          node.getNumResults() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs || !nodeAttrInt(attrs, "batch"))
        continue; // the decode form is not this region
      if (nodeAttrInt(attrs, "dim").value_or(0) <= 0)
        continue;
      if (nodeAttrInt(attrs, "slot").value_or(0) <= 0)
        continue; // the generated kernel applies a norm weight (slot 0 = none)
      if (!node.getResult(0).hasOneUse())
        continue;
      mlir::Operation *add = node.getResult(0).use_begin()->getOwner();
      if (add->getName().getStringRef() != addName || add->getNumOperands() != 2 ||
          add->getNumResults() != 1)
        continue;
      if (!usedAdds.insert(add).second)
        continue; // both operands are single-reader norms; one region per add
      matches.push_back({&node, add});
    }

    for (const Match &m : matches) {
      // Re-read the operands here, not at match time: an earlier rewrite in
      // this same pass may have replaced the add this norm feeds into (x2 is
      // the post-FF region's result and later the PLE x3 add's residual), and
      // a Value captured before that rewrite would point at the erased op.
      mlir::Value x = m.norm->getOperand(0);
      // The other add operand is the residual stream; operand order is not
      // fixed by the emitter (the post-FF chain puts the norm first, the PLE
      // post-norm chain puts the residual first).
      mlir::Value residual = m.add->getOperand(0) == m.norm->getResult(0)
                                 ? m.add->getOperand(1)
                                 : m.add->getOperand(0);
      auto attrs = llvm::to_vector(
          m.norm->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "fused_region", builder.getStringAttr("rms_norm_add")));
      auto name = m.add->getAttrOfType<mlir::StringAttr>("node_name");
      const int64_t addId = GraphEditor::resultIds(*m.add)[0];
      const int64_t normId = GraphEditor::resultIds(*m.norm)[0];
      // Kernel ABI: (x, residual, w, y, dim, eps).
      llvm::SmallVector<mlir::Value> operands{x, residual};

      builder.setInsertionPoint(m.add);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*m.add),
          name ? name.getValue() : llvm::StringRef("rms_norm_add"),
          builder.getArrayAttr(attrs), GraphEditor::resultIds(*m.add), operands,
          m.add->getResultTypes(),
          m.add->getAttrOfType<mlir::StringAttr>("backend"));

      m.add->getResult(0).replaceAllUsesWith(fused->getResult(0));
      view.noteValue(addId, fused->getResult(0));
      // The norm's tensor is now unreferenced. Drop it before the table is
      // written back: an entry with no producer and no reader would otherwise
      // turn into a free block argument when the graph is read again.
      view.dropTensor(normId);
      // The add holds the norm result's only use, so it has to go first: an op
      // with a live use cannot be erased.
      m.add->erase();
      m.norm->erase();
    }

    view.commitTable();
  }
};

// FuseSliceScalePass — the PLE column extraction (Stage 5e). The per-layer
// embedding column is a strided Slice whose result is immediately scaled (the
// Gemma PLE scale) before the layer combine:
//
//   slice(stride>0) -> scale   =>   fused_region "slice_scale"
//
// The slice's result has no other reader, so the chain is one launch instead of
// two. The fused node keeps the scale's tensor ids (its result is the graph
// value) and the slice's offset/length/stride, and carries the scale factor as
// the `scale` attribute. The generated kernel is slice_strided_kernel's index
// arithmetic with scale_kernel's multiply folded in, so the output is
// byte-identical to the pair.
class FuseSliceScalePass
    : public mlir::PassWrapper<FuseSliceScalePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseSliceScalePass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    const std::string sliceName = boomOpName(planner::OpKind::Slice);
    const std::string scaleName = boomOpName(planner::OpKind::Scale);
    struct Match {
      mlir::Operation *slice;
      mlir::Operation *scale;
    };
    llvm::SmallVector<Match> matches;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != sliceName ||
          node.getNumOperands() != 1 || node.getNumResults() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs || nodeAttrInt(attrs, "stride").value_or(0) <= 0)
        continue; // the contiguous span form is not this region
      if (!node.getResult(0).hasOneUse())
        continue;
      mlir::Operation *scale = node.getResult(0).use_begin()->getOwner();
      if (scale->getName().getStringRef() != scaleName ||
          scale->getNumOperands() != 1 || scale->getNumResults() != 1)
        continue;
      matches.push_back({&node, scale});
    }

    for (const Match &m : matches) {
      // Re-read the operands now, not at match time (see FuseRmsNormAddPass).
      mlir::Value x = m.slice->getOperand(0);
      auto sliceAttrs =
          llvm::to_vector(m.slice->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
      auto scaleAttrs = m.scale->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      auto scaleValue = scaleAttrs ? nodeAttrFloat(scaleAttrs, "scale")
                                   : std::nullopt;
      if (!scaleValue) continue; // no scale factor to fold in
      llvm::SmallVector<mlir::Attribute> attrs(sliceAttrs.begin(),
                                               sliceAttrs.end());
      // Keep the planner's exact f64 value (buildNode re-emits the typed attr).
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "scale", builder.getF64FloatAttr(*scaleValue)));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "fused_region", builder.getStringAttr("slice_scale")));
      auto name = m.scale->getAttrOfType<mlir::StringAttr>("node_name");
      const int64_t scaleId = GraphEditor::resultIds(*m.scale)[0];
      const int64_t sliceId = GraphEditor::resultIds(*m.slice)[0];

      builder.setInsertionPoint(m.scale);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*m.scale),
          name ? name.getValue() : llvm::StringRef("slice_scale"),
          builder.getArrayAttr(attrs), GraphEditor::resultIds(*m.scale), {x},
          m.scale->getResultTypes(),
          m.scale->getAttrOfType<mlir::StringAttr>("backend"));

      m.scale->getResult(0).replaceAllUsesWith(fused->getResult(0));
      view.noteValue(scaleId, fused->getResult(0));
      view.dropTensor(sliceId);
      m.scale->erase();
      m.slice->erase();
    }

    view.commitTable();
  }
};

// The gated activation (Stage 5f): an activation followed by the gate*up
// multiply is one region when the product feeds a dense projection. Both forms
// share the match and the rewrite; only the activation op and the region name
// differ.
//
//   gelu_fp16(gate) -> mul(_, up)   =>   fused_region "gelu_mul"
//   silu(gate)      -> mul(_, up)   =>   fused_region "silu_mul"
//
// The generated kernel is the two reference kernels' arithmetic in one launch,
// and the activation result has no other reader, so the intermediate tensor is
// dropped. The passes run after the mmq activation folds, which take the chains
// that feed a quantized matmul; what remains here feeds a dense projection.
namespace {

void fuseActivationMul(boom::GraphOp graph, planner::OpKind actKind,
                       llvm::StringRef region) {
  GraphEditor view(graph);
  mlir::OpBuilder builder(view.ctx());
  const mlir::Location loc = graph.getLoc();

  const std::string actName = boomOpName(actKind);
  const std::string mulName = boomOpName(planner::OpKind::Mul);
  struct Match {
    mlir::Operation *act;
    mlir::Operation *mul;
  };
  llvm::SmallVector<Match> matches;
  for (mlir::Operation &node : view.body()) {
    if (node.getName().getStringRef() != actName ||
        node.getNumOperands() != 1 || node.getNumResults() != 1)
      continue;
    if (!node.getResult(0).hasOneUse())
      continue;
    mlir::Operation *mul = node.getResult(0).use_begin()->getOwner();
    if (mul->getName().getStringRef() != mulName || mul->getNumOperands() != 2 ||
        mul->getNumResults() != 1)
      continue;
    matches.push_back({&node, mul});
  }

  for (const Match &m : matches) {
    // Re-read the operands now, not at match time (see FuseRmsNormAddPass).
    mlir::Value gate = m.act->getOperand(0);
    mlir::Value up = m.mul->getOperand(0) == m.act->getResult(0)
                         ? m.mul->getOperand(1)
                         : m.mul->getOperand(0);
    auto attrs =
        llvm::to_vector(m.mul->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
    attrs.push_back(boom::NodeAttr::get(
        view.ctx(), "fused_region", builder.getStringAttr(region)));
    auto name = m.mul->getAttrOfType<mlir::StringAttr>("node_name");
    const int64_t mulId = GraphEditor::resultIds(*m.mul)[0];
    const int64_t actId = GraphEditor::resultIds(*m.act)[0];

    builder.setInsertionPoint(m.mul);
    mlir::Operation *fused = buildNode(
        builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
        GraphEditor::nodeId(*m.mul),
        name ? name.getValue() : region,
        builder.getArrayAttr(attrs), GraphEditor::resultIds(*m.mul),
        {gate, up}, m.mul->getResultTypes(),
        m.mul->getAttrOfType<mlir::StringAttr>("backend"));

    m.mul->getResult(0).replaceAllUsesWith(fused->getResult(0));
    view.noteValue(mulId, fused->getResult(0));
    view.dropTensor(actId);
    m.mul->erase();
    m.act->erase();
  }

  view.commitTable();
}

}  // namespace

// FuseGeluMulPass — the GELU form (Gemma's FFN and PLE gate).
class FuseGeluMulPass
    : public mlir::PassWrapper<FuseGeluMulPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGeluMulPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;
    fuseActivationMul(getOperation(), planner::OpKind::GeluFp16, "gelu_mul");
  }
};

// FuseSiluMulPass — the SwiGLU form. Same chain and same rewrite as the GELU
// pass; it is separate because its numerics profile decides whether it runs
// (the caller adds the pass only under the standard profile, whose acceptance
// gate covers the generated exp, where the strict profile's plan goldens are
// the historical contract). What it takes here is the Qwen2-class FFN whose
// down projection has no mmq activation layout (Q6_K): SiluMulMmqPass has
// already folded every chain that feeds an mmq matmul.
class FuseSiluMulPass
    : public mlir::PassWrapper<FuseSiluMulPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseSiluMulPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;
    fuseActivationMul(getOperation(), planner::OpKind::Silu, "silu_mul");
  }
};

// FuseHeadsNormRopeCastPass — the attention-prep glue region (Stage 5d). The
// per-head RMS-norm that starts each of Q/K/V, the NeoX RoPE on Q/K, and the
// fp16 rounding on K/V are one region per chain:
//
//   rope=1, cast=0   Q: rms_norm_heads -> rope
//   rope=1, cast=1   K: rms_norm_heads -> rope -> cast_fp16
//   rope=0, cast=1   V: rms_norm_heads -> cast_fp16
//
// The norm's attribute record carries layer/slot (and `batch` in prefill), the
// rope contributes its position operand, and the chain's last node names the
// region and keeps its tensor ids. The intermediate tensors have no other
// reader and are dropped with their producers. One generated kernel replaces
// the chain's 2-3 launches, so a layer's attention prep is 3 launches instead
// of 7 in decode.
//
// The KV-cache append stays inside OpKind::Attention: the cache buffers are
// backend resources, not graph tensors, so no graph region can reach them
// (recorded as a Stage 5d blocker in the run report).
//
// Runs last in both pipelines. The earlier folds take norms that feed a
// quantizer or a matmul; the attention-prep norms feed rope/cast, which none of
// them match, so the ordering is by construction and not by luck.
class FuseHeadsNormRopeCastPass
    : public mlir::PassWrapper<FuseHeadsNormRopeCastPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseHeadsNormRopeCastPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    struct Match {
      mlir::Operation *norm;
      mlir::Operation *rope; // null for the V chain
      mlir::Operation *cast; // null for the Q chain
    };
    llvm::SmallVector<Match> matches;
    const std::string normName = boomOpName(planner::OpKind::RmsNormHeads);
    const std::string ropeName = boomOpName(planner::OpKind::Rope);
    const std::string castName = boomOpName(planner::OpKind::CastFp16);
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != normName ||
          node.getNumOperands() != 1 || node.getNumResults() != 1 ||
          !node.getResult(0).hasOneUse())
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs)
        continue;
      // A segment-width norm (the per-layer projection norm) is not a head
      // norm: the region's geometry is one block per head, dim = head_dim.
      if (nodeAttrInt(attrs, "dim").value_or(0) > 0)
        continue;
      if (!nodeAttrInt(attrs, "layer").has_value())
        continue;

      mlir::Operation *use = node.getResult(0).use_begin()->getOwner();
      mlir::Operation *rope = nullptr, *cast = nullptr;
      if (use->getName().getStringRef() == ropeName) {
        // The rope reads the norm and the position: (x, pos).
        if (use->getNumOperands() != 2 || use->getNumResults() != 1 ||
            use->getOperand(0) != node.getResult(0))
          continue;
        rope = use;
        if (!rope->getResult(0).hasOneUse())
          continue;
        mlir::Operation *ruse = rope->getResult(0).use_begin()->getOwner();
        if (ruse->getName().getStringRef() == castName &&
            ruse->getNumOperands() == 1 && ruse->getNumResults() == 1 &&
            ruse->getOperand(0) == rope->getResult(0))
          cast = ruse;
      } else if (use->getName().getStringRef() == castName) {
        if (use->getNumOperands() != 1 || use->getNumResults() != 1 ||
            use->getOperand(0) != node.getResult(0))
          continue;
        cast = use;
      } else {
        continue;
      }
      mlir::Operation *last = cast ? cast : rope;
      if (!last)
        continue;
      matches.push_back({&node, rope, cast});
    }

    for (const Match &m : matches) {
      mlir::Operation *last = m.cast ? m.cast : m.rope;
      auto attrs = llvm::to_vector(
          m.norm->getAttrOfType<mlir::ArrayAttr>("node_attrs"));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "rope",
          builder.getI32IntegerAttr(m.rope ? 1 : 0)));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "cast",
          builder.getI32IntegerAttr(m.cast ? 1 : 0)));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "fused_region",
          builder.getStringAttr("heads_norm_rope_cast")));
      auto name = last->getAttrOfType<mlir::StringAttr>("node_name");
      const int64_t lastId = GraphEditor::resultIds(*last)[0];
      const int64_t normId = GraphEditor::resultIds(*m.norm)[0];
      // Kernel ABI: (x, w, has_w, y, pos, head_dim, eps, freq_scale, log_base,
      // ff, heads_per_pos, do_rope, do_cast). `x` is the norm's input; `pos`
      // is the rope's position operand (the position is a backend buffer; the
      // operand keeps the graph dependency).
      llvm::SmallVector<mlir::Value> operands{m.norm->getOperand(0)};
      if (m.rope) operands.push_back(m.rope->getOperand(1));

      builder.setInsertionPoint(last);
      mlir::Operation *fused = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*last),
          name ? name.getValue() : llvm::StringRef("heads_norm_rope_cast"),
          builder.getArrayAttr(attrs), GraphEditor::resultIds(*last), operands,
          last->getResultTypes(),
          last->getAttrOfType<mlir::StringAttr>("backend"));

      last->getResult(0).replaceAllUsesWith(fused->getResult(0));
      view.noteValue(lastId, fused->getResult(0));
      view.dropTensor(normId);
      // The rope's tensor is an intermediate only when a cast follows it; when
      // the rope is the chain's last node its id is the region's result id.
      if (m.rope && m.cast)
        view.dropTensor(GraphEditor::resultIds(*m.rope)[0]);
      // The chain is linear, so erase from the last node back.
      last->erase();
      if (m.rope && m.cast) m.rope->erase();
      m.norm->erase();
    }

    view.commitTable();
  }
};

// FuseKvAppendIntoPrepPass — the KV-cache append into the attention-prep region
// (Stage 6d). The cache used to be a backend resource no graph region could
// reach, which is why the append stayed inside OpKind::Attention through Stage
// 5d; Stage 6a made it a graph resource (the `!boom.kv` handle), so the K and V
// prep regions can carry the handle and the append becomes a store the
// generated kernel performs on values it already has in registers.
//
//   K prep (heads_norm_rope_cast, rope=1, cast=1): kv_store = 1
//   V prep (heads_norm_rope_cast, rope=0, cast=1): kv_store = 2
//
// Both region nodes take the position and then the layer's handle as their last
// two operands (the V chain has no rope to carry the position), and the
// attention node is marked `kv_appended` so the backend skips its two
// kv_append launches. The cache content is unchanged: the stored values are the
// fp16-rounded values the region already writes to its output, which is what
// Attention appended before.
//
// Decode only: the prefill graph appends a whole segment with
// kv_append_batched, and has no OpKind::Attention node to match.
class FuseKvAppendIntoPrepPass
    : public mlir::PassWrapper<FuseKvAppendIntoPrepPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseKvAppendIntoPrepPass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

    boom::GraphOp graph = getOperation();
    mlir::OpBuilder builder(graph.getContext());
    const std::string attnName = boomOpName(planner::OpKind::Attention);

    // The region `op` is a heads_norm_rope_cast chain with `rope` and a cast:
    // the K chain (rope=1) and the V chain (rope=0) are the two that append.
    auto is_prep = [](mlir::Operation *op, int64_t rope) {
      if (op == nullptr ||
          op->getName().getStringRef() != "boom.fused_region")
        return false;
      auto attrs = op->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs) return false;
      auto region = nodeAttrString(attrs, "fused_region");
      return region && *region == "heads_norm_rope_cast" &&
             nodeAttrInt(attrs, "rope").value_or(0) == rope &&
             nodeAttrInt(attrs, "cast").value_or(0) == 1;
    };
    auto add_int_attr = [&](mlir::Operation *op, llvm::StringRef name,
                            int32_t value) {
      auto attrs = op->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      llvm::SmallVector<mlir::Attribute> entries(attrs.getValue());
      entries.push_back(boom::NodeAttr::get(graph.getContext(), name,
                                            builder.getI32IntegerAttr(value)));
      op->setAttr("node_attrs", builder.getArrayAttr(entries));
    };

    for (mlir::Operation &node : graph.getBody().front()) {
      if (node.getName().getStringRef() != attnName) continue;
      // {q, kc, vc, pos, handle}: only a KV layer has K/V to append. A shared
      // layer reads its donor's cache and has nothing to write.
      if (node.getNumOperands() < 5) continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs || nodeAttrInt(attrs, "kv_appended").value_or(0) != 0)
        continue;
      mlir::Value handle = node.getOperand(node.getNumOperands() - 1);
      auto kv = llvm::dyn_cast<boom::KvType>(handle.getType());
      if (!kv) continue;
      mlir::Operation *k = node.getOperand(1).getDefiningOp();
      mlir::Operation *v = node.getOperand(2).getDefiningOp();
      if (!is_prep(k, 1) || !is_prep(v, 0)) continue;
      // The handle must be the layer the prep regions write (the verifier
      // states the same rule for the attention node).
      auto kattrs = k->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (nodeAttrInt(kattrs, "layer").value_or(-1) != (int64_t)kv.getLayer() ||
          nodeAttrInt(v->getAttrOfType<mlir::ArrayAttr>("node_attrs"), "layer")
                  .value_or(-1) != (int64_t)kv.getLayer())
        continue;

      mlir::Value pos = node.getOperand(3);
      // The K chain is {x, pos}; the V chain is {x}. Both take the handle last.
      llvm::SmallVector<mlir::Value> kop{k->getOperand(0), k->getOperand(1),
                                         handle};
      llvm::SmallVector<mlir::Value> vop{v->getOperand(0), pos, handle};
      k->setOperands(kop);
      v->setOperands(vop);
      add_int_attr(k, "kv_store", 1);
      add_int_attr(v, "kv_store", 2);
      add_int_attr(&node, "kv_appended", 1);
    }
  }
};

// BufferReusePass — liveness-based buffer assignment (Stage 7).
//
// The planner's memory assignment used to give every activation its own buffer.
// This pass computes, over the final node set of the phase, which activations
// can share one, and records the assignment on the nodes so it survives the
// Graph round trip:
//
//   inplace=1  the node's result may be written into operand 0's device
//              buffer. Only a same-type elementwise op qualifies (Scale, Add,
//              Mul, Silu), and only when operand 0 is a node result with no
//              use but this node and the result is not a graph output. The
//              CUDA backend borrows the buffer; the executor moves ownership
//              to the result value.
//   buf=N      the node's result belongs to buffer class N. A class is reused
//              by a later result of the same byte size once every earlier
//              member's last use is behind it; an in-place result joins its
//              operand's class.
//
// Both are facts about the graph (uses, sizes, the output list), not about a
// backend: a backend that ignores the attributes keeps its per-tensor
// allocation, so a mark can make a backend refuse an optimization but cannot
// make it compute a wrong value.
class BufferReusePass
    : public mlir::PassWrapper<BufferReusePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferReusePass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());

    llvm::DenseSet<int64_t> outputs;
    for (int64_t id : graph.getOutputs()) outputs.insert(id);

    // Liveness over the flat node list: the last node position that reads a
    // tensor (a graph output is live through the whole graph), and the position
    // that produces each node result.
    llvm::DenseMap<int64_t, int64_t> lastUse;
    llvm::DenseMap<int64_t, int64_t> producedAt;
    int64_t pos = 0;
    for (mlir::Operation &node : view.body()) {
      for (mlir::Value operand : node.getOperands())
        lastUse[view.idOf(operand)] = pos;
      for (int64_t id : GraphEditor::resultIds(node)) producedAt[id] = pos;
      ++pos;
    }
    for (int64_t id : graph.getOutputs()) lastUse[id] = INT64_MAX;

    struct BufferClass {
      uint64_t bytes = 0;
      int64_t lastLive = -1;
    };
    std::vector<BufferClass> classes;
    llvm::DenseMap<int64_t, int64_t> tensorClass;  // tensor id -> class id

    auto add_attr = [&](mlir::Operation *op, llvm::StringRef name,
                        int64_t value) {
      llvm::SmallVector<mlir::Attribute> entries;
      if (auto attrs = op->getAttrOfType<mlir::ArrayAttr>("node_attrs"))
        entries.append(attrs.begin(), attrs.end());
      entries.push_back(boom::NodeAttr::get(
          view.ctx(), name, builder.getI32IntegerAttr(int32_t(value))));
      op->setAttr("node_attrs", builder.getArrayAttr(entries));
    };

    auto is_inplace_candidate = [&](mlir::Operation &node) -> bool {
      const llvm::StringRef name = node.getName().getStringRef();
      if (name != boomOpName(planner::OpKind::Scale) &&
          name != boomOpName(planner::OpKind::Add) &&
          name != boomOpName(planner::OpKind::Mul) &&
          name != boomOpName(planner::OpKind::Silu))
        return false;
      if (node.getNumResults() != 1 || node.getNumOperands() < 1) return false;
      if (GraphEditor::resultIds(node).size() != 1) return false;
      mlir::Value src = node.getOperand(0);
      // A block argument is a graph input, a weight or a constant: none of
      // them is the node's to overwrite.
      if (src.getDefiningOp() == nullptr) return false;
      if (src.getType() != node.getResult(0).getType()) return false;
      if (!src.hasOneUse()) return false;
      return !outputs.contains(GraphEditor::resultIds(node)[0]);
    };

    pos = 0;
    for (mlir::Operation &node : view.body()) {
      const bool inplace = is_inplace_candidate(node);
      if (inplace) add_attr(&node, "inplace", 1);

      // A class names one result; a node with several keeps the planner's
      // per-tensor assignment for all of them.
      if (node.getNumResults() != 1 ||
          GraphEditor::resultIds(node).size() != 1) {
        ++pos;
        continue;
      }
      const int64_t rid = GraphEditor::resultIds(node)[0];
      const boom::TensorDescAttr desc = view.desc(rid);
      if (!desc) {
        ++pos;
        continue;
      }
      const uint64_t bytes = desc.getByteSize();

      int64_t cls = -1;
      if (inplace) {
        auto it = tensorClass.find(view.idOf(node.getOperand(0)));
        if (it != tensorClass.end()) cls = it->second;
      }
      if (cls < 0) {
        // Reuse by size class: a buffer of this byte size whose members are
        // all dead before this node.
        for (std::size_t c = 0; c < classes.size(); ++c)
          if (classes[c].bytes == bytes && classes[c].lastLive < pos) {
            cls = int64_t(c);
            break;
          }
        if (cls < 0) {
          cls = int64_t(classes.size());
          classes.push_back(BufferClass{bytes, -1});
        }
      }
      tensorClass[rid] = cls;
      add_attr(&node, "buf", cls);
      auto lu = lastUse.find(rid);
      const int64_t live =
          lu != lastUse.end() ? lu->second : producedAt.lookup(rid);
      classes[cls].lastLive = std::max(classes[cls].lastLive, live);
      ++pos;
    }
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
// weight-format design note, docs/weight-quant-format.md).
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
    llvm::DenseMap<mlir::Operation *, int64_t> outOfNorm;
    llvm::DenseMap<mlir::Operation *, uint64_t> q8BytesOfNorm;
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
      outOfNorm.insert({&node, out});
      q8BytesOfNorm.insert({&node, mmqActivationBytes(*dim, *nb)});
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

      // The buffer now holds q8_1 blocks only: declare the mmq activation size
      // (the f32 superset is no longer read by anyone).
      view.setTensorSize(outOfNorm.lookup(norm), q8BytesOfNorm.lookup(norm),
                         mmqActivationBaseBytes());
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
    const int64_t fhId = GraphEditor::resultIds(*mul)[0];
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

    // The Mul's buffer holds q8_1 blocks only now: declare the mmq activation
    // size in place of the f32 superset.
    view.setTensorSize(fhId,
                       mmqActivationBytes(dimOfAct.lookup(a),
                                          batchOfAct.lookup(a)),
                       mmqActivationBaseBytes());

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

// scale_into_dense_pass (step 10c): a scalar `scale` that reads the only output
// of a batched dense projection becomes an epilogue of the projection:
//
//   matvec_bf16 -> scale   =>   matvec_bf16 (epilogue_scale = s)
//
// In both models this is the per-layer model scaling applied to the projection
// of per_layer_model_proj (42 sites per Gemma prefill, 1 per decode step — the
// decode form is not matched). The projection keeps its identity, operands and
// place, takes the scale's output tensor over (the scale writes the projection's
// buffer in place) and carries the factor as `epilogue_scale`.
//
// The epilogue applies the factor in the projection's own store when the
// generated dense GEMM is selected — one f32 multiply, the same operation and
// rounding the separate scale kernel did — and as a scale pass after the
// cuBLASLt loop or the gemv fallback otherwise, so the ablation arm (generated
// GEMM off) has exactly the semantics it had before. Prefill only: the batched
// form, as dense_layout_propagation is.
class ScaleIntoDensePass
    : public mlir::PassWrapper<ScaleIntoDensePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScaleIntoDensePass)

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

    // Scale op -> (dense projection, the factor). These Scale nodes are exactly
    // the nodes the rewrite removes.
    llvm::SmallVector<mlir::Operation *> scales;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> denseOf;
    llvm::DenseMap<mlir::Operation *, double> factorOf;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::Scale) ||
          node.getNumOperands() != 1 || node.getNumResults() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      const std::optional<double> factor =
          attrs ? nodeAttrFloat(attrs, "scale") : std::nullopt;
      if (!factor)
        continue; // no scalar to fold in
      const int64_t src = view.idOf(node.getOperand(0));
      auto producedBy = producer.find(src);
      if (producedBy == producer.end())
        continue;
      mlir::Operation *m = producedBy->second;
      if (m->getName().getStringRef() !=
              boomOpName(planner::OpKind::MatvecBf16) ||
          m->getNumResults() != 1)
        continue;
      auto matvecAttrs = m->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!matvecAttrs || !nodeAttrInt(matvecAttrs, "batch"))
        continue; // decode form: the gemv has no epilogue
      bool already = false;
      for (mlir::Attribute entry : matvecAttrs) {
        auto attr = mlir::cast<boom::NodeAttr>(entry);
        if (attr.getName() == "epilogue_scale") already = true;
      }
      if (already)
        continue;
      if (consumers[src] != 1)
        continue; // the projection has another reader
      scales.push_back(&node);
      denseOf.insert({&node, m});
      factorOf.insert({&node, *factor});
    }

    if (scales.empty())
      return;

    for (mlir::Operation *scale : scales) {
      mlir::Operation *m = denseOf.lookup(scale);

      // The projection's attributes plus the factor it now applies, appended —
      // the same ordered record the other folds use.
      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          m->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      nodeAttrs.push_back(boom::NodeAttr::get(
          view.ctx(), "epilogue_scale",
          builder.getF64FloatAttr(factorOf.lookup(scale))));

      // The projection keeps its identity and place and takes the scale's
      // output; its own f32 output tensor stops being a tensor of the graph.
      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*m));
      builder.setInsertionPoint(m);
      mlir::Operation *node = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::MatvecBf16),
          GraphEditor::nodeId(*m),
          m->getAttrOfType<mlir::StringAttr>("node_name").getValue(),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*scale),
          m->getOperands(), scale->getResultTypes(),
          m->getAttrOfType<mlir::StringAttr>("backend"));

      scale->replaceAllUsesWith(node);
      view.noteReplaced(scale, node);
      scale->erase();
      for (int64_t id : dead)
        view.dropTensor(id);
      m->erase();
    }

    view.commitTable();
  }
};

// fold_scale_into_glue_pass (step 11): a scalar `scale` that reads the only
// output of a generated glue region becomes an epilogue of that region:
//
//   fused_region "rms_norm_add" -> scale   =>   fused_region (+ epilogue_scale)
//
// This is the glue-region counterpart of ScaleIntoDensePass. The generated
// dense GEMM's epilogue takes the scale of the projection beside it; the region
// that reads the GEMM's output in the Gemma PLE chain is `rms_norm_add`, and the
// scale that follows it is a chain of its own:
//
//   matvec_bf16 -> rms_norm_add -> scale
//
// The norm itself cannot be an epilogue of the GEMM: its reduction spans the
// projection's whole output row (256 wide in the PLE chain, 2560 in the
// post-FF one), while the dense GEMM's M tile partitions that row across blocks
// — the norm's row tile and the GEMM's output tile are different partitions.
// The scale, however, is elementwise on the region's own store, so
// the region absorbs it exactly: one fp32 multiply after the apply, the same
// operation and rounding the separate scale kernel did. The post-FF chain
// (matvec_f32 -> rms_norm_add -> scale) has the same shape and folds by the
// same rule; the pass matches the producer's region kind, not its input
// operator, so both chains are covered (42 + 42 sites per Gemma prefill).
//
// The region keeps its identity, operands and place, takes the scale's output
// tensor over (its own output tensor stops being a graph tensor) and carries
// the factor as `epilogue_scale`; the CUDA backend passes it to the generated
// kernel, whose apply multiplies by it. With the generated region off
// (GlueCodegen disabled) this pass does not run, so the ablation arm plans the
// separate scale kernel it always had.
class FoldScaleIntoGluePass
    : public mlir::PassWrapper<FoldScaleIntoGluePass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldScaleIntoGluePass)

  void runOnOperation() override {
    if (runtime::feature_disabled(runtime::Feature::GlueCodegen))
      return;

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

    llvm::SmallVector<mlir::Operation *> scales;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> regionOf;
    llvm::DenseMap<mlir::Operation *, double> factorOf;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() != boomOpName(planner::OpKind::Scale) ||
          node.getNumOperands() != 1 || node.getNumResults() != 1)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      const std::optional<double> factor =
          attrs ? nodeAttrFloat(attrs, "scale") : std::nullopt;
      if (!factor)
        continue; // no scalar to fold in
      const int64_t src = view.idOf(node.getOperand(0));
      auto producedBy = producer.find(src);
      if (producedBy == producer.end())
        continue;
      mlir::Operation *m = producedBy->second;
      if (m->getName().getStringRef() !=
              boomOpName(planner::OpKind::FusedRegion) ||
          m->getNumResults() != 1)
        continue;
      auto regionAttrs = m->getAttrOfType<mlir::ArrayAttr>("node_attrs");
      auto region =
          regionAttrs ? nodeAttrString(regionAttrs, "fused_region") : std::nullopt;
      // Only the region kind whose generated kernel has the epilogue multiply.
      if (!region || *region != "rms_norm_add")
        continue;
      if (nodeAttrFloat(regionAttrs, "epilogue_scale"))
        continue;
      if (consumers[src] != 1)
        continue; // the region has another reader
      scales.push_back(&node);
      regionOf.insert({&node, m});
      factorOf.insert({&node, *factor});
    }

    if (scales.empty())
      return;

    for (mlir::Operation *scale : scales) {
      mlir::Operation *m = regionOf.lookup(scale);

      // The region's attributes plus the factor it now applies, appended — the
      // same ordered record the other folds use.
      llvm::SmallVector<mlir::Attribute> nodeAttrs(
          m->getAttrOfType<mlir::ArrayAttr>("node_attrs").getValue());
      nodeAttrs.push_back(boom::NodeAttr::get(
          view.ctx(), "epilogue_scale",
          builder.getF64FloatAttr(factorOf.lookup(scale))));

      // The region keeps its identity and place and takes the scale's output;
      // its own f32 output tensor stops being a tensor of the graph.
      const llvm::SmallVector<int64_t> dead(GraphEditor::resultIds(*m));
      auto name = m->getAttrOfType<mlir::StringAttr>("node_name");
      builder.setInsertionPoint(m);
      mlir::Operation *node = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::FusedRegion),
          GraphEditor::nodeId(*m),
          name ? name.getValue() : llvm::StringRef("fused_region"),
          builder.getArrayAttr(nodeAttrs), GraphEditor::resultIds(*scale),
          m->getOperands(), scale->getResultTypes(),
          m->getAttrOfType<mlir::StringAttr>("backend"));

      scale->replaceAllUsesWith(node);
      view.noteReplaced(scale, node);
      scale->erase();
      for (int64_t id : dead)
        view.dropTensor(id);
      m->erase();
    }

    view.commitTable();
  }
};

// dense_layout_propagation_pass: the dense projections consume the activation
// in the layout its producer already writes (position-major) instead of the
// executor repacking it into fixed-width blocks and copying the result back.
// The rewrite marks every batched `matvec_f32` / `matvec_bf16` node with
// `dense_layout = "strided"`; the CUDA backend reads that attribute at
// execution time (sonic_cuda_backend.cpp, MatvecF32 / MatvecBf16). Prefill
// only: the batched form is the prefill form. Pure data movement — the GEMM
// keeps its fixed 128-position block width, so per-position numerics stay
// independent of batch size and offset (prefix-cache reuse).
class DenseLayoutPropagationPass
    : public mlir::PassWrapper<DenseLayoutPropagationPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseLayoutPropagationPass)

  void runOnOperation() override {
    boom::GraphOp graph = getOperation();
    mlir::OpBuilder builder(graph.getContext());

    for (mlir::Operation &node : graph.getBody().front()) {
      const llvm::StringRef name = node.getName().getStringRef();
      if (name != boomOpName(planner::OpKind::MatvecF32) &&
          name != boomOpName(planner::OpKind::MatvecBf16))
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      if (!attrs || !nodeAttrInt(attrs, "batch"))
        continue; // decode form: the per-row gemv has no packed layout
      bool already = false;
      for (mlir::Attribute entry : attrs) {
        auto attr = mlir::cast<boom::NodeAttr>(entry);
        if (attr.getName() == "dense_layout") already = true;
      }
      if (already) continue;
      llvm::SmallVector<mlir::Attribute> entries(attrs.getValue());
      entries.push_back(boom::NodeAttr::get(
          graph.getContext(), "dense_layout", builder.getStringAttr("strided")));
      node.setAttr("node_attrs", builder.getArrayAttr(entries));
    }
  }
};

// prefill_dispatch_pass (MMQ1): the batched prefill projections run on the q8_1
// tensor-core kernel the compiler generates (MmqTcCodegen), or on the ggml mmq
// kernel they ran on before. The choice is structural on the IR plus one
// device-independent number:
//
//   * eligible — a batched quantized_matmul_q8 (a fused q8_1 quantizer feeds
//     it, so the activation layout is the one the generated kernel takes), with
//     a fixed K that is a multiple of 128 (the mmq activation block and the
//     kernel's 32-value mma chunks) and a non-empty output width;
//   * affordable — the per-weight q8_1 copies the eligible nodes need (K int8
//     values + K/32 f32 scales per output row, i.e. rows*K*9/8 bytes each)
//     fit SONICBOOM_MMQ_TC_MAX_MB. The copies are additional to the K-quant
//     weights that stay resident, so a model whose q8_1 copy alone exceeds the
//     budget keeps the ggml path: Gemma 4 E4B's eligible prefill projections
//     alone measure 3662 MiB (the whole model's quantized tensors would be
//     7.79 GiB), against a 2048 MiB default budget on an 8 GiB device. The
//     runtime builds each copy lazily on first use, so the budget is what the
//     pass may spend, not what it allocates up front.
//
// `SONICBOOM_MMQ_TC=on` bypasses the budget for a machine with headroom to
// verify the path; `=off` is the pre-MMQ1 behavior (and the unset default). A
// node keeps every attribute it had (the runtime dispatches on `slot`/`ds`/
// `batch` exactly as it does for quantized_matmul_q8) — only the operator name
// changes.
//
// Status: experimental and default off (boom-fix1 C5). Output is not
// bit-identical to the ggml mmq path (first divergence at step 19 of 30 on two
// prompts) and the q8_1 weight copies cost 8.5 bit/weight against Q4_K's 4.5.
// The path exists for the fused prologue/epilogue seam the generated kernel
// gives us, not for standalone matmul speed.
class PrefillDispatchPass
    : public mlir::PassWrapper<PrefillDispatchPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefillDispatchPass)

  void runOnOperation() override {
    const MmqTcMode mode = mmq_tc_mode();
    if (mode == MmqTcMode::Off) return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);
    mlir::OpBuilder builder(view.ctx());
    const mlir::Location loc = graph.getLoc();

    llvm::SmallVector<mlir::Operation *> eligible;
    uint64_t copy_bytes = 0;
    for (mlir::Operation &node : view.body()) {
      if (node.getName().getStringRef() !=
              boomOpName(planner::OpKind::QuantizedMatmulQ8) ||
          node.getNumOperands() == 0 || node.getNumResults() == 0)
        continue;
      auto attrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
      auto nb = nodeAttrInt(attrs, "batch");
      auto ds = nodeAttrInt(attrs, "ds");
      if (!nb || *nb <= 0 || !ds || *ds < 0 || *ds > 2)
        continue;  // a decode node, or a weight type with no mmq layout
      const int64_t in_id = view.idOf(node.getOperand(0));
      const int64_t out_id = GraphEditor::resultIds(node)[0];
      if (!view.hasTensor(in_id) || !view.hasTensor(out_id)) continue;
      const auto in_shape = view.desc(in_id).getShape();
      const auto out_shape = view.desc(out_id).getShape();
      if (in_shape.size() != 2 || out_shape.size() != 2) continue;
      const int64_t k = in_shape[1], rows = out_shape[1];
      if (k <= 0 || rows <= 0 || k % 128 != 0) continue;
      eligible.push_back(&node);
      copy_bytes += uint64_t(rows) * (uint64_t(k) + 4 * (uint64_t(k) / 32));
    }
    if (eligible.empty()) return;

    // One line per process: a plan build runs this pass once per segment class,
    // and the decision is the same for every one of them.
    static bool reported = false;
    const uint64_t budget = uint64_t(mmq_tc_max_mb()) * 1024 * 1024;
    if (mode == MmqTcMode::Auto && copy_bytes > budget) {
      if (!reported) {
        std::fprintf(stderr,
                     "[mmq-tc] %zu batched projections need a %.0f MiB q8_1 "
                     "weight copy, over the %d MiB budget: keeping the ggml "
                     "mmq path (SONICBOOM_MMQ_TC=on overrides)\n",
                     eligible.size(), double(copy_bytes) / (1024.0 * 1024.0),
                     mmq_tc_max_mb());
        reported = true;
      }
      return;
    }
    if (!reported) {
      std::fprintf(stderr,
                   "[mmq-tc] %zu batched projections on the generated "
                   "tensor-core kernel (%.0f MiB q8_1 weight copy)\n",
                   eligible.size(), double(copy_bytes) / (1024.0 * 1024.0));
      reported = true;
    }

    for (mlir::Operation *node : eligible) {
      builder.setInsertionPoint(node);
      mlir::Operation *tc = buildNode(
          builder, loc, planner::op_kind_name(planner::OpKind::MatmulQ8Tc),
          GraphEditor::nodeId(*node),
          planner::op_kind_name(planner::OpKind::MatmulQ8Tc),
          node->getAttrOfType<mlir::ArrayAttr>("node_attrs"),
          GraphEditor::resultIds(*node), node->getOperands(),
          node->getResultTypes(),
          node->getAttrOfType<mlir::StringAttr>("backend"));
      node->replaceAllUsesWith(tc);
      view.noteReplaced(node, tc);
      node->erase();
    }
    // Attribute-only: no tensor is added, dropped or resized.
    view.commitTable();
  }
};

//--- VS1: layer-wise async weight prefetch ----------------------------------

// Insert the prefetch markers a graph runs under the async-prefetch switch.
//
// The decision is the model-bound weight footprint: the byte sizes of the
// tensor table's constant tensors (the IR's model-bound weights; they are the
// block arguments no node produces). When that footprint does not exceed
// SONICBOOM_VRAM_THRESHOLD_MB there is nothing to stream and the pass is a
// no-op. `SONICBOOM_ASYNC_PREFETCH=on` is the switch; unset/off leaves the
// graph untouched.
//
// Placement, per distinct `layer` attribute on the nodes (ascending):
//
//   async(layer 0)                       at the graph start
//   before layer L's first node:         sync(L), async(next layer)
//
// so a layer's weights are requested one layer ahead and waited for at that
// layer's boundary. The async marker's operands are the layer's constant
// weight tensors (deduplicated); its `prefetch_bytes` attribute records their
// total. The markers have no results: the runtime executes them as
// prefetch-stream bookkeeping (planner/async_prefetch.h).
//
// A graph with fewer than two layers has no boundary to overlap and is left
// alone. Idempotence: markers already present are skipped, so running the
// phase twice would not stack two prefetch schedules.
class AsyncPrefetchPass
    : public mlir::PassWrapper<AsyncPrefetchPass,
                               mlir::OperationPass<boom::GraphOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncPrefetchPass)

  void runOnOperation() override {
    if (async_prefetch_mode() == AsyncPrefetchMode::Off) return;

    boom::GraphOp graph = getOperation();
    GraphEditor view(graph);

    // The resident model-bound weight footprint.
    uint64_t footprint = 0;
    for (mlir::BlockArgument arg : view.body().getArguments()) {
      if (llvm::isa<boom::KvType>(arg.getType())) continue;  // cache handle
      const auto id = view.idOf(arg);
      if (!view.hasTensor(id)) continue;
      const boom::TensorDescAttr desc = view.desc(id);
      if (desc.getIsConstant()) footprint += desc.getByteSize();
    }
    const uint64_t threshold = uint64_t(vram_threshold_mb()) * 1024 * 1024;
    if (footprint <= threshold) return;

    // The layers, in first-appearance order, and each one's first node.
    llvm::SmallVector<int64_t> layers;
    llvm::DenseMap<int64_t, mlir::Operation *> firstOfLayer;
    for (mlir::Operation &node : view.body()) {
      const llvm::StringRef name = node.getName().getStringRef();
      if (name == "boom.async_weight_prefetch" ||
          name == "boom.sync_prefetch")
        return;  // already scheduled
      auto layer = node.getAttrOfType<mlir::IntegerAttr>("layer");
      if (!layer || layer.getInt() < 0) continue;
      auto [it, inserted] = firstOfLayer.try_emplace(layer.getInt(), &node);
      if (inserted) layers.push_back(layer.getInt());
    }
    if (layers.size() < 2) return;

    struct LayerWeights {
      llvm::SmallVector<mlir::Value> operands;
      uint64_t bytes = 0;
    };
    llvm::DenseMap<int64_t, LayerWeights> weights;
    llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> seen;
    for (mlir::Operation &node : view.body()) {
      auto layer = node.getAttrOfType<mlir::IntegerAttr>("layer");
      if (!layer || layer.getInt() < 0) continue;
      const int64_t l = layer.getInt();
      for (mlir::Value operand : node.getOperands()) {
        if (llvm::isa<boom::KvType>(operand.getType())) continue;
        const auto id = view.idOf(operand);
        if (!view.hasTensor(id) || !view.desc(id).getIsConstant()) continue;
        if (!seen[l].insert(id).second) continue;
        weights[l].operands.push_back(operand);
        weights[l].bytes += view.desc(id).getByteSize();
      }
    }

    const mlir::Location loc = graph.getLoc();
    mlir::OpBuilder builder(view.ctx());
    auto insertMarker = [&](llvm::StringRef opName, int64_t layer,
                            uint64_t bytes,
                            mlir::ValueRange operands) -> mlir::Operation * {
      llvm::SmallVector<mlir::Attribute> attrs;
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "layer", builder.getI64IntegerAttr(layer)));
      attrs.push_back(boom::NodeAttr::get(
          view.ctx(), "prefetch_bytes",
          builder.getI64IntegerAttr(int64_t(bytes))));
      const std::string nodeName =
          (opName + "_" + std::to_string(layer)).str();
      return buildNode(builder, loc, opName, view.nextNodeId(), nodeName,
                       builder.getArrayAttr(attrs), llvm::ArrayRef<int64_t>(),
                       operands, mlir::TypeRange(), nullptr);
    };

    // The first layer's weights are requested before the graph's first node;
    // there is no earlier work to overlap them with.
    const int64_t first = layers.front();
    builder.setInsertionPointToStart(&view.body());
    insertMarker(planner::op_kind_name(planner::OpKind::AsyncWeightPrefetch),
                 first, weights[first].bytes, weights[first].operands);

    for (size_t i = 0; i < layers.size(); ++i) {
      const int64_t l = layers[i];
      builder.setInsertionPoint(firstOfLayer.lookup(l));
      insertMarker(planner::op_kind_name(planner::OpKind::SyncPrefetch), l, 0,
                   mlir::ValueRange());
      if (i + 1 < layers.size()) {
        const int64_t next = layers[i + 1];
        insertMarker(
            planner::op_kind_name(planner::OpKind::AsyncWeightPrefetch), next,
            weights[next].bytes, weights[next].operands);
      }
    }

    // Marker-only: no tensor is added, dropped or resized.
    view.commitTable();
  }
};

//--- the per-pass disable switch --------------------------------------------

// Ablation for one fusion at a time: the passes a run skips, named on the
// command line (SONICBOOM_IR_PASS_DISABLE, documented on planner::IrPassPhase in
// planner/ir_passes.h, and read here, where the passes and their names live).
//
// Skipping a pass leaves the graph as the pass before it would, so a later pass
// sees what it is written to see. That is not the same as the later pass being
// unaffected: the pipeline is a sequence over one graph, and QuantizeOncePass
// inserts the Quantize nodes RmsNormQuantizePass and GeluMulQuantizePass fold
// into their producer, so skipping it also removes those two folds (measured —
// see the plan deltas in the run that landed this). A disabled pass is read as
// "this pass and the folds it feeds", which is what the plan diff records.
struct DisabledPasses {
  bool prefill_dispatch = false;
  bool quantize_once = false;
  bool rms_norm_quantize = false;
  bool gelu_mul_quantize = false;
  bool silu_mul_quantize = false;
  bool bias_into_gemv = false;
  bool residual_add_into_gemv = false;
  bool residual_epilogue = false;
  bool sibling_gemv = false;
  bool rms_norm_mmq = false;
  bool gelu_mul_mmq = false;
  bool silu_mul_mmq = false;
  bool fuse_glue_regions = false;
  bool fuse_add_rms_norm_mmq = false;
  bool fuse_rms_norm_add = false;
  bool fuse_heads_norm_rope_cast = false;
  bool fuse_kv_append_into_prep = false;
  bool fuse_slice_scale = false;
  bool fuse_gelu_mul = false;
  bool fuse_silu_mul = false;
  bool scale_into_dense = false;
  bool fold_scale_into_glue = false;
  bool dense_layout_propagation = false;
  bool buffer_reuse = false;
  bool async_prefetch = false;
};

// The names the switch accepts: the pass class names, one table so the parser
// and the error message cannot disagree about them.
struct PassName {
  const char *name;
  bool DisabledPasses::*flag;
};
constexpr PassName kPassNames[] = {
    {"PrefillDispatchPass", &DisabledPasses::prefill_dispatch},
    {"QuantizeOncePass", &DisabledPasses::quantize_once},
    {"RmsNormQuantizePass", &DisabledPasses::rms_norm_quantize},
    {"GeluMulQuantizePass", &DisabledPasses::gelu_mul_quantize},
    {"SiluMulQuantizePass", &DisabledPasses::silu_mul_quantize},
    {"BiasIntoGemvPass", &DisabledPasses::bias_into_gemv},
    {"ResidualAddIntoGemvPass", &DisabledPasses::residual_add_into_gemv},
    {"FuseResidualEpiloguePass", &DisabledPasses::residual_epilogue},
    {"FuseSiblingGemvPass", &DisabledPasses::sibling_gemv},
    {"FuseGlueRegionsPass", &DisabledPasses::fuse_glue_regions},
    {"FuseAddRmsNormMmqPass", &DisabledPasses::fuse_add_rms_norm_mmq},
    {"FuseRmsNormAddPass", &DisabledPasses::fuse_rms_norm_add},
    {"FuseHeadsNormRopeCastPass", &DisabledPasses::fuse_heads_norm_rope_cast},
    {"FuseKvAppendIntoPrepPass", &DisabledPasses::fuse_kv_append_into_prep},
    {"FuseSliceScalePass", &DisabledPasses::fuse_slice_scale},
    {"FuseGeluMulPass", &DisabledPasses::fuse_gelu_mul},
    {"FuseSiluMulPass", &DisabledPasses::fuse_silu_mul},
    {"ScaleIntoDensePass", &DisabledPasses::scale_into_dense},
    {"FoldScaleIntoGluePass", &DisabledPasses::fold_scale_into_glue},
    {"RmsNormMmqPass", &DisabledPasses::rms_norm_mmq},
    {"GeluMulMmqPass", &DisabledPasses::gelu_mul_mmq},
    {"SiluMulMmqPass", &DisabledPasses::silu_mul_mmq},
    {"DenseLayoutPropagationPass", &DisabledPasses::dense_layout_propagation},
    {"BufferReusePass", &DisabledPasses::buffer_reuse},
    {"AsyncPrefetchPass", &DisabledPasses::async_prefetch},
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

// A phase's passes, in the phase's execution order (docs/boom-ir.md): the port
// of each Graph rewrite in planner/rewrites.cpp, in that pass's
// place in the order and with its preconditions. Both phases run entirely here,
// and the Graph rewrites are gone, so this is the emitter's only rewrite path.
void addPasses(mlir::PassManager &pm, planner::IrPassPhase phase,
               bool skip_kv_append_fusion, bool fuse_silu_mul) {
  DisabledPasses off;
  // A rejected list disables nothing: running the phase with no passes would
  // leave the graph unfused, and an ablation run would then report that as the
  // effect of the pass it meant to skip.
  if (!parseDisabledPasses(off)) off = DisabledPasses{};
  // The caller's explicit skip is a correctness requirement (verify), not an
  // ablation: it applies regardless of the environment list.
  if (skip_kv_append_fusion) off.fuse_kv_append_into_prep = true;
  switch (phase) {
  case planner::IrPassPhase::Verify:
    // The decode pipeline, except the KV append stays on the attention node: the
    // verify graph appends one row per query position, which a prep region that
    // stores a single row at the base position cannot express.
    off.fuse_kv_append_into_prep = true;
    [[fallthrough]];
  case planner::IrPassPhase::Decode:
    if (!off.quantize_once) pm.addPass(std::make_unique<QuantizeOncePass>());
    if (!off.rms_norm_quantize) pm.addPass(std::make_unique<RmsNormQuantizePass>());
    if (!off.gelu_mul_quantize) pm.addPass(std::make_unique<GeluMulQuantizePass>());
    if (!off.silu_mul_quantize) pm.addPass(std::make_unique<SiluMulQuantizePass>());
    // The gated activation that feeds a dense projection (Gemma's PLE gate:
    // gelu_fp16 -> mul -> matvec_f32) is the same chain FuseGeluMulPass folds
    // in prefill. The mmq fold above already took the chains that feed a
    // quantized matmul, so what remains here is the dense form.
    if (!off.fuse_gelu_mul) pm.addPass(std::make_unique<FuseGeluMulPass>());
    if (!off.bias_into_gemv) pm.addPass(std::make_unique<BiasIntoGemvPass>());
    if (!off.residual_add_into_gemv)
      pm.addPass(std::make_unique<ResidualAddIntoGemvPass>());
    // The norm-produced residual adds (the Gemma sandwich norm and the PLE
    // post-norm) take the same epilogue treatment on the norm side.
    if (!off.residual_epilogue)
      pm.addPass(std::make_unique<FuseResidualEpiloguePass>());
    // After the epilogue folds: those decide which projections take a bias or
    // residual. Groups are formed by shared activation, not adjacency, so a
    // projection may be lifted across an unrelated node's position.
    if (!off.sibling_gemv) pm.addPass(std::make_unique<FuseSiblingGemvPass>());
    // Last: it consumes the nodes the folds above produce.
    if (!off.fuse_glue_regions)
      pm.addPass(std::make_unique<FuseGlueRegionsPass>());
    // The attention prep (norm/rope/cast) is independent of the folds above:
    // none of them match those chains.
    if (!off.fuse_heads_norm_rope_cast)
      pm.addPass(std::make_unique<FuseHeadsNormRopeCastPass>());
    // After the prep fusion: this pass marks the K/V prep regions it built and
    // the attention node that consumed them.
    if (!off.fuse_kv_append_into_prep)
      pm.addPass(std::make_unique<FuseKvAppendIntoPrepPass>());
    // Last: buffer assignment reads the final node set's uses and sizes.
    if (!off.buffer_reuse) pm.addPass(std::make_unique<BufferReusePass>());
    // VS1: the prefetch markers annotate the final node set's layer
    // boundaries; no pass before this one reads them and no pass after does.
    if (!off.async_prefetch) pm.addPass(std::make_unique<AsyncPrefetchPass>());
    return;
  case planner::IrPassPhase::Prefill:
    if (!off.rms_norm_mmq) pm.addPass(std::make_unique<RmsNormMmqPass>());
    // After RmsNormMmqPass: that is what creates the batched quantizer this
    // region consumes.
    if (!off.fuse_add_rms_norm_mmq)
      pm.addPass(std::make_unique<FuseAddRmsNormMmqPass>());
    // After the mmq fusion: the norm+add chains it did not take are this
    // region's.
    if (!off.fuse_rms_norm_add)
      pm.addPass(std::make_unique<FuseRmsNormAddPass>());
    // After the region exists: the scalar scale a single-reader chain runs after
    // it is the region's epilogue (attribute-only: the node keeps its place and
    // gets `epilogue_scale`).
    if (!off.fold_scale_into_glue)
      pm.addPass(std::make_unique<FoldScaleIntoGluePass>());
    if (!off.gelu_mul_mmq) pm.addPass(std::make_unique<GeluMulMmqPass>());
    if (!off.silu_mul_mmq) pm.addPass(std::make_unique<SiluMulMmqPass>());
    // The PLE column extraction is an elementwise chain of its own: none of the
    // region folds above match a strided slice, and the pass only reads the
    // nodes it replaces.
    if (!off.fuse_slice_scale) pm.addPass(std::make_unique<FuseSliceScalePass>());
    // After the mmq activation fold: only the gelu->mul chains that feed a
    // dense projection are left.
    if (!off.fuse_gelu_mul) pm.addPass(std::make_unique<FuseGeluMulPass>());
    // The SwiGLU counterpart, same place in the order and the same leftover
    // chains (a silu->mul feeding a dense projection). The caller enables it
    // under the standard numerics profile only; strict keeps the two-kernel
    // form its plan goldens record.
    if (fuse_silu_mul && !off.fuse_silu_mul)
      pm.addPass(std::make_unique<FuseSiluMulPass>());
    // The scalar scale after a dense projection is the projection's epilogue
    // (attribute-only: the node keeps its place and gets `epilogue_scale`).
    if (!off.scale_into_dense) pm.addPass(std::make_unique<ScaleIntoDensePass>());
    // Attention prep: same shape as decode (norm -> rope [-> cast] / norm ->
    // cast), the batched form selected by the norm's `batch` attribute.
    if (!off.fuse_heads_norm_rope_cast)
      pm.addPass(std::make_unique<FuseHeadsNormRopeCastPass>());
    // After the mmq fusions: only the batched quantized_matmul_q8 nodes they
    // produced are candidates for the tensor-core dispatch.
    if (!off.prefill_dispatch)
      pm.addPass(std::make_unique<PrefillDispatchPass>());
    // Last: marks the projections the folds above left, so it sees the final
    // node set (attribute-only, no node replacement).
    if (!off.dense_layout_propagation)
      pm.addPass(std::make_unique<DenseLayoutPropagationPass>());
    // Last: buffer assignment reads the final node set's uses and sizes.
    if (!off.buffer_reuse) pm.addPass(std::make_unique<BufferReusePass>());
    if (!off.async_prefetch) pm.addPass(std::make_unique<AsyncPrefetchPass>());
    return;
  case planner::IrPassPhase::Training:
    // A traced training graph: the forward matmul/norm/activation vocabulary
    // plus the explicit adjoint nodes. None of the inference folds above apply
    // (no quantized weights, no KV cache, no decode/serve split), so the phase
    // runs only what is semantic-neutral here — buffer reuse marks single-use
    // elementwise results the device executor may write in place.
    if (!off.buffer_reuse) pm.addPass(std::make_unique<BufferReusePass>());
    return;
  }
}

const char *phaseName(planner::IrPassPhase phase) {
  switch (phase) {
  case planner::IrPassPhase::Decode:
    return "decode";
  case planner::IrPassPhase::Prefill:
    return "prefill";
  case planner::IrPassPhase::Verify:
    return "verify";
  case planner::IrPassPhase::Training:
    return "training";
  }
  return "?";
}

} // namespace

MmqTcMode mmq_tc_mode() {
  // Read on every call (not cached): the switch is an ablation input, and a
  // caller that changes it between runs must see the new value. Unset means
  // off: the ggml mmq path, which the reference GPU measured faster.
  const char *env = std::getenv("SONICBOOM_MMQ_TC");
  if (env == nullptr || *env == '\0') return MmqTcMode::Off;
  const std::string_view v(env);
  if (v == "on") return MmqTcMode::On;
  if (v == "auto") return MmqTcMode::Auto;
  if (v != "off" && v != "0")
    std::fprintf(stderr,
                 "SONICBOOM_MMQ_TC: '%s' is not on/off/auto; reading it as "
                 "off\n",
                 env);
  return MmqTcMode::Off;
}

int mmq_tc_max_mb() {
  const char *env = std::getenv("SONICBOOM_MMQ_TC_MAX_MB");
  if (env == nullptr || *env == '\0') return 2048;
  const long v = std::strtol(env, nullptr, 10);
  return v > 0 ? int(std::min<long>(v, 1L << 20)) : 2048;
}

AsyncPrefetchMode async_prefetch_mode() {
  // Read on every call, like the other ablation switches: a caller that changes
  // it between runs must see the new value. Unset means off — the skeleton
  // inserts nothing until a run opts in.
  const char *env = std::getenv("SONICBOOM_ASYNC_PREFETCH");
  if (env == nullptr || *env == '\0') return AsyncPrefetchMode::Off;
  const std::string_view v(env);
  if (v == "on") return AsyncPrefetchMode::On;
  if (v != "off" && v != "0")
    std::fprintf(stderr,
                 "SONICBOOM_ASYNC_PREFETCH: '%s' is not on/off; reading it as "
                 "off\n",
                 env);
  return AsyncPrefetchMode::Off;
}

int vram_threshold_mb() {
  const char *env = std::getenv("SONICBOOM_VRAM_THRESHOLD_MB");
  if (env == nullptr || *env == '\0') return 6000;
  const long v = std::strtol(env, nullptr, 10);
  return v > 0 ? int(std::min<long>(v, 1L << 20)) : 6000;
}

// The phase's passes over the IR a caller built (compiler/BoomPasses.h).
bool runBoomPasses(mlir::ModuleOp module, planner::IrPassPhase phase,
                   bool skip_kv_append_fusion, bool fuse_silu_mul) {
  // The unit of meaning in this IR is the graph, not the module that holds it
  // (BoomVerify.cpp:21-24), and that is where the passes run: the graph op is
  // IsolatedFromAbove, which is what makes it a pass anchor at all.
  auto graphs = module.getOps<boom::GraphOp>();
  if (graphs.empty()) {
    std::fprintf(stderr, "runBoomPasses(%s): the module holds no graph\n",
                 phaseName(phase));
    return false;
  }
  boom::GraphOp graphOp = *graphs.begin();

  // The pass manager verifies after every pass, so a pass that leaves the IR in
  // a state the dialect rejects fails here, naming that pass, rather than three
  // passes later.
  mlir::PassManager pm(module.getContext());
  addPasses(pm, phase, skip_kv_append_fusion, fuse_silu_mul);
  if (mlir::failed(pm.run(graphOp))) {
    std::fprintf(stderr,
                 "runBoomPasses(%s): a Boom IR pass failed (diagnostic "
                 "above); the IR is not a phase output\n",
                 phaseName(phase));
    return false;
  }
  return true;
}

} // namespace sonicboom::compiler
