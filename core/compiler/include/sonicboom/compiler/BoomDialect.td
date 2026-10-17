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

//===----------------------------------------------------------------------===//
// Boom dialect — Boom IR, SonicBoom's compiler IR.
//
// This is step (a) of the MLIR migration (design/mlir-migration.md §6–§7):
// the dialect and its types, with a bidirectional OpKind correspondence added
// in the next step. Nothing here is on the runtime path — the runtime keeps
// executing the planner::Graph; the dialect is the compiler's IR.
//
// Shape of the design (see the doc for the reasoning):
//
//   * One op per planner::OpKind, mnemonic == op_kind_name(OpKind). That makes
//     the correspondence mechanical in both directions instead of a mapping
//     table that can drift.
//   * Quantized formats are *types*, not emitter-baked integers (§6.2), so
//     quant/dtype consistency is a verifier check instead of a private
//     predicate inside a rewrite function.
//   * The KV cache is a memory-effect resource (§6.3), not a side channel:
//     attention reads it, kv_append writes it, and MLIR's effect machinery can
//     therefore order them.
//
// Every op shares one attribute record (the "node record"): the graph-node
// identity fields (node_name/node_id/result_ids/backend) plus the Blade
// attribute vocabulary, typed. Per-op *legality* (which of those attributes an
// op may carry) is a graph-verifier check — ODS Optional attrs cannot express
// "not allowed on this op", and a per-op table can.
//===----------------------------------------------------------------------===//

#ifndef SONICBOOM_COMPILER_BOOMDIALECT_TD
#define SONICBOOM_COMPILER_BOOMDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Boom_Dialect : Dialect {
  let name = "boom";

  let cppNamespace = "::sonicboom::compiler::boom";

  let summary = "SonicBoom compiler IR (Boom IR)";

  let description = [{
    The compiler IR of SonicBoom. It is the target of the lowering from Blade
    (the user-facing S-expression interchange format) and the IR the planner
    passes are being ported onto, one pass per commit
    (design/mlir-migration.md §7).

    Text form of a graph:

    ```mlir
    module {
      boom.graph @decode(tensors = [#boom.tensor_desc<...>, ...],
                         inputs = array<i64: 0, 1>, outputs = array<i64: 900>) {
      ^bb0(%arg0: !boom.tensor<[1], "int64">, %arg1: !boom.tensor<[1], "int64">):
        %0 = boom.embedding(%arg0) {node_id = 0 : i64, result_ids = array<i64: 2>,
                                    slot = 0 : i64, node_name = "embedding"}
              : (!boom.tensor<[1], "int64">) -> !boom.tensor<[1, 2560], "float32">
        ...
      }
    }
    ```

    A tensor value's type must agree with its `#boom.tensor_desc` entry in the
    graph's tensor table; that agreement is the dialect's shape/dtype
    consistency check. The tensor table (not the SSA types) is authoritative
    for the fields the types cannot carry — byte size and the
    constant/external/graph-input/graph-output flags.

    The body block's arguments are the graph's free variables: every tensor the
    graph references but does not produce — its feedable inputs, its
    model-bound constants (a K-quant weight, say) and its externals — in
    ascending tensor-id order. `inputs` names the subset a caller feeds.
  }];

  // Types print through the generated dispatch (`!boom.<mnemonic><...>`).
  let useDefaultTypePrinterParser = 1;

  // Attributes print through the generated dispatch (`#boom.<mnemonic><...>`).
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

// The per-layer KV cache, as a memory-effect resource. The cache is not an SSA
// value in Boom IR: it is device-resident state keyed by layer, and §6.3's
// "the ring semantics and slot count are attributes of the cache value" is a
// later step. Modelling it as an effect resource is what the current IR needs:
//
//   * boom.attention      reads the prefix, then appends the current row;
//   * boom.flash_attention reads the prefix only (the batched append is
//                         deferred to boom.kv_append);
//   * boom.kv_append      writes only.
//
// Between them these give MLIR the ordering it needs. Today the runtime gets
// the same ordering for free from "the executor walks the nodes in order"; in
// the dialect it is stated, so a later pass can reason about it.
def Boom_KvCache
    : Resource<"::sonicboom::compiler::boom::KvCacheResource">;

def Boom_KvRead  : MemoryEffects<[MemRead<Boom_KvCache>]>;
def Boom_KvWrite : MemoryEffects<[MemWrite<Boom_KvCache>]>;
def Boom_KvReadWrite
    : MemoryEffects<[MemRead<Boom_KvCache>, MemWrite<Boom_KvCache>]>;

//===----------------------------------------------------------------------===//
// Node-parameter attributes
//===----------------------------------------------------------------------===//

// One sx::Attribute, carried losslessly. `GraphNodeDesc::attributes` is a
// `std::vector<sx::Attribute>` whose *order* is part of the model identity
// (model_fingerprint hashes the vector in order), so the vector is carried as
// an ordered ArrayAttr of these rather than as a name-keyed dictionary.
//
// The typed per-op attributes below are the view the verifiers and future
// passes read; this is the view the OpKind round trip reproduces exactly.
def Boom_Node : AttrDef<Boom_Dialect, "Node"> {
  let mnemonic = "attr";

  let summary = "one Blade node attribute (name + value)";

  let parameters = (ins
    StringRefParameter<"attribute name">:$name,
    AttrParameter<"::mlir::Attribute", "attribute value">:$value);

  let assemblyFormat = "`<` $name `,` $value `>`";
}

// One planner::TensorDesc. Carries every field of the runtime descriptor; the
// SSA side is the graph body's block arguments (every tensor the graph does not
// produce — its inputs, constants and externals) and op results (whose ids are
// the op's `result_ids`). Without constants there would be no SSA value for a
// model-bound weight, and a node reading one could not be expressed.
def Boom_TensorDesc : AttrDef<Boom_Dialect, "TensorDesc"> {
  let mnemonic = "tensor_desc";

  let summary = "one planner::TensorDesc entry of the graph's tensor table";

  let parameters = (ins
    "int64_t":$id,
    ArrayRefParameter<"int64_t", "dimensions">:$shape,
    StringRefParameter<"dtype name (sx::DType)">:$dtype,
    "uint64_t":$byte_size,
    "uint64_t":$dynamic_base_bytes,
    "bool":$is_input,
    "bool":$is_output,
    "bool":$is_constant,
    "bool":$is_external,
    StringRefParameter<"tensor name">:$name);

  let assemblyFormat = [{
    `<` $id `,` custom<DimList>($shape) `,` $dtype `,` $byte_size
        `,` $dynamic_base_bytes `,` $is_input `,` $is_output `,` $is_constant
        `,` $is_external `,` $name `>`
  }];
}

#endif // SONICBOOM_COMPILER_BOOMDIALECT_TD
