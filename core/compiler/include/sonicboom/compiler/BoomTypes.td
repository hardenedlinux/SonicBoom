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
// Boom IR value types (design/mlir-migration.md §6.2).
//
//   !boom.tensor<[4096, 2048], "float32">   a plain tensor value
//   !boom.q8_1<[256, 2048]>                 an activation in the q8_1 activation
//                                           layout the mmq path consumes
//   !boom.q4_k<[4096, 2048]>                a weight in the Q4_K format
//   !boom.q5_k<...> !boom.q6_k<...> !boom.q5_0<...> !boom.q8_0<...
//
// `!boom.tensor` is the type of every SSA value whose element type is one of
// the frozen ten sx::DType names. The quantized types exist because a quantized
// format is a property of the *value's layout*, and the planner's type system
// has no way to say that: `sx::DType` is the frozen ten-name set, so a q8_1
// activation buffer is a Float32 tensor to the runtime and the format is
// recoverable only from which node produced it.
//
// Two consequences worth being explicit about:
//
//   * The shapes of a quant type are the *logical* element shape (what the
//     format expands to), not the serialized byte count. Block geometry
//     follows from the format (QuantTypeInfo in
//     quant/quantized_tensor.h): the fastest-moving dimension is
//     quantized in whole blocks, so it must be a multiple of the block size.
//     That is what verify() enforces — it is the same precondition
//     QuantizedTensor documents, stated once in the IR instead of once per
//     kernel.
//   * A `!boom.q8_1` value's *storage* in the planning layer is a Float32
//     buffer that is a superset of the q8_1 bytes (see quantize_once_pass in
//     planner/rewrites.cpp: the q8_1 blocks are packed into the front of an
//     f32-sized buffer, so the executor can treat it as an ordinary
//     intermediate). The graph verifier therefore accepts a tensor-table entry
//     of dtype float32 for a q8_1-typed value, and requires its shape to match.
//
// Today only `!boom.q8_1` types an SSA value: it is the activation the mmq path
// consumes, so `boom.rms_norm_quantize` & co. produce it and
// `boom.quantized_matmul_q8` consumes it. The weight formats are declared and
// verified here because they are part of the type system (§6.2) and because
// typing them is what will let `ds` — the mmq activation layout an op's weight
// format selects, today an emitter-baked integer read by
// fuse_rms_norm_mmq_pass (rewrites.cpp:362-377) — become a derived property
// (§8.5). No op produces one yet.
//===----------------------------------------------------------------------===//

#ifndef SONICBOOM_COMPILER_BOOMTYPES_TD
#define SONICBOOM_COMPILER_BOOMTYPES_TD

include "BoomDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base for every Boom type. The mnemonic is the textual name after `!boom.`.
class Boom_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Boom_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//--- tensor -----------------------------------------------------------------

// A plain (non-quantized) tensor value. `shape` is the ranked static shape;
// `dtype` is one of the ten names dtype_name() returns for sx::DType
// (sx/ir.h:39-50) and is checked by verify().
def Boom_TensorType : Boom_Type<"Tensor", "tensor"> {
  let summary = "ranked tensor of a plain element type";

  let description = [{
    A ranked static tensor, mirroring planner::TensorDesc's shape and
    sx::DType. Example: `!boom.tensor<[1, 2560], "float32">`.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "static dimensions (ranked)">:$shape,
    StringRefParameter<"element dtype name (sx::DType)">:$dtype);

  let assemblyFormat = "`<` custom<DimList>($shape) `,` $dtype `>`";

  let genVerifyDecl = 1;
}

//--- quantized formats ------------------------------------------------------

// Shared shape of every quantized type: one logical element shape, no separate
// serialized-size parameter (the format determines the block geometry).
class Boom_QuantType<string name, string typeMnemonic, string formatName,
                     int blockSize>
    : Boom_Type<name, typeMnemonic> {
  let parameters = (ins
    ArrayRefParameter<"int64_t", "logical element dimensions (block-contiguous dim first)">:$shape);

  let assemblyFormat = "`<` custom<DimList>($shape) `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The quantized format this type denotes, as spelled by
    /// quant::QuantTypeInfo::name (quant/quantized_tensor.h).
    static ::llvm::StringRef formatName() { return "}] # formatName # [{"; }

    /// Elements per block in the format's serialized layout. The
    /// block-contiguous (fastest-moving, i.e. last) dimension of the shape
    /// must be a multiple of this.
    static int blockSize() { return }] # blockSize # [{; }
  }];
}

// The K-quants: 256-element super-blocks. Listed in quant::QuantType's order;
// the set is that table's, and q3_K is the format the primary model
// (gemma-4-E4B-it-Q3_K_M) stores its weights in.
def Boom_Q3KType : Boom_QuantType<"Q3K", "q3_k", "q3_K", 256>;
def Boom_Q4KType : Boom_QuantType<"Q4K", "q4_k", "q4_K", 256>;
def Boom_Q5KType : Boom_QuantType<"Q5K", "q5_k", "q5_K", 256>;
def Boom_Q6KType : Boom_QuantType<"Q6K", "q6_k", "q6_K", 256>;
// The legacy 32-element formats, which is what every Qwen2.5 Q4_K_M projection
// except ffn_down uses.
def Boom_Q5_0Type : Boom_QuantType<"Q5_0", "q5_0", "q5_0", 32>;
def Boom_Q8_0Type : Boom_QuantType<"Q8_0", "q8_0", "q8_0", 32>;

// The activation layout the mmq path consumes. Not a weight format: it is what
// the fused norm→quantize emitters produce (rms_norm_quantize & co.) and what
// the mmq matmul reads. It has two block geometries in the runtime — the
// decode-side activation quantizer packs 32-element blocks
// (quantize_q8_1_dev) while the mmq activation plane is 128 values wide
// (block_q8_1_mmq, mmq_q8_planes) — and 32 divides 128, so requiring the
// block-contiguous dimension to be a multiple of 32 is the constraint common to
// both.
def Boom_Q8_1Type : Boom_QuantType<"Q8_1", "q8_1", "q8_1", 32> {
  let summary = "activation in the q8_1 activation layout";
}

//--- value-type constraint --------------------------------------------------

// The types an SSA value in Boom IR may have. Every op's operands and results
// are constrained by this, so the generated verifyInvariants rejects a value
// of any other type without any hand-written check.
def Boom_ValueType : AnyTypeOf<[
  Boom_TensorType,
  Boom_Q8_1Type,
  Boom_Q3KType,
  Boom_Q4KType,
  Boom_Q5KType,
  Boom_Q6KType,
  Boom_Q5_0Type,
  Boom_Q8_0Type
]>;

#endif // SONICBOOM_COMPILER_BOOMTYPES_TD
