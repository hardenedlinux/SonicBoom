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

// The generated FlashAttention-2 kernel (compiler/AttentionCodegen.h).
//
// The kernel text is emitted directly in the GPU dialect: one gpu.func, a
// gpu.block_id x/y grid (query tile, query head), 16-row sub-tiles mma'd with
// m16n8k16, online softmax in f32 and P rounded to f16 for the PV product.
// Everything is runtime-shaped except (head_dim, gqa, mask kind, Br, Bc):
// sequence length, cache length, window, scale, and the batch/cache split of
// the key sequence are kernel arguments, so one compiled kernel serves every
// launch of its shape class.
//
// The generated kernel's numerics contract: a key's value depends only on its
// absolute position, and key tiles are aligned to multiples of Bc, so a prefix
// reuse is bit-identical to the equivalent full prefill. That is checked by
// test_cuda_attention_codegen.

#include <sonicboom/compiler/AttentionCodegen.h>

#include <llvm/Config/llvm-config.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <sonicboom/compiler/GlueCodegen.h>

namespace sonicboom::compiler {
namespace {

struct Emitter {
  std::string body;
  int next = 0;

  void op(const std::string &s) { body += "      " + s + "\n"; }

  std::string val(const std::string &expr) {
    std::string v = "%v" + std::to_string(++next);
    body += "      " + v + " = " + expr + "\n";
    return v;
  }

  // Two-result op (gpu.shuffle returns value + predicate).
  std::pair<std::string, std::string> val2(const std::string &expr) {
    std::string a = "%v" + std::to_string(next + 1);
    std::string b = "%v" + std::to_string(next + 2);
    next += 2;
    body += "      " + a + ", " + b + " = " + expr + "\n";
    return {a, b};
  }
};

std::string attention_region_text(const AttentionShape &sh, int sm_arch) {
  const int D = sh.head_dim;
  const int GQA = sh.gqa;
  const int BR = sh.br;
  const int BC = sh.bc;
  const int WROW = BR / 16;   // 16-row query chunks per block
  const int DPS = sh.dps;     // warps the head_dim is split across
  const int DC = D / DPS;     // output columns per warp
  const int NW = WROW * DPS;  // warps per block
  const int THREADS = NW * 32;
  const int SUBS = BC / 16;
  const int NT = DC / 8;   // output n-tiles of 8 columns per warp
  const int KT = DC / 16;  // Q/K k-steps of 16 per warp

  Emitter e;
  std::map<std::string, std::string> C;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i32");
  };
  const auto c64 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i64");
  };
  const auto cf = [&](const char *n, const char *v) {
    C[n] = e.val(std::string("arith.constant ") + v + " : f32");
  };
  const auto ci = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : index");
  };

  c32("c0", 0); c32("c1", 1); c32("c2", 2); c32("c3", 3); c32("c4", 4);
  c32("c5", 5); c32("c7", 7); c32("c8", 8); c32("c15", 15); c32("c16", 16);
  c32("c31", 31); c32("c32", 32); c32("c63", 63); c32("c64", 64);
  c32("c128", 128); c32("cbr", BR);
  c64("l0", 0); c64("l1", 1); c64("l8", 8);
  c64("lbc", BC);
  c64("lbr", BR);
  c64("ld", D);
  cf("cf0", "0.0"); cf("cneg", "-1.0e+30"); cf("cone", "1.0");
  ci("x0", 0);
  ci("xD", D);
  ci("xbc", BC);
  ci("xq", BR * D);       // Q staging elements
  ci("xkv", D * BC);      // K/V staging elements
  ci("xthr", THREADS);    // staging stride
  // Dynamic shared memory: Q [BR][D] f16, K^T [D][BC] f16, V [BC][D] f16, and
  // (when the head_dim is split across warps) the f32 partial-S buffer
  // [DPS][BR][BC].
  const int kByteOff = BR * D * 2;
  const int vByteOff = kByteOff + D * BC * 2;
  const int sByteOff = vByteOff + BC * D * 2;
  ci("xkb", kByteOff);
  ci("xvb", vByteOff);
  ci("xsb", sByteOff);
  if (GQA != 4 && GQA != 7) c32("gqa", GQA);

  const std::string FUNC =
      "gpu.func @fa(%q: !llvm.ptr, %kb: !llvm.ptr, %vb: !llvm.ptr, "
      "%kc: !llvm.ptr, %vc: !llvm.ptr, %startp: !llvm.ptr, %nslots: i64, "
      "%nhq: i32, %nhkv: i32, %n: i32, %window: i64, %scale: f32, "
      "%out: !llvm.ptr) kernel {";

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string smem =
      e.val("gpu.dynamic_shared_memory : memref<?xi8, "
            "#gpu.address_space<workgroup>>");
  const std::string qmem = e.val("memref.view " + smem + "[" + C["x0"] +
                                 "][] : memref<?xi8, "
                                 "#gpu.address_space<workgroup>> to memref<" +
                                 s(BR) + "x" + s(D) +
                                 "xf16, #gpu.address_space<workgroup>>");
  const std::string ktmem = e.val("memref.view " + smem + "[" + C["xkb"] +
                                  "][] : memref<?xi8, "
                                  "#gpu.address_space<workgroup>> to memref<" +
                                  s(D) + "x" + s(BC) +
                                  "xf16, #gpu.address_space<workgroup>>");
  const std::string vmem = e.val("memref.view " + smem + "[" + C["xvb"] +
                                 "][] : memref<?xi8, "
                                 "#gpu.address_space<workgroup>> to memref<" +
                                 s(BC) + "x" + s(D) +
                                 "xf16, #gpu.address_space<workgroup>>");
  const std::string smem_s =
      DPS > 1 ? e.val("memref.view " + smem + "[" + C["xsb"] +
                      "][] : memref<?xi8, "
                      "#gpu.address_space<workgroup>> to memref<" +
                      s(DPS) + "x" + s(BR) + "x" + s(BC) +
                      "xf32, #gpu.address_space<workgroup>>")
              : std::string();

  const std::string tid_i =
      e.val("arith.index_cast " + tid + " : index to i32");
  const std::string lane =
      e.val("arith.andi " + tid_i + ", " + C["c31"] + " : i32");
  const std::string warp =
      e.val("arith.shrui " + tid_i + ", " + C["c5"] + " : i32");
  // Warp w owns query rows [rowchunk*16, rowchunk*16+16) and output columns
  // [dchunk*DC, dchunk*DC+DC). With dps = 1 the row chunk is the whole warp
  // role and the D split is trivial.
  if (DPS != 1) c32("cdps", DPS);
  const std::string rowchunk = DPS == 1
      ? warp
      : e.val("arith.divui " + warp + ", " + C["cdps"] + " : i32");
  const std::string dchunk = DPS == 1
      ? C["c0"]
      : e.val("arith.remui " + warp + ", " + C["cdps"] + " : i32");
  const std::string dbase = DPS == 1
      ? C["c0"]
      : e.val("arith.muli " + dchunk + ", " +
              e.val("arith.constant " + s(DC) + " : i32") + " : i32");
  // dbase as an index (smem views / V columns) and as i64 (output offsets).
  const std::string dbase_i = e.val("arith.index_cast " + dbase +
                                    " : i32 to index");
  const std::string dbase64 = e.val("arith.extsi " + dbase + " : i32 to i64");
  const std::string g = e.val("arith.shrui " + lane + ", " + C["c2"] + " : i32");
  const std::string tc = e.val("arith.andi " + lane + ", " + C["c3"] + " : i32");
  const std::string lane15 =
      e.val("arith.andi " + lane + ", " + C["c15"] + " : i32");
  const std::string lsh16 =
      e.val("arith.shrui " + lane + ", " + C["c4"] + " : i32");  // lane>>4
  const std::string bx_i = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string by_i = e.val("arith.index_cast " + by + " : index to i32");
  const std::string bx64 = e.val("arith.extsi " + bx_i + " : i32 to i64");
  const std::string by64 = e.val("arith.extsi " + by_i + " : i32 to i64");
  const std::string n64 = e.val("arith.extsi %n : i32 to i64");
  const std::string nhq64 = e.val("arith.extui %nhq : i32 to i64");
  const std::string nhkv64 = e.val("arith.extui %nhkv : i32 to i64");
  const std::string start = e.val("llvm.load %startp : !llvm.ptr -> i64");
  const std::string divc =
      GQA == 7 ? C["c7"] : GQA == 4 ? C["c4"] : C["gqa"];
  const std::string hkv =
      e.val("arith.divui " + by_i + ", " + divc + " : i32");
  const std::string hkv64 = e.val("arith.extui " + hkv + " : i32 to i64");
  const std::string g64 = e.val("arith.extsi " + g + " : i32 to i64");
  const std::string tc64 = e.val("arith.extsi " + tc + " : i32 to i64");
  // Absolute first query position of this block: bx * Br.
  const std::string bxbr =
      e.val("arith.muli " + bx64 + ", " + C["lbr"] + " : i64");

  // ---- Q tile: [BR][D] f16 ------------------------------------------------
  e.op("scf.for %qi = " + tid + " to " + C["xq"] + " step " + C["xthr"] +
       " : index {");
  {
    const std::string kk = e.val("arith.divui %qi, " + C["xD"] + " : index");
    const std::string dd = e.val("arith.remui %qi, " + C["xD"] + " : index");
    const std::string kk64 = e.val("arith.index_cast " + kk + " : index to i64");
    const std::string dd64 = e.val("arith.index_cast " + dd + " : index to i64");
    const std::string qrow = e.val("arith.addi " + bxbr + ", " + kk64 + " : i64");
    const std::string okq = e.val("arith.cmpi slt, " + qrow + ", " + n64 +
                                  " : i64");
    const std::string qv = e.val("scf.if " + okq + " -> f32 {");
    {
      const std::string off =
          e.val("arith.muli " + qrow + ", " + nhq64 + " : i64");
      const std::string off2 =
          e.val("arith.addi " + off + ", " + by64 + " : i64");
      const std::string off3 =
          e.val("arith.muli " + off2 + ", " + C["ld"] + " : i64");
      const std::string off4 =
          e.val("arith.addi " + off3 + ", " + dd64 + " : i64");
      const std::string qp = e.val("llvm.getelementptr %q[" + off4 +
                                   "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      const std::string qval = e.val("llvm.load " + qp + " : !llvm.ptr -> f32");
      e.op("scf.yield " + qval + " : f32");
    }
    e.op("} else {");
    e.op("scf.yield " + C["cf0"] + " : f32");
    e.op("}");
    const std::string qh = e.val("arith.truncf " + qv + " : f32 to f16");
    e.op("memref.store " + qh + ", " + qmem + "[" + kk + ", " + dd +
         "] : memref<" + s(BR) + "x" + s(D) +
         "xf16, #gpu.address_space<workgroup>>");
  }
  e.op("}");
  e.op("gpu.barrier");

  // ---- Q fragments (invariant across key tiles) ---------------------------
  // Each warp loads its 16 query rows over its DC-column slice of head_dim.
  std::vector<std::string> aq;
  for (int k0 = 0; k0 < DC; k0 += 16) {
    std::string row =
        e.val("arith.muli " + rowchunk + ", " + C["c16"] + " : i32");
    row = e.val("arith.addi " + row + ", " + lane15 + " : i32");
    std::string col = e.val("arith.muli " + lsh16 + ", " + C["c8"] + " : i32");
    if (k0 != 0) {
      const std::string kc = e.val("arith.constant " + s(k0) + " : i32");
      col = e.val("arith.addi " + col + ", " + kc + " : i32");
    }
    if (DPS != 1)
      col = e.val("arith.addi " + col + ", " + dbase + " : i32");
    const std::string ri = e.val("arith.index_cast " + row + " : i32 to index");
    const std::string ci2 = e.val("arith.index_cast " + col + " : i32 to index");
    aq.push_back(e.val("nvgpu.ldmatrix " + qmem + "[" + ri + ", " + ci2 +
                       "] {numTiles = 4 : i32, transpose = false} : memref<" +
                       s(BR) + "x" + s(D) +
                       "xf16, #gpu.address_space<workgroup>> -> "
                       "vector<4x2xf16>"));
  }

  // ---- key-tile loop bounds ----------------------------------------------
  // n_blk = min(BR, n - bx*BR); the last absolute key this block sees is
  // start + bx*BR + n_blk - 1, and the loop's upper bound is exclusive.
  const std::string nrem =
      e.val("arith.subi " + n64 + ", " + bxbr + " : i64");
  const std::string nblk =
      e.val("arith.minsi " + nrem + ", " + C["lbr"] + " : i64");
  const std::string keyend0 =
      e.val("arith.addi " + start + ", " + bxbr + " : i64");
  const std::string keyend1 =
      e.val("arith.addi " + keyend0 + ", " + nblk + " : i64");
  const std::string keyend =
      e.val("arith.subi " + keyend1 + ", " + C["l1"] + " : i64");
  const std::string abs_pb = keyend0;
  std::string j0s = C["l0"];
  if (sh.window_mask) {
    const std::string haswin =
        e.val("arith.cmpi sgt, %window, " + C["l0"] + " : i64");
    const std::string big =
        e.val("arith.cmpi sge, " + abs_pb + ", %window : i64");
    const std::string ws =
        e.val("arith.subi " + abs_pb + ", %window : i64");
    const std::string ws1 = e.val("arith.addi " + ws + ", " + C["l1"] + " : i64");
    const std::string cond =
        e.val("arith.andi " + haswin + ", " + big + " : i1");
    const std::string key_start =
        e.val("arith.select " + cond + ", " + ws1 + ", " + C["l0"] + " : i64");
    j0s = e.val("arith.muli " +
                e.val("arith.divsi " + key_start + ", " + C["lbc"] + " : i64") +
                ", " + C["lbc"] + " : i64");
  }

  // ---- per-row absolute positions ----------------------------------------
  const std::string w16_64 = e.val("arith.constant 16 : i64");
  const std::string rowchunk64 =
      e.val("arith.extsi " + rowchunk + " : i32 to i64");
  const std::string warprow = e.val("arith.muli " + rowchunk64 + ", " + w16_64 +
                                    " : i64");
  const std::string rowbase =
      e.val("arith.addi " + abs_pb + ", " + warprow + " : i64");
  const std::string p0 = e.val("arith.addi " + rowbase + ", " + g64 + " : i64");
  const std::string p1 = e.val("arith.addi " + p0 + ", " + C["l8"] + " : i64");
  const auto low_of = [&](const std::string &p) -> std::string {
    if (!sh.window_mask) return C["l0"];
    const std::string hasw =
        e.val("arith.cmpi sgt, %window, " + C["l0"] + " : i64");
    const std::string big = e.val("arith.cmpi sge, " + p + ", %window : i64");
    const std::string ws = e.val("arith.subi " + p + ", %window : i64");
    const std::string ws1 = e.val("arith.addi " + ws + ", " + C["l1"] + " : i64");
    const std::string cond = e.val("arith.andi " + hasw + ", " + big + " : i1");
    return e.val("arith.select " + cond + ", " + ws1 + ", " + C["l0"] +
                 " : i64");
  };
  const std::string low0 = low_of(p0);
  const std::string low1 = low_of(p1);

  // ---- online-softmax loop ------------------------------------------------
  // iter_args: O[0..NT-1] (2x2 f32 each), then m0, m1, l0, l1.
  std::vector<std::string> names;
  std::vector<std::string> init;
  std::vector<std::string> types;
  std::vector<std::string> o_cur(NT);
  for (int i = 0; i < NT; ++i) {
    o_cur[i] = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    names.push_back("%o" + s(i));
    init.push_back(o_cur[i]);
    types.push_back("vector<2x2xf32>");
  }
  names.push_back("%m0"); init.push_back(C["cneg"]); types.push_back("f32");
  names.push_back("%m1"); init.push_back(C["cneg"]); types.push_back("f32");
  names.push_back("%l0"); init.push_back(C["cf0"]); types.push_back("f32");
  names.push_back("%l1"); init.push_back(C["cf0"]); types.push_back("f32");
  std::string ia = "iter_args(";
  std::string ty = " -> (";
  std::string loopres;
  for (size_t i = 0; i < names.size(); ++i) {
    ia += (i ? ", " : "") + names[i] + " = " + init[i];
    ty += (i ? ", " : "") + types[i];
    loopres += (i ? ", " : "") + std::string("%r") + s(int(i));
  }
  ia += ")";
  ty += ")";
  const std::string j0sx = e.val("arith.index_cast " + j0s + " : i64 to index");
  const std::string kex = e.val("arith.index_cast " + keyend1 +
                                " : i64 to index");
  e.op(loopres + " = scf.for %j0 = " + j0sx + " to " + kex + " step " +
       C["xbc"] + " " + ia + ty + " {");
  const std::string j0v = e.val("arith.index_cast %j0 : index to i64");
  std::map<std::string, std::string> cur;  // iter-arg name -> current value
  for (size_t i = 0; i < names.size(); ++i) cur[names[i]] = names[i];

  // ---- stage K/V for this key tile ---------------------------------------
  for (const char *which : {"k", "v"}) {
    const bool is_k = std::string(which) == "k";
    e.op("scf.for %si = " + tid + " to " + C["xkv"] + " step " + C["xthr"] +
         " : index {");
    const std::string keyoff =
        e.val("arith.divui %si, " + C["xD"] + " : index");
    const std::string dd = e.val("arith.remui %si, " + C["xD"] + " : index");
    const std::string keyoff64 =
        e.val("arith.index_cast " + keyoff + " : index to i64");
    const std::string dd64 =
        e.val("arith.index_cast " + dd + " : index to i64");
    const std::string j = e.val("arith.addi " + j0v + ", " + keyoff64 + " : i64");
    const std::string okj =
        e.val("arith.cmpi sle, " + j + ", " + keyend + " : i64");
    const std::string vv = e.val("scf.if " + okj + " -> f32 {");
    {
      const std::string isc =
          e.val("arith.cmpi slt, " + j + ", " + start + " : i64");
      const std::string slot = e.val("arith.remui " + j + ", %nslots : i64");
      const std::string co1 =
          e.val("arith.muli " + slot + ", " + nhkv64 + " : i64");
      const std::string co2 =
          e.val("arith.addi " + co1 + ", " + hkv64 + " : i64");
      const std::string co3 =
          e.val("arith.muli " + co2 + ", " + C["ld"] + " : i64");
      const std::string co4 =
          e.val("arith.addi " + co3 + ", " + dd64 + " : i64");
      const std::string cp =
          e.val(std::string("llvm.getelementptr %") + (is_k ? "kc" : "vc") +
                "[" + co4 + "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      const std::string bi =
          e.val("arith.subi " + j + ", " + start + " : i64");
      const std::string bo1 =
          e.val("arith.muli " + bi + ", " + nhkv64 + " : i64");
      const std::string bo2 =
          e.val("arith.addi " + bo1 + ", " + hkv64 + " : i64");
      const std::string bo3 =
          e.val("arith.muli " + bo2 + ", " + C["ld"] + " : i64");
      const std::string bo4 =
          e.val("arith.addi " + bo3 + ", " + dd64 + " : i64");
      const std::string bp =
          e.val(std::string("llvm.getelementptr %") + (is_k ? "kb" : "vb") +
                "[" + bo4 + "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      const std::string rp =
          e.val("llvm.select " + isc + ", " + cp + ", " + bp +
                " : i1, !llvm.ptr");
      const std::string vf = e.val("llvm.load " + rp + " : !llvm.ptr -> f32");
      e.op("scf.yield " + vf + " : f32");
    }
    e.op("} else {");
    e.op("scf.yield " + C["cf0"] + " : f32");
    e.op("}");
    const std::string h = e.val("arith.truncf " + vv + " : f32 to f16");
    if (is_k)
      e.op("memref.store " + h + ", " + ktmem + "[" + dd + ", " + keyoff +
           "] : memref<" + s(D) + "x" + s(BC) +
           "xf16, #gpu.address_space<workgroup>>");
    else
      e.op("memref.store " + h + ", " + vmem + "[" + keyoff + ", " + dd +
           "] : memref<" + s(BC) + "x" + s(D) +
           "xf16, #gpu.address_space<workgroup>>");
    e.op("}");
  }
  e.op("gpu.barrier");

  // ---- S = Q K^T for every sub-tile --------------------------------------
  // Each warp accumulates S over its DC-column slice of head_dim. With the
  // head_dim split across warps (DPS > 1) the partials go through shared
  // memory; every warp then sums them in warp order, so the reduction order
  // is fixed and the result does not depend on scheduling (batch invariance).
  // C fragment: [0][0]=row r col 2c, [0][1]=col 2c+1, [1][*]=row r+8.
  const auto ext = [&](const std::string &v, int r, int c) {
    return e.val("vector.extract " + v + "[" + s(r) + ", " + s(c) +
                 "] : f32 from vector<2x2xf32>");
  };
  std::vector<std::string> s0v(SUBS), s1v(SUBS);
  for (int sub = 0; sub < SUBS; ++sub) {
    const int key_off = sub * 16;
    std::string s0 = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    std::string s1 = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    for (int k0i = 0; k0i < KT; ++k0i) {
      const std::string kc =
          e.val("arith.constant " + s(k0i * 16) + " : i32");
      std::string row32 = e.val("arith.addi " + lane15 + ", " + kc + " : i32");
      if (DPS != 1)
        row32 = e.val("arith.addi " + row32 + ", " + dbase + " : i32");
      const std::string ri =
          e.val("arith.index_cast " + row32 + " : i32 to index");
      for (int half = 0; half < 2; ++half) {
        const std::string cidx =
            e.val("arith.constant " + s(key_off + half * 8) + " : index");
        const std::string b = e.val(
            "nvgpu.ldmatrix " + ktmem + "[" + ri + ", " + cidx +
            "] {numTiles = 2 : i32, transpose = true} : memref<" + s(D) + "x" +
            s(BC) +
            "xf16, #gpu.address_space<workgroup>> -> vector<2x2xf16>");
        std::string &acc = half == 0 ? s0 : s1;
        acc = e.val("nvgpu.mma.sync (" + aq[k0i] + ", " + b + ", " + acc +
                    ") {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, "
                    "vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>");
      }
    }
    s0v[sub] = s0;
    s1v[sub] = s1;
  }

  // The shared-S handoff: each warp stores the 16xBC partial of its column
  // slice; the barrier below makes all of them visible before the sums.
  std::string prow0i, prow1i, pdchi, ptc2i, ptc2p1i;
  if (DPS > 1) {
    const std::string rowchunk16 =
        e.val("arith.muli " + rowchunk + ", " + C["c16"] + " : i32");
    const std::string prow0 = e.val("arith.addi " + rowchunk16 + ", " + g +
                                    " : i32");
    const std::string prow1 =
        e.val("arith.addi " + prow0 + ", " + C["c8"] + " : i32");
    const std::string tc2 =
        e.val("arith.muli " + tc + ", " + C["c2"] + " : i32");
    const std::string tc2p1 =
        e.val("arith.addi " + tc2 + ", " + C["c1"] + " : i32");
    const auto idx = [&](const std::string &v) {
      return e.val("arith.index_cast " + v + " : i32 to index");
    };
    prow0i = idx(prow0);
    prow1i = idx(prow1);
    pdchi = idx(dchunk);
    ptc2i = idx(tc2);
    ptc2p1i = idx(tc2p1);
    const std::string stype = " : memref<" + s(DPS) + "x" + s(BR) + "x" +
                              s(BC) + "xf32, #gpu.address_space<workgroup>>";
    for (int sub = 0; sub < SUBS; ++sub) {
      const int key_off = sub * 16;
      for (int half = 0; half < 2; ++half) {
        const std::string hbase = e.val("arith.constant " +
                                        s(key_off + half * 8) + " : index");
        const std::string c0i =
            e.val("arith.addi " + hbase + ", " + ptc2i + " : index");
        const std::string c1i =
            e.val("arith.addi " + hbase + ", " + ptc2p1i + " : index");
        const std::string &vv = half == 0 ? s0v[sub] : s1v[sub];
        e.op("memref.store " + ext(vv, 0, 0) + ", " + smem_s + "[" + pdchi +
             ", " + prow0i + ", " + c0i + "]" + stype);
        e.op("memref.store " + ext(vv, 0, 1) + ", " + smem_s + "[" + pdchi +
             ", " + prow0i + ", " + c1i + "]" + stype);
        e.op("memref.store " + ext(vv, 1, 0) + ", " + smem_s + "[" + pdchi +
             ", " + prow1i + ", " + c0i + "]" + stype);
        e.op("memref.store " + ext(vv, 1, 1) + ", " + smem_s + "[" + pdchi +
             ", " + prow1i + ", " + c1i + "]" + stype);
      }
    }
    e.op("gpu.barrier");
  }

  // ---- per 16-key sub-tile: mask + online softmax + PV --------------------
  for (int sub = 0; sub < SUBS; ++sub) {
    const int key_off = sub * 16;

    // se: n-tile 0 (keys key_off+0..7); sf: n-tile 1 (keys key_off+8..15)
    std::string se00, se01, se10, se11, sf00, sf01, sf10, sf11;
    if (DPS == 1) {
      se00 = ext(s0v[sub], 0, 0);
      se01 = ext(s0v[sub], 0, 1);
      se10 = ext(s0v[sub], 1, 0);
      se11 = ext(s0v[sub], 1, 1);
      sf00 = ext(s1v[sub], 0, 0);
      sf01 = ext(s1v[sub], 0, 1);
      sf10 = ext(s1v[sub], 1, 0);
      sf11 = ext(s1v[sub], 1, 1);
    } else {
      const std::string stype =
          " : memref<" + s(DPS) + "x" + s(BR) + "x" + s(BC) +
          "xf32, #gpu.address_space<workgroup>>";
      const auto sum_d = [&](const std::string &ri,
                             const std::string &ci) -> std::string {
        std::string acc;
        for (int d = 0; d < DPS; ++d) {
          const std::string di =
              e.val("arith.constant " + s(d) + " : index");
          const std::string v = e.val("memref.load " + smem_s + "[" + di +
                                      ", " + ri + ", " + ci + "]" + stype);
          acc = d == 0 ? v : e.val("arith.addf " + acc + ", " + v + " : f32");
        }
        return acc;
      };
      const std::string ce = e.val("arith.constant " + s(key_off) + " : index");
      const std::string cf = e.val("arith.constant " + s(key_off + 8) +
                                   " : index");
      const std::string ce0 = e.val("arith.addi " + ce + ", " + ptc2i +
                                    " : index");
      const std::string ce1 = e.val("arith.addi " + ce + ", " + ptc2p1i +
                                    " : index");
      const std::string cf0 = e.val("arith.addi " + cf + ", " + ptc2i +
                                    " : index");
      const std::string cf1 = e.val("arith.addi " + cf + ", " + ptc2p1i +
                                    " : index");
      se00 = sum_d(prow0i, ce0);
      se01 = sum_d(prow0i, ce1);
      se10 = sum_d(prow1i, ce0);
      se11 = sum_d(prow1i, ce1);
      sf00 = sum_d(prow0i, cf0);
      sf01 = sum_d(prow0i, cf1);
      sf10 = sum_d(prow1i, cf0);
      sf11 = sum_d(prow1i, cf1);
    }

    // Absolute key of each of this lane's four columns (row r: two keys from
    // each n-tile; row r+8 shares them).
    const std::string kbase = e.val("arith.addi " + j0v + ", " +
                                    e.val("arith.constant " + s(key_off) +
                                          " : i64") +
                                    " : i64");
    const std::string k2 = e.val("arith.constant 2 : i64");
    const std::string k0a = e.val("arith.addi " + kbase + ", " +
                                  e.val("arith.muli " + tc64 + ", " + k2 +
                                        " : i64") +
                                  " : i64");
    const std::string k0b = e.val("arith.addi " + k0a + ", " + C["l1"] + " : i64");
    const std::string k1a = e.val("arith.addi " + k0a + ", " + C["l8"] + " : i64");
    const std::string k1b = e.val("arith.addi " + k0b + ", " + C["l8"] + " : i64");

    const auto masked = [&](const std::string &sv, const std::string &key,
                            const std::string &p,
                            const std::string &low) -> std::string {
      const std::string le = e.val("arith.cmpi sle, " + key + ", " + p + " : i64");
      const std::string ge = e.val("arith.cmpi sge, " + key + ", " + low +
                                   " : i64");
      const std::string ok = e.val("arith.andi " + le + ", " + ge + " : i1");
      const std::string sc = e.val("arith.mulf " + sv + ", %scale : f32");
      return e.val("arith.select " + ok + ", " + sc + ", " + C["cneg"] +
                   " : f32");
    };
    const std::string m0a = masked(se00, k0a, p0, low0);
    const std::string m0b = masked(se01, k0b, p0, low0);
    const std::string m0c = masked(sf00, k1a, p0, low0);
    const std::string m0d = masked(sf01, k1b, p0, low0);
    const std::string m1a = masked(se10, k0a, p1, low1);
    const std::string m1b = masked(se11, k0b, p1, low1);
    const std::string m1c = masked(sf10, k1a, p1, low1);
    const std::string m1d = masked(sf11, k1b, p1, low1);

    // Row reductions over the four lanes that share a row. Width 32 with XOR
    // 1/2 keeps the shuffle closed over each 4-lane group (width 4 lowers to
    // a 0xF member mask, which only lanes 0-3 honour).
    const auto rmax4 = [&](const std::string &x) {
      const std::string a =
          e.val2("gpu.shuffle xor " + x + ", " + C["c1"] + ", " + C["c32"] +
                 " : f32").first;
      const std::string b = e.val("arith.maximumf " + x + ", " + a + " : f32");
      const std::string c =
          e.val2("gpu.shuffle xor " + b + ", " + C["c2"] + ", " + C["c32"] +
                 " : f32").first;
      return e.val("arith.maximumf " + b + ", " + c + " : f32");
    };
    const auto rsum4 = [&](const std::string &x) {
      const std::string a =
          e.val2("gpu.shuffle xor " + x + ", " + C["c1"] + ", " + C["c32"] +
                 " : f32").first;
      const std::string b = e.val("arith.addf " + x + ", " + a + " : f32");
      const std::string c =
          e.val2("gpu.shuffle xor " + b + ", " + C["c2"] + ", " + C["c32"] +
                 " : f32").first;
      return e.val("arith.addf " + b + ", " + c + " : f32");
    };
    const auto maxf = [&](const std::string &a, const std::string &b) {
      return e.val("arith.maximumf " + a + ", " + b + " : f32");
    };
    const std::string mx0 =
        rmax4(maxf(maxf(m0a, m0b), maxf(m0c, m0d)));
    const std::string mx1 =
        rmax4(maxf(maxf(m1a, m1b), maxf(m1c, m1d)));
    const std::string mn0 = maxf(cur["%m0"], mx0);
    const std::string mn1 = maxf(cur["%m1"], mx1);
    const std::string rs0 =
        e.val("math.exp " + e.val("arith.subf " + cur["%m0"] + ", " + mn0 +
                                  " : f32") + " : f32");
    const std::string rs1 =
        e.val("math.exp " + e.val("arith.subf " + cur["%m1"] + ", " + mn1 +
                                  " : f32") + " : f32");
    const auto prob = [&](const std::string &sv, const std::string &mn) {
      return e.val("math.exp " + e.val("arith.subf " + sv + ", " + mn +
                                       " : f32") + " : f32");
    };
    const std::string p0a = prob(m0a, mn0), p0b = prob(m0b, mn0);
    const std::string p0c = prob(m0c, mn0), p0d = prob(m0d, mn0);
    const std::string p1a = prob(m1a, mn1), p1b = prob(m1b, mn1);
    const std::string p1c = prob(m1c, mn1), p1d = prob(m1d, mn1);
    const auto addf = [&](const std::string &a, const std::string &b) {
      return e.val("arith.addf " + a + ", " + b + " : f32");
    };
    const std::string sm0 = rsum4(addf(addf(p0a, p0b), addf(p0c, p0d)));
    const std::string sm1 = rsum4(addf(addf(p1a, p1b), addf(p1c, p1d)));
    const std::string nl0 =
        e.val("arith.addf " +
              e.val("arith.mulf " + cur["%l0"] + ", " + rs0 + " : f32") + ", " +
              sm0 + " : f32");
    const std::string nl1 =
        e.val("arith.addf " +
              e.val("arith.mulf " + cur["%l1"] + ", " + rs1 + " : f32") + ", " +
              sm1 + " : f32");

    // P fragment: A operand of O += P V (rows r / r+8, k = 16 keys).
    std::string pv3 = e.val("arith.constant dense<0.0> : vector<4x2xf16>");
    const std::string ph[8] = {p0a, p0b, p1a, p1b, p0c, p0d, p1c, p1d};
    for (int idx = 0; idx < 8; ++idx) {
      const std::string h = e.val("arith.truncf " + ph[idx] + " : f32 to f16");
      pv3 = e.val("vector.insert " + h + ", " + pv3 + "[" + s(idx / 2) + ", " +
                  s(idx % 2) + "] : f16 into vector<4x2xf16>");
    }
    // O rescale: row r by rs0, row r+8 by rs1.
    std::string rsv = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    const std::string rsel[4] = {rs0, rs0, rs1, rs1};
    for (int idx = 0; idx < 4; ++idx) {
      rsv = e.val("vector.insert " + rsel[idx] + ", " + rsv + "[" + s(idx / 2) +
                  ", " + s(idx % 2) + "] : f32 into vector<2x2xf32>");
    }
    std::string vri = e.val("arith.index_cast " + lane15 + " : i32 to index");
    if (key_off) {
      vri = e.val("arith.addi " + vri + ", " +
                  e.val("arith.constant " + s(key_off) + " : index") +
                  " : index");
    }
    for (int nt = 0; nt < NT; ++nt) {
      const std::string o =
          e.val("arith.mulf " + cur["%o" + s(nt)] + ", " + rsv +
                " : vector<2x2xf32>");
      std::string ccol = e.val("arith.constant " + s(nt * 8) + " : index");
      if (DPS != 1)
        ccol = e.val("arith.addi " + ccol + ", " + dbase_i + " : index");
      const std::string bv = e.val(
          "nvgpu.ldmatrix " + vmem + "[" + vri + ", " + ccol +
          "] {numTiles = 2 : i32, transpose = true} : memref<" + s(BC) + "x" +
          s(D) +
          "xf16, #gpu.address_space<workgroup>> -> vector<2x2xf16>");
      cur["%o" + s(nt)] =
          e.val("nvgpu.mma.sync (" + pv3 + ", " + bv + ", " + o +
                ") {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, "
                "vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>");
    }
    cur["%m0"] = mn0;
    cur["%m1"] = mn1;
    cur["%l0"] = nl0;
    cur["%l1"] = nl1;
  }
  e.op("gpu.barrier");
  {
    std::string y = "scf.yield ";
    std::string t;
    for (size_t i = 0; i < names.size(); ++i) {
      y += (i ? ", " : "") + cur[names[i]];
      t += (i ? ", " : "") + types[i];
    }
    e.op(y + " : " + t);
  }
  e.op("}");
  for (size_t i = 0; i < names.size(); ++i) cur[names[i]] = "%r" + s(int(i));

  // ---- finalize: O / l, guarded by the row's position in n -----------------
  const std::string inv0 =
      e.val("arith.divf " + C["cone"] + ", " + cur["%l0"] + " : f32");
  const std::string inv1 =
      e.val("arith.divf " + C["cone"] + ", " + cur["%l1"] + " : f32");
  std::string iq0 = e.val("arith.addi " +
                          e.val("arith.muli " + bx_i + ", " + C["cbr"] +
                                " : i32") +
                          ", " +
                          e.val("arith.muli " + rowchunk + ", " + C["c16"] +
                                " : i32") +
                          " : i32");
  iq0 = e.val("arith.addi " + iq0 + ", " + g + " : i32");
  const std::string iq1 = e.val("arith.addi " + iq0 + ", " + C["c8"] + " : i32");
  for (int row = 0; row < 2; ++row) {
    const std::string &iq = row == 0 ? iq0 : iq1;
    const std::string &inv = row == 0 ? inv0 : inv1;
    const std::string okr = e.val("arith.cmpi slt, " + iq + ", %n : i32");
    e.op("scf.if " + okr + " {");
    const std::string iq64 = e.val("arith.extsi " + iq + " : i32 to i64");
    const std::string ro1 =
        e.val("arith.muli " + iq64 + ", " + nhq64 + " : i64");
    const std::string ro2 = e.val("arith.addi " + ro1 + ", " + by64 + " : i64");
    const std::string ro3 =
        e.val("arith.muli " + ro2 + ", " + C["ld"] + " : i64");
    for (int nt = 0; nt < NT; ++nt) {
      const std::string &o = cur["%o" + s(nt)];
      const std::string e0 =
          e.val("vector.extract " + o + "[" + s(row) +
                ", 0] : f32 from vector<2x2xf32>");
      const std::string e1 =
          e.val("vector.extract " + o + "[" + s(row) +
                ", 1] : f32 from vector<2x2xf32>");
      const std::string y0 = e.val("arith.mulf " + e0 + ", " + inv + " : f32");
      const std::string y1 = e.val("arith.mulf " + e1 + ", " + inv + " : f32");
      std::string off = e.val("arith.addi " + ro3 + ", " +
                              e.val("arith.constant " + s(nt * 8) + " : i64") +
                              " : i64");
      off = e.val("arith.addi " + off + ", " +
                  e.val("arith.muli " + tc64 + ", " + e.val("arith.constant 2 : i64") +
                        " : i64") +
                  " : i64");
      if (DPS != 1)
        off = e.val("arith.addi " + off + ", " + dbase64 + " : i64");
      const std::string pout = e.val("llvm.getelementptr %out[" + off +
                                     "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      e.op("llvm.store " + y0 + ", " + pout + " : f32, !llvm.ptr");
      const std::string p1p = e.val(
          "llvm.getelementptr %out[" + e.val("arith.addi " + off + ", " +
                                             C["l1"] + " : i64") +
          "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
      e.op("llvm.store " + y1 + ", " + p1p + " : f32, !llvm.ptr");
    }
    e.op("}");
  }
  e.op("gpu.return");
  e.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @fa [#nvvm.target<chip = \"sm_" + s(sm_arch) + "\">] {\n";
  text += "      " + FUNC + "\n";
  text += e.body;
  text += "  }\n}\n";
  return text;
}

// Flash-decoding text: `fa_decode` (one warp per (head, chunk) partial) and
// `fa_combine` (one block per head, ascending-chunk reduction). The split is
// over key chunks of a runtime length, aligned to absolute positions: chunk s
// covers keys [s*C, (s+1)*C), clamped to the window's valid lower bound and to
// the current position. Each lane owns head_dim/32 output columns and keeps
// them in registers; the q·k dot is reduced across the warp with XOR shuffles,
// so the reduction order depends only on head_dim. The combine reads chunks in
// ascending order, so the output does not depend on scheduling.
std::string attention_decode_region_text(const AttentionShape &sh, int sm_arch) {
  const int D = sh.head_dim;
  const int GQA = sh.gqa;
  const int K = D / 32;  // output columns per lane

  Emitter e;
  std::map<std::string, std::string> C;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i32");
  };
  const auto c64 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i64");
  };
  const auto cf = [&](const char *n, const char *v) {
    C[n] = e.val(std::string("arith.constant ") + v + " : f32");
  };
  const auto ci = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : index");
  };

  e.op("gpu.func @fa_decode(%q: !llvm.ptr, %kc: !llvm.ptr, %vc: !llvm.ptr, "
       "%startp: !llvm.ptr, %nslots: i64, %nhq: i32, %nhkv: i32, "
       "%window: i64, %scale: f32, %part: !llvm.ptr, %chunk: i64) kernel {");
  c32("c0", 0); c32("c1", 1); c32("c4", 4); c32("c7", 7); c32("c16", 16);
  c32("c31", 31); c32("c32", 32);
  c64("l0", 0); c64("l1", 1); c64("l2", 2); c64("ldp", D + 2);
  c64("lD", D);
  cf("cf0", "0.0"); cf("cneg", "-1.0e+30"); cf("cone", "1.0");
  ci("x0", 0); ci("x1", 1);
  if (GQA != 4 && GQA != 7) c32("gqa", GQA);

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string tid_i = e.val("arith.index_cast " + tid + " : index to i32");
  const std::string bx_i = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string by_i = e.val("arith.index_cast " + by + " : index to i32");
  const std::string lane =
      e.val("arith.andi " + tid_i + ", " + C["c31"] + " : i32");
  const std::string lane64 = e.val("arith.extsi " + lane + " : i32 to i64");
  const std::string bx64 = e.val("arith.extsi " + bx_i + " : i32 to i64");
  const std::string by64 = e.val("arith.extsi " + by_i + " : i32 to i64");
  const std::string nhq64 = e.val("arith.extui %nhq : i32 to i64");
  const std::string nhkv64 = e.val("arith.extui %nhkv : i32 to i64");
  const std::string start = e.val("llvm.load %startp : !llvm.ptr -> i64");
  const std::string div =
      GQA == 7 ? C["c7"] : GQA == 4 ? C["c4"] : C["gqa"];
  const std::string hkv = e.val("arith.divui " + by_i + ", " + div + " : i32");
  const std::string hkv64 = e.val("arith.extui " + hkv + " : i32 to i64");

  // Valid key window: [low, start] intersect the block's chunk.
  std::string low = C["l0"];
  if (sh.window_mask) {
    const std::string hasw = e.val("arith.cmpi sgt, %window, " + C["l0"] +
                                   " : i64");
    const std::string big = e.val("arith.cmpi sge, " + start + ", %window : i64");
    const std::string ws = e.val("arith.subi " + start + ", %window : i64");
    const std::string ws1 = e.val("arith.addi " + ws + ", " + C["l1"] + " : i64");
    const std::string cond = e.val("arith.andi " + hasw + ", " + big + " : i1");
    low = e.val("arith.select " + cond + ", " + ws1 + ", " + C["l0"] + " : i64");
  }
  // Chunks are aligned to the window's first key (`low`), not to absolute
  // position 0: the grid is sized for ceil(window / chunk) chunks, and they
  // must cover any window of that size ending at the current position.
  const std::string cbo = e.val("arith.muli " + bx64 + ", %chunk : i64");
  const std::string clo = e.val("arith.addi " + low + ", " + cbo + " : i64");
  const std::string chi = e.val("arith.addi " + clo + ", %chunk : i64");
  const std::string endp1 = e.val("arith.addi " + start + ", " + C["l1"] +
                                  " : i64");
  const std::string jlo = e.val("arith.maxsi " + clo + ", " + low + " : i64");
  const std::string jhi = e.val("arith.minsi " + chi + ", " + endp1 + " : i64");
  const std::string jlo_i = e.val("arith.index_cast " + jlo + " : i64 to index");
  const std::string jhi_i = e.val("arith.index_cast " + jhi + " : i64 to index");

  // q: [n_heads_q][head_dim] f32, one token.
  const std::string qbase = e.val("arith.muli " + by64 + ", " + C["lD"] +
                                  " : i64");
  std::vector<std::string> qv(K);
  for (int i = 0; i < K; ++i) {
    const std::string oi = e.val("arith.constant " + s(i * 32) + " : i64");
    const std::string off = e.val("arith.addi " + qbase + ", " + lane64 +
                                  " : i64");
    const std::string offo = e.val("arith.addi " + off + ", " + oi + " : i64");
    const std::string p = e.val("llvm.getelementptr %q[" + offo +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
    qv[i] = e.val("llvm.load " + p + " : !llvm.ptr -> f32");
  }

  // The chunk's scores go through a fixed shared-memory view: the block is one
  // warp, and the P·V pass needs every key's score in every lane. Two passes
  // (scores + max, then exp + accumulate) keep the computation identical to
  // the hand-written kernel's chunking — no online rescaling of O — and cost
  // a store plus a broadcast load per key.
  const std::string smem =
      e.val("gpu.dynamic_shared_memory : memref<?xi8, "
            "#gpu.address_space<workgroup>>");
  const std::string smemf = e.val(
      "memref.view " + smem + "[" + C["x0"] +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<" +
      s(kAttentionDecodeMaxChunk) + "xf32, #gpu.address_space<workgroup>>");
  const std::string smtype = " : memref<" + s(kAttentionDecodeMaxChunk) +
                             "xf32, #gpu.address_space<workgroup>>";
  const std::string is0 = e.val("arith.cmpi eq, " + tid_i + ", " + C["c0"] +
                                " : i32");

  // The first key's cache slot costs one 64-bit modulo; every later key's is
  // the predecessor plus one with a compare-and-reset (device division is a
  // software routine and dominated this loop before).
  const std::string slot0 = e.val("arith.remui " + jlo + ", %nslots : i64");

  // Pass 1: one q·k per key, stored by lane 0.
  e.op("scf.for %j = " + jlo_i + " to " + jhi_i + " step " + C["x1"] +
       " iter_args(%slot = " + slot0 + ") -> (i64) {");
  {
    const std::string so1 =
        e.val("arith.muli %slot, " + nhkv64 + " : i64");
    const std::string so2 = e.val("arith.addi " + so1 + ", " + hkv64 + " : i64");
    const std::string so3 = e.val("arith.muli " + so2 + ", " + C["lD"] +
                                  " : i64");
    const std::string base =
        e.val("arith.addi " + so3 + ", " + lane64 + " : i64");
    const auto at = [&](const std::string &ptr, int i) {
      const std::string oi = e.val("arith.constant " + s(i * 32) + " : i64");
      const std::string off = i == 0 ? base
                                     : e.val("arith.addi " + base + ", " + oi +
                                             " : i64");
      return e.val("llvm.getelementptr %" + ptr + "[" + off +
                   "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
    };
    std::string acc;
    for (int i = 0; i < K; ++i) {
      const std::string k =
          e.val("llvm.load " + at("kc", i) + " : !llvm.ptr -> f32");
      const std::string t = e.val("arith.mulf " + qv[i] + ", " + k + " : f32");
      acc = i == 0 ? t : e.val("arith.addf " + acc + ", " + t + " : f32");
    }
    // Full-warp XOR reduction: every lane ends with the dot product.
    for (int d = 1; d < 32; d *= 2) {
      const std::string c = e.val("arith.constant " + s(d) + " : i32");
      const std::string sh =
          e.val2("gpu.shuffle xor " + acc + ", " + c + ", " + C["c32"] +
                 " : f32").first;
      acc = e.val("arith.addf " + acc + ", " + sh + " : f32");
    }
    const std::string sc = e.val("arith.mulf " + acc + ", %scale : f32");
    const std::string sid = e.val("arith.subi %j, " + jlo_i + " : index");
    e.op("scf.if " + is0 + " {");
    e.op("memref.store " + sc + ", " + smemf + "[" + sid + "]" + smtype);
    e.op("}");
    const std::string sp1 = e.val("arith.addi %slot, " + C["l1"] + " : i64");
    const std::string full =
        e.val("arith.cmpi eq, " + sp1 + ", %nslots : i64");
    const std::string next =
        e.val("arith.select " + full + ", " + C["l0"] + ", " + sp1 + " : i64");
    e.op("scf.yield " + next + " : i64");
  }
  e.op("}");
  e.op("gpu.barrier");

  // Pass 2: chunk maximum, lane-strided local max then an exact warp reduce.
  const std::string lane_idx =
      e.val("arith.index_cast " + lane + " : i32 to index");
  const std::string jstart = e.val("arith.addi " + jlo_i + ", " + lane_idx +
                                   " : index");
  std::map<std::string, std::string> cur;
  {
    const std::string m0 = e.val("arith.constant 32 : index");
    const std::string mr = e.val(
        "scf.for %j = " + jstart + " to " + jhi_i + " step " + m0 +
        " iter_args(%mm = " + C["cneg"] + ") -> (f32) {");
    const std::string sid = e.val("arith.subi %j, " + jlo_i + " : index");
    const std::string sv = e.val("memref.load " + smemf + "[" + sid + "]" +
                                 smtype);
    const std::string nm = e.val("arith.maximumf %mm, " + sv + " : f32");
    e.op("scf.yield " + nm + " : f32");
    e.op("}");
    std::string mx = mr;
    for (int d = 1; d < 32; d *= 2) {
      const std::string c = e.val("arith.constant " + s(d) + " : i32");
      const std::string sh =
          e.val2("gpu.shuffle xor " + mx + ", " + c + ", " + C["c32"] +
                 " : f32").first;
      mx = e.val("arith.maximumf " + mx + ", " + sh + " : f32");
    }
    // Pass 3: exp + accumulate. Every lane iterates the whole chunk, so each
    // lane's l is the full sum (identical across lanes; lane 0 stores it).
    std::vector<std::string> names;
    std::vector<std::string> init;
    std::vector<std::string> types;
    for (int i = 0; i < K; ++i) {
      names.push_back("%o" + s(i));
      init.push_back(C["cf0"]);
      types.push_back("f32");
    }
    names.push_back("%l"); init.push_back(C["cf0"]); types.push_back("f32");
    names.push_back("%slot"); init.push_back(slot0); types.push_back("i64");
    std::string ia = "iter_args(";
    std::string ty = " -> (";
    std::string loopres;
    for (size_t i = 0; i < names.size(); ++i) {
      ia += (i ? ", " : "") + names[i] + " = " + init[i];
      ty += (i ? ", " : "") + types[i];
      loopres += (i ? ", " : "") + std::string("%r") + s(int(i));
    }
    ia += ")";
    ty += ")";
    e.op(loopres + " = scf.for %j = " + jlo_i + " to " + jhi_i + " step " +
         C["x1"] + " " + ia + ty + " {");
    std::map<std::string, std::string> iv;
    for (size_t i = 0; i < names.size(); ++i) iv[names[i]] = names[i];

    const std::string so1 =
        e.val("arith.muli %slot, " + nhkv64 + " : i64");
    const std::string so2 = e.val("arith.addi " + so1 + ", " + hkv64 + " : i64");
    const std::string so3 = e.val("arith.muli " + so2 + ", " + C["lD"] +
                                  " : i64");
    const std::string base =
        e.val("arith.addi " + so3 + ", " + lane64 + " : i64");
    const auto at = [&](const std::string &ptr, int i) {
      const std::string oi = e.val("arith.constant " + s(i * 32) + " : i64");
      const std::string off = i == 0 ? base
                                     : e.val("arith.addi " + base + ", " + oi +
                                             " : i64");
      return e.val("llvm.getelementptr %" + ptr + "[" + off +
                   "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
    };
    const std::string psid = e.val("arith.subi %j, " + jlo_i + " : index");
    const std::string psv = e.val("memref.load " + smemf + "[" + psid + "]" +
                                  smtype);
    const std::string p =
        e.val("math.exp " + e.val("arith.subf " + psv + ", " + mx + " : f32") +
              " : f32");
    // l: each lane sums its own keys and the warp tree-reduces (same reduction
    // shape as the hand-written kernel), rather than every lane serially
    // summing the whole chunk.
    const std::string jj = e.val("arith.subi %j, " + jlo_i + " : index");
    const std::string idx31 = e.val("arith.constant 31 : index");
    const std::string jj32 = e.val("arith.andi " + jj + ", " + idx31 +
                                   " : index");
    const std::string mine =
        e.val("arith.cmpi eq, " + jj32 + ", " + lane_idx + " : index");
    const std::string nlr = e.val("scf.if " + mine + " -> (f32) {");
    e.op("scf.yield " +
         e.val("arith.addf " + iv["%l"] + ", " + p + " : f32") + " : f32");
    e.op("} else {");
    e.op("scf.yield " + iv["%l"] + " : f32");
    e.op("}");
    iv["%l"] = nlr;
    for (int i = 0; i < K; ++i) {
      const std::string v =
          e.val("llvm.load " + at("vc", i) + " : !llvm.ptr -> f32");
      iv["%o" + s(i)] =
          e.val("arith.addf " + iv["%o" + s(i)] + ", " +
                e.val("arith.mulf " + p + ", " + v + " : f32") + " : f32");
    }
    {
      const std::string sp1 = e.val("arith.addi %slot, " + C["l1"] + " : i64");
      const std::string full =
          e.val("arith.cmpi eq, " + sp1 + ", %nslots : i64");
      iv["%slot"] =
          e.val("arith.select " + full + ", " + C["l0"] + ", " + sp1 + " : i64");
    }
    {
      std::string y = "scf.yield ";
      std::string t;
      for (size_t i = 0; i < names.size(); ++i) {
        y += (i ? ", " : "") + iv[names[i]];
        t += (i ? ", " : "") + types[i];
      }
      e.op(y + " : " + t);
    }
    e.op("}");
    std::map<std::string, std::string> out;
    for (size_t i = 0; i < names.size(); ++i)
      out[names[i]] = "%r" + s(int(i));
    out["%m"] = mx;
    // Warp tree sum of the per-lane l partials; lane 0 ends with the total,
    // which is the only copy stored (same reduction as the own kernel).
    std::string lt = out["%l"];
    for (int d = 1; d < 32; d *= 2) {
      const std::string c = e.val("arith.constant " + s(d) + " : i32");
      const std::string sh =
          e.val2("gpu.shuffle down " + lt + ", " + c + ", " + C["c32"] +
                 " : f32").first;
      lt = e.val("arith.addf " + lt + ", " + sh + " : f32");
    }
    out["%l"] = lt;
    // Partial store below uses `cur`.
    cur = out;
  }

  // Partial slot: (chunk, head) -> [head_dim + 2] f32; O stays unnormalized.
  const std::string pid1 = e.val("arith.muli " + bx64 + ", " + nhq64 + " : i64");
  const std::string pid2 = e.val("arith.addi " + pid1 + ", " + by64 + " : i64");
  const std::string pid = e.val("arith.muli " + pid2 + ", " + C["ldp"] + " : i64");
  for (int i = 0; i < K; ++i) {
    const std::string oi = e.val("arith.constant " + s(i * 32) + " : i64");
    const std::string t = e.val("arith.addi " + pid + ", " + oi + " : i64");
    const std::string off = e.val("arith.addi " + t + ", " + lane64 + " : i64");
    const std::string pp = e.val("llvm.getelementptr %part[" + off +
                                 "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
    e.op("llvm.store " + cur["%o" + s(i)] + ", " + pp + " : f32, !llvm.ptr");
  }
  e.op("scf.if " + is0 + " {");
  const std::string moff = e.val("arith.addi " + pid + ", " + C["lD"] + " : i64");
  const std::string mp = e.val("llvm.getelementptr %part[" + moff +
                               "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  e.op("llvm.store " + cur["%m"] + ", " + mp + " : f32, !llvm.ptr");
  const std::string loff = e.val("arith.addi " + moff + ", " + C["l1"] + " : i64");
  const std::string lp = e.val("llvm.getelementptr %part[" + loff +
                               "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  e.op("llvm.store " + cur["%l"] + ", " + lp + " : f32, !llvm.ptr");
  e.op("}");
  e.op("gpu.return");
  e.op("}");

  // ---- combine: one block per head, head_dim threads ----------------------
  Emitter f;
  std::map<std::string, std::string> C2;
  const auto d32 = [&](const char *n, int v) {
    C2[n] = f.val("arith.constant " + s(v) + " : i32");
  };
  const auto d64 = [&](const char *n, int v) {
    C2[n] = f.val("arith.constant " + s(v) + " : i64");
  };
  const auto df = [&](const char *n, const char *v) {
    C2[n] = f.val(std::string("arith.constant ") + v + " : f32");
  };
  const auto di = [&](const char *n, int v) {
    C2[n] = f.val("arith.constant " + s(v) + " : index");
  };
  f.op("gpu.func @fa_combine(%part: !llvm.ptr, %out: !llvm.ptr, "
       "%nhq: i32, %nchunks: i32) kernel {");
  d32("c0", 0); d32("c1", 1); d32("c16", 16); d32("c31", 31); d32("c32", 32);
  d64("l0", 0); d64("l1", 1); d64("ldp", D + 2); d64("lD", D);
  df("cf0", "0.0"); df("cneg", "-1.0e+30"); df("cone", "1.0");
  di("x0", 0); di("x1", 1);
  const std::string ct = f.val("gpu.thread_id x");
  const std::string bh = f.val("gpu.block_id x");
  const std::string ct_i = f.val("arith.index_cast " + ct + " : index to i32");
  const std::string bh_i = f.val("arith.index_cast " + bh + " : index to i32");
  const std::string ct64 = f.val("arith.extsi " + ct_i + " : i32 to i64");
  const std::string bh64 = f.val("arith.extsi " + bh_i + " : i32 to i64");
  const std::string cnq = f.val("arith.extui %nhq : i32 to i64");
  const std::string sn = f.val("arith.index_cast %nchunks : i32 to index");
  // Pass 1: chunk maximum over chunks whose l > 0.
  const std::string mr =
      f.val("scf.for %s = " + C2["x0"] + " to " + sn + " step " + C2["x1"] +
            " iter_args(%mm = " + C2["cneg"] + ") -> (f32) {");
  const std::string s64 = f.val("arith.index_cast %s : index to i64");
  const std::string q1 = f.val("arith.muli " + s64 + ", " + cnq + " : i64");
  const std::string q2 = f.val("arith.addi " + q1 + ", " + bh64 + " : i64");
  const std::string q3 = f.val("arith.muli " + q2 + ", " + C2["ldp"] + " : i64");
  // Partial layout: O at [0, D), m at D, l at D+1. Chunks with l == 0 are
  // empty (beyond the key window) and must not contribute -1e30 to the max.
  const std::string lm = f.val("arith.addi " + q3 + ", " + C2["lD"] + " : i64");
  const std::string lmi = f.val("arith.addi " + lm + ", " + C2["l1"] + " : i64");
  const std::string lmp = f.val("llvm.getelementptr %part[" + lmi +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  const std::string lv = f.val("llvm.load " + lmp + " : !llvm.ptr -> f32");
  const std::string lok = f.val("arith.cmpf ogt, " + lv + ", " + C2["cf0"] +
                                " : f32");
  const std::string lmp2 = f.val("llvm.getelementptr %part[" + lm +
                                 "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  const std::string mv = f.val("llvm.load " + lmp2 + " : !llvm.ptr -> f32");
  const std::string mv2 = f.val("arith.select " + lok + ", " + mv + ", " +
                                C2["cneg"] + " : f32");
  const std::string nm = f.val("arith.maximumf %mm, " + mv2 + " : f32");
  f.op("scf.yield " + nm + " : f32");
  f.op("}");
  const std::string mmax = mr;
  // Pass 2: ascending-chunk weighted sum.
  f.op("%ca, %cw = scf.for %s = " + C2["x0"] + " to " + sn + " step " +
       C2["x1"] + " iter_args(%a = " + C2["cf0"] + ", %w = " + C2["cf0"] +
       ") -> (f32, f32) {");
  const std::string t64 = f.val("arith.index_cast %s : index to i64");
  const std::string w1 = f.val("arith.muli " + t64 + ", " + cnq + " : i64");
  const std::string w2 = f.val("arith.addi " + w1 + ", " + bh64 + " : i64");
  const std::string w3 = f.val("arith.muli " + w2 + ", " + C2["ldp"] + " : i64");
  const std::string wl = f.val("arith.addi " + w3 + ", " + C2["lD"] + " : i64");
  const std::string wl1 = f.val("arith.addi " + wl + ", " + C2["l1"] + " : i64");
  const std::string wlp = f.val("llvm.getelementptr %part[" + wl1 +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  const std::string lv2 = f.val("llvm.load " + wlp + " : !llvm.ptr -> f32");
  const std::string wok = f.val("arith.cmpf ogt, " + lv2 + ", " + C2["cf0"] +
                                " : f32");
  const std::string wmp = f.val("llvm.getelementptr %part[" + wl +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  const std::string mv3 = f.val("llvm.load " + wmp + " : !llvm.ptr -> f32");
  const std::string ex0 = f.val("math.exp " + f.val("arith.subf " + mv3 + ", " +
                                                   mmax + " : f32") + " : f32");
  // O is stored unnormalized, so the numerator rescales it by exp(m - mmax)
  // only; the chunk's l contributes to the denominator weight alone.
  const std::string ex =
      f.val("arith.select " + wok + ", " + ex0 + ", " + C2["cf0"] + " : f32");
  const std::string wv = f.val("arith.mulf " + ex + ", " + lv2 + " : f32");
  const std::string oo = f.val("arith.addi " + w3 + ", " + ct64 + " : i64");
  const std::string op = f.val("llvm.getelementptr %part[" + oo +
                               "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  const std::string ov = f.val("llvm.load " + op + " : !llvm.ptr -> f32");
  const std::string na = f.val("arith.addf %a, " +
                               f.val("arith.mulf " + ex + ", " + ov + " : f32") +
                               " : f32");
  const std::string nw = f.val("arith.addf %w, " + wv + " : f32");
  f.op("scf.yield " + na + ", " + nw + " : f32, f32");
  f.op("}");
  const std::string cacc = "%ca";
  const std::string csumw = "%cw";
  const std::string inv = f.val("arith.divf " + C2["cone"] + ", " + csumw +
                                " : f32");
  const std::string den = f.val("arith.cmpf ogt, " + csumw + ", " + C2["cf0"] +
                                " : f32");
  const std::string yv = f.val("arith.select " + den + ", " +
                               f.val("arith.mulf " + cacc + ", " + inv +
                                     " : f32") + ", " + C2["cf0"] + " : f32");
  const std::string ob = f.val("arith.muli " + bh64 + ", " + C2["lD"] + " : i64");
  const std::string oe = f.val("arith.addi " + ob + ", " + ct64 + " : i64");
  const std::string op2 = f.val("llvm.getelementptr %out[" + oe +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
  f.op("llvm.store " + yv + ", " + op2 + " : f32, !llvm.ptr");
  f.op("gpu.return");
  f.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @fa [#nvvm.target<chip = \"sm_" + s(sm_arch) + "\">] {\n";
  text += e.body;
  text += f.body;
  text += "  }\n}\n";
  return text;
}

} // namespace

bool attention_shape_supported(const AttentionShape &shape) {
  // 16-row sub-tiles and 8/16-wide mma tiles fix the granularity. Each warp
  // owns head_dim/dps output columns — 8/16-wide fragments, so that slice, not
  // head_dim, is what bounds the register budget (128 columns = 16 f32
  // fragments per thread; wider needs a different accumulation). The staging
  // is Q + K + V plus the f32 partial-S buffer when the head_dim is split.
  if (shape.head_dim <= 0 || shape.gqa <= 0) return false;
  if (shape.decode) {
    // One warp per head/chunk: each lane holds head_dim/32 output floats.
    return shape.head_dim % 32 == 0 && shape.head_dim / 32 <= 16 &&
           shape.head_dim <= 1024;
  }
  if (shape.br < 16 || shape.br > 64 || shape.br % 16 != 0) return false;
  if (shape.bc < 16 || shape.bc > 64 || shape.bc % 16 != 0) return false;
  const int dps = shape.dps;
  if (dps != 1 && dps != 2 && dps != 4 && dps != 8) return false;
  if (shape.head_dim % dps != 0) return false;
  const int dc = shape.head_dim / dps;
  if (dc < 16 || dc % 16 != 0 || dc / 8 > 16) return false;
  if ((shape.br / 16) * dps > 32) return false;
  // Up to 96 KiB of dynamic shared memory; the loader raises the per-function
  // limit above 48 KiB.
  return attention_shared_bytes(shape) <= 96 * 1024;
}

int attention_block_threads(const AttentionShape &shape) {
  if (shape.decode) return 32;  // one warp for the partial kernel
  return (shape.br / 16) * shape.dps * 32;
}

int attention_shared_bytes(const AttentionShape &shape) {
  // Decode: the fixed score view the partial kernel stages the chunk in.
  if (shape.decode) return 4 * kAttentionDecodeMaxChunk;
  const int q = 2 * shape.br * shape.head_dim;
  const int k = 2 * shape.head_dim * shape.bc;
  const int v = 2 * shape.bc * shape.head_dim;
  // Partial-S exchange buffer (one f32 per row x key per d-slice).
  const int sp = shape.dps > 1 ? 4 * shape.dps * shape.br * shape.bc : 0;
  return q + k + v + sp;
}

std::optional<std::string> generate_attention_ptx(const AttentionShape &shape,
                                                  int sm_arch) {
  if (!attention_shape_supported(shape)) return std::nullopt;
  const std::string text = shape.decode
      ? attention_decode_region_text(shape, sm_arch)
      : attention_region_text(shape, sm_arch);
  // The softmax exp is libdevice's __nv_expf (math.exp -> nvvm).
  return lower_gpu_module_to_ptx(text, /*link_libdevice=*/true, "attention");
}

std::string attention_codegen_version() {
  // attn3: flash-decoding split-KV kernels (fa_decode + fa_combine).
  return std::string("llvm-") + LLVM_VERSION_STRING + "-attn3";
}

} // namespace sonicboom::compiler
