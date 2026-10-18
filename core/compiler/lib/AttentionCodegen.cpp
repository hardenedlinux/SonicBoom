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
  const int NW = BR / 16;
  const int THREADS = NW * 32;
  const int SUBS = BC / 16;
  const int NT = D / 8;   // output n-tiles of 8 columns
  const int KT = D / 16;  // Q/K k-steps of 16

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
  c32("c128", 128);
  c64("l0", 0); c64("l1", 1); c64("l8", 8); c64("l64", 64);
  cf("cf0", "0.0"); cf("cneg", "-1.0e+30"); cf("cone", "1.0");
  ci("x0", 0);
  ci("xD", D);
  ci("xbc", BC);
  ci("x64", 64);
  ci("xq", BR * D);       // Q staging elements
  ci("xkv", D * BC);      // K/V staging elements
  ci("xthr", THREADS);    // staging stride
  ci("xkb", BR * D * 2);  // Kt byte offset: after the Q stage
  ci("xvb", BR * D * 4);  // V byte offset: after the transposed K stage
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

  const std::string tid_i =
      e.val("arith.index_cast " + tid + " : index to i32");
  const std::string lane =
      e.val("arith.andi " + tid_i + ", " + C["c31"] + " : i32");
  const std::string warp =
      e.val("arith.shrui " + tid_i + ", " + C["c5"] + " : i32");
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
  const std::string warp64 = e.val("arith.extsi " + warp + " : i32 to i64");
  const std::string g64 = e.val("arith.extsi " + g + " : i32 to i64");
  const std::string tc64 = e.val("arith.extsi " + tc + " : i32 to i64");
  // Absolute first query position of this block: bx * Br.
  const std::string bxbr =
      e.val("arith.muli " + bx64 + ", " + C["l64"] + " : i64");

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
          e.val("arith.muli " + off2 + ", " + C["l64"] + " : i64");
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
  std::vector<std::string> aq;
  for (int k0 = 0; k0 < D; k0 += 16) {
    std::string row = e.val("arith.muli " + warp + ", " + C["c16"] + " : i32");
    row = e.val("arith.addi " + row + ", " + lane15 + " : i32");
    std::string col = e.val("arith.muli " + lsh16 + ", " + C["c8"] + " : i32");
    if (k0 != 0) {
      const std::string kc = e.val("arith.constant " + s(k0) + " : i32");
      col = e.val("arith.addi " + col + ", " + kc + " : i32");
    }
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
      e.val("arith.minsi " + nrem + ", " + C["l64"] + " : i64");
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
                e.val("arith.divsi " + key_start + ", " + C["l64"] + " : i64") +
                ", " + C["l64"] + " : i64");
  }

  // ---- per-row absolute positions ----------------------------------------
  const std::string w16_64 = e.val("arith.constant 16 : i64");
  const std::string warprow = e.val("arith.muli " + warp64 + ", " + w16_64 +
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
       C["x64"] + " " + ia + ty + " {");
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
          e.val("arith.muli " + co2 + ", " + C["l64"] + " : i64");
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
          e.val("arith.muli " + bo2 + ", " + C["l64"] + " : i64");
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

  // ---- per 16-key sub-tile ------------------------------------------------
  for (int sub = 0; sub < SUBS; ++sub) {
    const int key_off = sub * 16;

    // S = Q K^T, two 8-key n-tiles, k-steps of 16.
    std::string s0 = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    std::string s1 = e.val("arith.constant dense<0.0> : vector<2x2xf32>");
    for (int k0i = 0; k0i < KT; ++k0i) {
      const std::string kc =
          e.val("arith.constant " + s(k0i * 16) + " : i32");
      const std::string row32 = e.val("arith.addi " + lane15 + ", " + kc +
                                      " : i32");
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
    // C fragment: [0][0]=row r col 2c, [0][1]=col 2c+1, [1][*]=row r+8.
    const auto ext = [&](const std::string &v, int r, int c) {
      return e.val("vector.extract " + v + "[" + s(r) + ", " + s(c) +
                   "] : f32 from vector<2x2xf32>");
    };
    // se: n-tile 0 (keys key_off+0..7); sf: n-tile 1 (keys key_off+8..15)
    const std::string se00 = ext(s0, 0, 0), se01 = ext(s0, 0, 1);
    const std::string se10 = ext(s0, 1, 0), se11 = ext(s0, 1, 1);
    const std::string sf00 = ext(s1, 0, 0), sf01 = ext(s1, 0, 1);
    const std::string sf10 = ext(s1, 1, 0), sf11 = ext(s1, 1, 1);

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
      const std::string ccol = e.val("arith.constant " + s(nt * 8) + " : index");
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
                          e.val("arith.muli " + bx_i + ", " + C["c64"] +
                                " : i32") +
                          ", " +
                          e.val("arith.muli " + warp + ", " + C["c16"] +
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
        e.val("arith.muli " + ro2 + ", " + C["l64"] + " : i64");
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

} // namespace

bool attention_shape_supported(const AttentionShape &shape) {
  // 16-row sub-tiles and 8/16-wide mma tiles fix the granularity; head_dim/8
  // output fragments bound the registers, which keeps the generated path to
  // narrow heads (64 is the shipped shape; the generator is D-generic up to
  // that budget). The K/V staging is 2*head_dim*(br+bc) bytes.
  return shape.head_dim > 0 && shape.head_dim % 16 == 0 &&
         shape.head_dim / 8 <= 16 && shape.gqa > 0 && shape.br % 16 == 0 &&
         shape.bc % 16 == 0 && shape.br <= 64 && shape.bc <= 64 &&
         attention_shared_bytes(shape) <= 48 * 1024;
}

int attention_block_threads(const AttentionShape &shape) {
  return (shape.br / 16) * 32;
}

int attention_shared_bytes(const AttentionShape &shape) {
  return 2 * shape.br * shape.head_dim + 2 * shape.head_dim * shape.bc +
         2 * shape.bc * shape.head_dim;
}

std::optional<std::string> generate_attention_ptx(const AttentionShape &shape,
                                                  int sm_arch) {
  if (!attention_shape_supported(shape)) return std::nullopt;
  const std::string text = attention_region_text(shape, sm_arch);
  // The softmax exp is libdevice's __nv_expf (math.exp -> nvvm).
  return lower_gpu_module_to_ptx(text, /*link_libdevice=*/true, "attention");
}

std::string attention_codegen_version() {
  return std::string("llvm-") + LLVM_VERSION_STRING + "-attn1";
}

} // namespace sonicboom::compiler
