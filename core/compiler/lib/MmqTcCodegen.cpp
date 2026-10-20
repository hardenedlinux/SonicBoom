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

// The generated q8_1 x q8_1 int8 tensor-core prefill matmul
// (compiler/MmqTcCodegen.h).
//
// The kernel is emitted in the GPU dialect like the dense GEMM: one gpu.func,
// a grid of (token tile, weight-row tile) blocks, every block staging one
// 32-K chunk of both operands into shared memory and running m16n8k32 int8
// mma over it. The int32 accumulator of each mma is converted to f32 and
// scaled by the activation's per-32 scale and the weight's per-32 scale before
// it is added to the lane's accumulator, so a block reads each chunk's scales
// once and the K loop is the only reduction.
//
// Fragment mapping (lane group = lane >> 2, tid = lane & 3), the
// m16n8k32.s8 layout:
//   A (16 tokens x 32 K)   a0 = (group,     tid*4..+3)
//                          a1 = (group + 8, tid*4..+3)
//                          a2 = (group,     16 + tid*4..+3)
//                          a3 = (group + 8, 16 + tid*4..+3)
//   B (8 rows x 32 K)      b0 = (group,     tid*4..+3)
//                          b1 = (group,     16 + tid*4..+3)
//   C (16 tokens x 8 rows) c0 = (group, tid*2), c1 = (group, tid*2+1),
//                          c2 = (group+8, tid*2), c3 = (group+8, tid*2+1)
// The shared stage holds each row as 32 int8 values plus 16 bytes of padding,
// so every row is 16-byte aligned and the 4-byte word a fragment reads lands
// on a distinct bank for the eight rows one warp touches. See the module doc in
// MmqTcCodegen.h for the operand layouts.

#include <sonicboom/compiler/MmqTcCodegen.h>

#include <llvm/Config/llvm-config.h>

#include <string>
#include <vector>

#include <sonicboom/compiler/GlueCodegen.h>

namespace sonicboom::compiler {
namespace {

constexpr int kRowBytes = 48;    // 32 int8 values + 16 bytes padding
constexpr int kRowWords = 12;    // 48 / 4
constexpr int kRowChunks = 3;    // 48 / 16
constexpr int kMmaK = 32;        // one m16n8k32 mma per chunk

struct Emitter {
  std::string body;
  int next = 0;

  void op(const std::string &s) { body += "      " + s + "\n"; }

  std::string val(const std::string &expr) {
    std::string v = "%v" + std::to_string(++next);
    body += "      " + v + " = " + expr + "\n";
    return v;
  }
};

std::string mmq_tc_region_text(const MmqTcShape &sh, int sm_arch) {
  const int BM = sh.bm;
  const int BN = sh.bn;
  const int DS = sh.ds;
  const int WG = BM / 8;           // 8-row weight groups per block
  const int NW = WG < 8 ? WG : 8;  // warps that own weight groups
  const int WGPW = WG / NW;        // weight groups per warp
  const int TH = NW * 32;          // threads per block
  const int TG = BN / 16;          // 16-token groups per block
  const bool HF = (DS == 1);       // activation scale is a half2 ds4

  Emitter e;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](int v) { return e.val("arith.constant " + s(v) + " : i32"); };
  const auto c64 = [&](int v) { return e.val("arith.constant " + s(v) + " : i64"); };
  const auto ci = [&](int v) { return e.val("arith.constant " + s(v) + " : index"); };
  const auto f32c = [&](const char *v) {
    return e.val(std::string("arith.constant ") + v + " : f32");
  };

  const std::string c0 = c32(0), c1 = c32(1), c2 = c32(2), c3 = c32(3);
  const std::string c4 = c32(4), c5 = c32(5), c7 = c32(7), c8 = c32(8);
  const std::string c15 = c32(15), c16 = c32(16), c31 = c32(31);
  const std::string c32_ = c32(32), c144 = c32(144);
  const std::string l0 = c64(0), l1 = c64(1);
  const std::string i0 = ci(0), i1 = ci(1), i8 = ci(8);
  const std::string ith = ci(TH);
  const std::string i2bn = ci(2 * BN), ibn = ci(BN);
  const std::string i2bm = ci(2 * BM), ibm = ci(BM);
  const std::string i12 = ci(kRowWords), i3 = ci(kRowChunks);
  const std::string zf = f32c("0.0");

  const int bw_off = 2 * BN * kRowBytes;
  const int asc_off = 2 * (BN + BM) * kRowBytes;
  const int bsc_off = asc_off + 2 * BN * 4;

  const std::string FUNC =
      "gpu.func @mmq_tc(%a: !llvm.ptr, %wq: !llvm.ptr, %ws: !llvm.ptr, "
      "%y: !llvm.ptr, %m: i32, %k: i32, %n: i32) kernel {";

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string smem = e.val("gpu.dynamic_shared_memory : memref<?xi8, "
                                 "#gpu.address_space<workgroup>>");
  // Two views of each staged operand: 16-byte chunks for staging, 4-byte words
  // for the mma fragments. Both cover the same bytes.
  const std::string ac = e.val(
      "memref.view " + smem + "[" + i0 +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BN) + "x" + s(kRowChunks) +
      "xvector<16xi8>, #gpu.address_space<workgroup>>");
  const std::string aw = e.val(
      "memref.view " + smem + "[" + i0 +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BN) + "x" + s(kRowWords) +
      "xvector<4xi8>, #gpu.address_space<workgroup>>");
  const std::string bc = e.val(
      "memref.view " + smem + "[" + ci(bw_off) +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BM) + "x" + s(kRowChunks) +
      "xvector<16xi8>, #gpu.address_space<workgroup>>");
  const std::string bws = e.val(
      "memref.view " + smem + "[" + ci(bw_off) +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BM) + "x" + s(kRowWords) +
      "xvector<4xi8>, #gpu.address_space<workgroup>>");
  const std::string ascl = e.val(
      "memref.view " + smem + "[" + ci(asc_off) +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BN) + "xf32, #gpu.address_space<workgroup>>");
  const std::string bscl = e.val(
      "memref.view " + smem + "[" + ci(bsc_off) +
      "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
      s(BM) + "xf32, #gpu.address_space<workgroup>>");
  const std::string tidi = e.val("arith.index_cast " + tid + " : index to i32");
  const std::string lane = e.val("arith.andi " + tidi + ", " + c31 + " : i32");
  const std::string warp = e.val("arith.shrui " + tidi + ", " + c5 + " : i32");
  const std::string group = e.val("arith.shrui " + lane + ", " + c2 + " : i32");
  const std::string tg = e.val("arith.andi " + lane + ", " + c3 + " : i32");
  const std::string bxi = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string byi = e.val("arith.index_cast " + by + " : index to i32");
  const std::string n0 = e.val("arith.muli " + bxi + ", " + c32(BN) + " : i32");
  const std::string m0 = e.val("arith.muli " + byi + ", " + c32(BM) + " : i32");
  const std::string k32 = e.val("arith.shrui %k, " + c5 + " : i32");
  const std::string kib = e.val("arith.index_cast %k : i32 to index");
  const std::string kchunks = e.val("arith.ceildivui " + kib + ", " + ci(kMmaK) +
                                    " : index");
  const std::string k64 = e.val("arith.extsi %k : i32 to i64");

  // ---- the K loop: stage, mma, accumulate (f32) ----
  const int NACC = WGPW * TG * 4;
  std::vector<std::string> init, names, types;
  for (int i = 0; i < NACC; ++i) {
    init.push_back(zf);
    names.push_back("%acc" + std::to_string(i));
    types.push_back("f32");
  }
  std::string ia = "iter_args(";
  for (int i = 0; i < NACC; ++i)
    ia += (i ? ", " : "") + names[i] + " = " + init[i];
  ia += ")";
  std::string ty = " -> (";
  std::string res;
  std::vector<std::string> outs;
  for (int i = 0; i < NACC; ++i) {
    ty += (i ? ", " : "") + std::string("f32");
    outs.push_back("%ar" + std::to_string(i));
    res += (i ? ", " : "") + outs[i];
  }
  ty += ")";
  e.op(res + " = scf.for %kc = " + i0 + " to " + kchunks + " step " + i1 +
       " " + ia + ty + " {");

  // ---- one staged 32-K chunk ----
  const std::string kci = e.val("arith.index_cast %kc : index to i32");
  const std::string kb = e.val("arith.shrui " + kci + ", " + c2 + " : i32");
  const std::string ks = e.val("arith.andi " + kci + ", " + c3 + " : i32");
  // Byte offset of (block kb, token 0) in the activation, and the data/scale
  // offsets within that block: scales are the first 16 bytes, q8_1 values
  // start at byte 16.
  const std::string ablk = e.val(
      "arith.muli " + e.val("arith.muli " + kb + ", %n : i32") + ", " + c144 +
      " : i32");
  const std::string abase = e.val(
      "arith.addi " + e.val("arith.addi " + ablk + ", " + c16 + " : i32") +
      ", " + e.val("arith.muli " + ks + ", " + c32_ + " : i32") + " : i32");
  const std::string asbase = e.val(
      "arith.addi " + ablk + ", " +
      e.val("arith.muli " + ks + ", " + c4 + " : i32") + " : i32");
  const std::string kbase = e.val("arith.muli " + kci + ", " + c32_ + " : i32");

  // A: one 16-byte chunk of a token's block_q8_1_mmq block per iteration.
  {
    const std::string it = "%ci_a";
    e.op("scf.for " + it + " = " + tid + " to " + i2bn + " step " + ith +
         " : index {");
    const std::string row = e.val("arith.shrui " + it + ", " + i1 + " : index");
    const std::string half = e.val("arith.andi " + it + ", " + i1 + " : index");
    const std::string row32 = e.val("arith.index_cast " + row + " : index to i32");
    const std::string half32 =
        e.val("arith.index_cast " + half + " : index to i32");
    const std::string tok = e.val("arith.addi " + n0 + ", " + row32 + " : i32");
    const std::string ok = e.val("arith.cmpi slt, " + tok + ", %n : i32");
    const std::string t = e.val("arith.select " + ok + ", " + tok + ", " + c0 +
                                " : i32");
    const std::string tb = e.val("arith.muli " + t + ", " + c144 + " : i32");
    const std::string hb = e.val("arith.muli " + half32 + ", " + c16 + " : i32");
    const std::string boff = e.val("arith.addi " + e.val("arith.addi " + abase +
                                                         ", " + tb + " : i32") +
                                   ", " + hb + " : i32");
    const std::string boff64 = e.val("arith.extsi " + boff + " : i32 to i64");
    const std::string p = e.val("llvm.getelementptr %a[" + boff64 +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, i8");
    const std::string v = e.val("llvm.load " + p + " : !llvm.ptr -> vector<16xi8>");
    const std::string zv = e.val("arith.constant dense<0> : vector<16xi8>");
    const std::string vz = e.val("arith.select " + ok + ", " + v + ", " + zv +
                                 " : vector<16xi8>");
    e.op("memref.store " + vz + ", " + ac + "[" + i0 + ", " + row + ", " +
         half + "] : memref<2x" + s(BN) + "x" + s(kRowChunks) +
         "xvector<16xi8>, #gpu.address_space<workgroup>>");
    e.op("}");
  }

  // B: one 16-byte chunk of a weight row per iteration.
  {
    const std::string it = "%ci_b";
    e.op("scf.for " + it + " = " + tid + " to " + i2bm + " step " + ith +
         " : index {");
    const std::string row = e.val("arith.shrui " + it + ", " + i1 + " : index");
    const std::string half = e.val("arith.andi " + it + ", " + i1 + " : index");
    const std::string row32 = e.val("arith.index_cast " + row + " : index to i32");
    const std::string half32 =
        e.val("arith.index_cast " + half + " : index to i32");
    const std::string wrow = e.val("arith.addi " + m0 + ", " + row32 + " : i32");
    const std::string ok = e.val("arith.cmpi slt, " + wrow + ", %m : i32");
    const std::string r = e.val("arith.select " + ok + ", " + wrow + ", " + c0 +
                                " : i32");
    const std::string r64 = e.val("arith.extsi " + r + " : i32 to i64");
    const std::string roff = e.val("arith.muli " + r64 + ", " + k64 + " : i64");
    const std::string kb64 = e.val("arith.extsi " + kbase + " : i32 to i64");
    const std::string hb64 = e.val("arith.extsi " + e.val("arith.muli " + half32 +
                                                          ", " + c16 + " : i32") +
                                   " : i32 to i64");
    const std::string boff = e.val(
        "arith.addi " + e.val("arith.addi " + roff + ", " + kb64 + " : i64") +
        ", " + hb64 + " : i64");
    const std::string p = e.val("llvm.getelementptr %wq[" + boff +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, i8");
    const std::string v = e.val("llvm.load " + p + " : !llvm.ptr -> vector<16xi8>");
    const std::string zv = e.val("arith.constant dense<0> : vector<16xi8>");
    const std::string vz = e.val("arith.select " + ok + ", " + v + ", " + zv +
                                 " : vector<16xi8>");
    e.op("memref.store " + vz + ", " + bc + "[" + i0 + ", " + row + ", " +
         half + "] : memref<2x" + s(BM) + "x" + s(kRowChunks) +
         "xvector<16xi8>, #gpu.address_space<workgroup>>");
    e.op("}");
  }

  // Activation scales, one per token: block (kb, token), sub-scale ks.
  {
    const std::string it = "%ci_sa";
    e.op("scf.for " + it + " = " + tid + " to " + ibn + " step " + ith +
         " : index {");
    const std::string i32v = e.val("arith.index_cast " + it + " : index to i32");
    const std::string tok = e.val("arith.addi " + n0 + ", " + i32v + " : i32");
    const std::string ok = e.val("arith.cmpi slt, " + tok + ", %n : i32");
    const std::string t = e.val("arith.select " + ok + ", " + tok + ", " + c0 +
                                " : i32");
    const std::string tb = e.val("arith.muli " + t + ", " + c144 + " : i32");
    const std::string boff =
        e.val("arith.addi " + asbase + ", " + tb + " : i32");
    const std::string boff64 = e.val("arith.extsi " + boff + " : i32 to i64");
    const std::string p = e.val("llvm.getelementptr %a[" + boff64 +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, i8");
    std::string sc;
    if (HF) {
      const std::string h = e.val("llvm.load " + p + " : !llvm.ptr -> f16");
      sc = e.val("arith.extf " + h + " : f16 to f32");
    } else {
      sc = e.val("llvm.load " + p + " : !llvm.ptr -> f32");
    }
    const std::string s0 = e.val("arith.select " + ok + ", " + sc + ", " + zf +
                                 " : f32");
    e.op("memref.store " + s0 + ", " + ascl + "[" + i0 + ", " + it +
         "] : memref<2x" + s(BN) + "xf32, #gpu.address_space<workgroup>>");
    e.op("}");
  }

  // Weight scales, one per weight row: ws[row][kc].
  {
    const std::string it = "%ci_sb";
    e.op("scf.for " + it + " = " + tid + " to " + ibm + " step " + ith +
         " : index {");
    const std::string i32v = e.val("arith.index_cast " + it + " : index to i32");
    const std::string wrow = e.val("arith.addi " + m0 + ", " + i32v + " : i32");
    const std::string ok = e.val("arith.cmpi slt, " + wrow + ", %m : i32");
    const std::string r = e.val("arith.select " + ok + ", " + wrow + ", " + c0 +
                                " : i32");
    const std::string roff = e.val("arith.muli " + r + ", " + k32 + " : i32");
    const std::string off = e.val("arith.addi " + roff + ", " + kci + " : i32");
    const std::string off64 = e.val("arith.extsi " + off + " : i32 to i64");
    const std::string p = e.val("llvm.getelementptr %ws[" + off64 +
                                "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
    const std::string sc = e.val("llvm.load " + p + " : !llvm.ptr -> f32");
    const std::string s0 = e.val("arith.select " + ok + ", " + sc + ", " + zf +
                                 " : f32");
    e.op("memref.store " + s0 + ", " + bscl + "[" + i0 + ", " + it +
         "] : memref<2x" + s(BM) + "xf32, #gpu.address_space<workgroup>>");
    e.op("}");
  }
  e.op("gpu.barrier");

  // ---- mma over the staged chunk ----
  std::vector<std::string> cur = names;
  const std::string czi = e.val("arith.constant dense<0> : vector<2x2xi32>");
  const std::string z4 = e.val("arith.constant dense<0> : vector<4x4xi8>");
  const std::string z2 = e.val("arith.constant dense<0> : vector<2x4xi8>");
  const std::string tgi = e.val("arith.index_cast " + tg + " : i32 to index");
  const std::string tg4 = e.val("arith.index_cast " +
                                e.val("arith.addi " + tg + ", " + c4 + " : i32") +
                                " : i32 to index");
  for (int wi = 0; wi < WGPW; ++wi) {
    std::string wg = warp;
    if (wi != 0)
      wg = e.val("arith.addi " + warp + ", " + c32(wi * NW) + " : i32");
    const std::string wg8 = e.val("arith.muli " + wg + ", " + c8 + " : i32");
    const std::string brow = e.val("arith.addi " + wg8 + ", " + group + " : i32");
    const std::string browi = e.val("arith.index_cast " + brow + " : i32 to index");
    const std::string bt = e.val("arith.muli " + wg8 + ", " + c1 + " : i32");
    const std::string bs0i32 = e.val("arith.addi " + bt + ", " +
                                     e.val("arith.muli " + tg + ", " + c2 +
                                           " : i32") + " : i32");
    const std::string bs0i = e.val("arith.index_cast " + bs0i32 + " : i32 to index");
    const std::string bs1i = e.val("arith.index_cast " +
                                   e.val("arith.addi " + bs0i32 + ", " + c1 +
                                         " : i32") + " : i32 to index");
    const std::string bsc0 = e.val("memref.load " + bscl + "[" + i0 + ", " +
                                   bs0i + "] : memref<2x" + s(BM) +
                                   "xf32, #gpu.address_space<workgroup>>");
    const std::string bsc1 = e.val("memref.load " + bscl + "[" + i0 + ", " +
                                   bs1i + "] : memref<2x" + s(BM) +
                                   "xf32, #gpu.address_space<workgroup>>");
    const std::string b0 = e.val("memref.load " + bws + "[" + i0 + ", " +
                                 browi + ", " + tgi + "] : memref<2x" + s(BM) +
                                 "x" + s(kRowWords) +
                                 "xvector<4xi8>, #gpu.address_space<workgroup>>");
    const std::string b1 = e.val("memref.load " + bws + "[" + i0 + ", " +
                                 browi + ", " + tg4 + "] : memref<2x" + s(BM) +
                                 "x" + s(kRowWords) +
                                 "xvector<4xi8>, #gpu.address_space<workgroup>>");
    const std::string b01 = e.val("vector.insert " + b0 + ", " + z2 +
                                  "[0] : vector<4xi8> into vector<2x4xi8>");
    const std::string B = e.val("vector.insert " + b1 + ", " + b01 +
                                "[1] : vector<4xi8> into vector<2x4xi8>");
    for (int g = 0; g < TG; ++g) {
      const std::string r0i32 =
          e.val("arith.addi " + group + ", " + c32(g * 16) + " : i32");
      const std::string r1i32 =
          e.val("arith.addi " + r0i32 + ", " + c8 + " : i32");
      const std::string r0 = e.val("arith.index_cast " + r0i32 + " : i32 to index");
      const std::string r1 = e.val("arith.index_cast " + r1i32 + " : i32 to index");
      const std::string a0 = e.val("memref.load " + aw + "[" + i0 + ", " + r0 +
                                   ", " + tgi + "] : memref<2x" + s(BN) + "x" +
                                   s(kRowWords) +
                                   "xvector<4xi8>, #gpu.address_space<workgroup>>");
      const std::string a1 = e.val("memref.load " + aw + "[" + i0 + ", " + r1 +
                                   ", " + tgi + "] : memref<2x" + s(BN) + "x" +
                                   s(kRowWords) +
                                   "xvector<4xi8>, #gpu.address_space<workgroup>>");
      const std::string a2 = e.val("memref.load " + aw + "[" + i0 + ", " + r0 +
                                   ", " + tg4 + "] : memref<2x" + s(BN) + "x" +
                                   s(kRowWords) +
                                   "xvector<4xi8>, #gpu.address_space<workgroup>>");
      const std::string a3 = e.val("memref.load " + aw + "[" + i0 + ", " + r1 +
                                   ", " + tg4 + "] : memref<2x" + s(BN) + "x" +
                                   s(kRowWords) +
                                   "xvector<4xi8>, #gpu.address_space<workgroup>>");
      const std::string t0 = e.val("vector.insert " + a0 + ", " + z4 +
                                   "[0] : vector<4xi8> into vector<4x4xi8>");
      const std::string t1 = e.val("vector.insert " + a1 + ", " + t0 +
                                   "[1] : vector<4xi8> into vector<4x4xi8>");
      const std::string t2 = e.val("vector.insert " + a2 + ", " + t1 +
                                   "[2] : vector<4xi8> into vector<4x4xi8>");
      const std::string A = e.val("vector.insert " + a3 + ", " + t2 +
                                  "[3] : vector<4xi8> into vector<4x4xi8>");
      const std::string as0i = e.val("arith.index_cast " + r0i32 + " : i32 to index");
      const std::string as1i = e.val("arith.index_cast " + r1i32 + " : i32 to index");
      const std::string asc0 = e.val("memref.load " + ascl + "[" + i0 + ", " +
                                     as0i + "] : memref<2x" + s(BN) +
                                     "xf32, #gpu.address_space<workgroup>>");
      const std::string asc1 = e.val("memref.load " + ascl + "[" + i0 + ", " +
                                     as1i + "] : memref<2x" + s(BN) +
                                     "xf32, #gpu.address_space<workgroup>>");
      const std::string p00 = e.val("arith.mulf " + asc0 + ", " + bsc0 + " : f32");
      const std::string p01 = e.val("arith.mulf " + asc0 + ", " + bsc1 + " : f32");
      const std::string p10 = e.val("arith.mulf " + asc1 + ", " + bsc0 + " : f32");
      const std::string p11 = e.val("arith.mulf " + asc1 + ", " + bsc1 + " : f32");
      const std::string r = e.val(
          "nvgpu.mma.sync (" + A + ", " + B + ", " + czi +
          ") {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, "
          "vector<2x2xi32>) -> vector<2x2xi32>");
      for (int rr = 0; rr < 2; ++rr) {
        for (int cc = 0; cc < 2; ++cc) {
          const int idx = (wi * TG + g) * 4 + rr * 2 + cc;
          const std::string cv = e.val("vector.extract " + r + "[" + s(rr) +
                                       ", " + s(cc) + "] : i32 from "
                                       "vector<2x2xi32>");
          const std::string fv = e.val("arith.sitofp " + cv + " : i32 to f32");
          const std::string pv =
              rr == 0 ? (cc == 0 ? p00 : p01) : (cc == 0 ? p10 : p11);
          const std::string mv = e.val("arith.mulf " + fv + ", " + pv + " : f32");
          cur[idx] = e.val("arith.addf " + cur[idx] + ", " + mv + " : f32");
        }
      }
    }
  }
  e.op("gpu.barrier");
  std::string yields, ytypes;
  for (int i = 0; i < NACC; ++i) {
    yields += (i ? ", " : "") + cur[i];
    ytypes += (i ? ", " : "") + std::string("f32");
  }
  e.op("scf.yield " + yields + " : " + ytypes);
  e.op("}");
  std::vector<std::string> accs = outs;

  // ---- store: Y[token][row], f32 ----
  const std::string m64 = e.val("arith.extsi %m : i32 to i64");
  for (int wi = 0; wi < WGPW; ++wi) {
    std::string wg = warp;
    if (wi != 0)
      wg = e.val("arith.addi " + warp + ", " + c32(wi * NW) + " : i32");
    const std::string wg8 = e.val("arith.muli " + wg + ", " + c8 + " : i32");
    const std::string row0 = e.val("arith.addi " + m0 + ", " +
                                   e.val("arith.addi " + wg8 + ", " +
                                         e.val("arith.muli " + tg + ", " + c2 +
                                               " : i32") + " : i32") + " : i32");
    for (int g = 0; g < TG; ++g) {
      for (int rr = 0; rr < 2; ++rr) {
        const std::string tok = e.val(
            "arith.addi " + n0 + ", " +
            e.val("arith.addi " + group + ", " + c32(g * 16 + rr * 8) +
                  " : i32") + " : i32");
        const std::string tok_ok =
            e.val("arith.cmpi slt, " + tok + ", %n : i32");
        for (int cc = 0; cc < 2; ++cc) {
          std::string row = row0;
          if (cc != 0)
            row = e.val("arith.addi " + row0 + ", " + c1 + " : i32");
          const std::string mok = e.val("arith.cmpi slt, " + row + ", %m : i32");
          const std::string ok = e.val("arith.andi " + mok + ", " + tok_ok +
                                       " : i1");
          e.op("scf.if " + ok + " {");
          const std::string tok64 = e.val("arith.extsi " + tok + " : i32 to i64");
          const std::string off0 = e.val("arith.muli " + tok64 + ", " + m64 +
                                         " : i64");
          const std::string row64 = e.val("arith.extsi " + row + " : i32 to i64");
          const std::string off = e.val("arith.addi " + off0 + ", " + row64 +
                                        " : i64");
          const std::string p = e.val("llvm.getelementptr %y[" + off +
                                      "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
          const int idx = (wi * TG + g) * 4 + rr * 2 + cc;
          e.op("llvm.store " + accs[idx] + ", " + p + " : f32, !llvm.ptr");
          e.op("}");
        }
      }
    }
  }
  e.op("gpu.return");
  e.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @mmq_tc [#nvvm.target<chip = \"sm_" + s(sm_arch) +
          "\">] {\n";
  text += "      " + FUNC + "\n" + e.body + "\n";
  text += "  }\n}\n";
  return text;
}

}  // namespace

bool mmq_tc_shape_supported(const MmqTcShape &shape) {
  const auto ok_bm = [](int v) {
    return v == 16 || v == 32 || v == 64 || v == 128;
  };
  if (!ok_bm(shape.bm) || !ok_bm(shape.bn)) return false;
  if (shape.ds != 0 && shape.ds != 1 && shape.ds != 2) return false;
  return mmq_tc_shared_bytes(shape) <= 96 * 1024;
}

int mmq_tc_block_threads(const MmqTcShape &shape) {
  const int wg = shape.bm / 8;
  return (wg < 8 ? wg : 8) * 32;
}

int mmq_tc_shared_bytes(const MmqTcShape &shape) {
  // Two buffers (one in flight until the double-buffered staging lands), each
  // (bm + bn) rows of 48 bytes, plus f32 scales.
  return 2 * (shape.bm + shape.bn) * (kRowBytes + 4);
}

std::optional<std::string> generate_mmq_tc_ptx(const MmqTcShape &shape,
                                               int sm_arch) {
  if (!mmq_tc_shape_supported(shape)) return std::nullopt;
  const std::string text = mmq_tc_region_text(shape, sm_arch);
  return lower_gpu_module_to_ptx(text, /*link_libdevice=*/false, "mmq-tc");
}

std::string mmq_tc_codegen_version() {
  return std::string("llvm-") + LLVM_VERSION_STRING + "-mmqtc1";
}

}  // namespace sonicboom::compiler
