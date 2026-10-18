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

// The generated dense bf16 GEMM (compiler/GemmCodegen.h).
//
// The kernel is emitted directly in the GPU dialect: one gpu.func, a grid of
// (token tile, row tile) blocks, each warp owning a 16-row strip of the output
// tile. Every K step stages BM x BK of W and BN x BK of X into double-buffered
// shared memory with nvgpu.device_async_copy (one cp.async per lane), waits
// with cp.async.wait_group (one group outstanding while the next tile is in
// flight), and then runs one m16n8k16 mma per (16-row strip, 8-token tile,
// 16-K slice).
//
// The A fragment is the non-transposed ldmatrix x4 form (row = warp*16 + l&15,
// col = kb*16 + (l>>4)*8); the B fragment is the non-transposed ldmatrix x2
// form read from a [BN][BK] token-major stage (row = nt*8 + l&7,
// col = kb*16 + ((l>>3)&1)*8). Both match the layouts documented in
// AttentionCodegen.h and validated there.

#include <sonicboom/compiler/GemmCodegen.h>

#include <llvm/Config/llvm-config.h>

#include <map>
#include <string>
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
};

std::string gemm_region_text(const DenseGemmShape &sh, int sm_arch) {
  const int BM = sh.bm;
  const int BN = sh.bn;
  const int BK = sh.bk;
  const int NW = BM / 16;  // warps (one 16-row strip each)
  const int TH = NW * 32;
  const int NT = BN / 8;   // 8-token accumulator tiles per lane
  const int KB = BK / 16;  // 16-K mma steps per stage

  Emitter e;
  std::map<std::string, std::string> C;
  const auto s = [](int v) { return std::to_string(v); };
  const auto c32 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i32");
  };
  const auto c64 = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : i64");
  };
  const auto ci = [&](const char *n, int v) {
    C[n] = e.val("arith.constant " + s(v) + " : index");
  };

  c32("c0", 0); c32("c1", 1); c32("c2", 2); c32("c3", 3); c32("c4", 4);
  c32("c5", 5); c32("c7", 7); c32("c8", 8); c32("c15", 15); c32("c16", 16);
  c32("c31", 31);
  c64("l0", 0); c64("l1", 1); c64("l8", 8);
  ci("i0", 0); ci("i1", 1); ci("i8", 8);
  ci("ith", TH);                 // staging stride (one chunk per lane)
  ci("ibk", BK);
  ci("ibsoff", 2 * BM * BK * 2); // B stage starts after the A double buffer
  ci("icha", BM * BK / 8);       // 8-bf16 chunks per A tile
  ci("ichb", BN * BK / 8);       // 8-bf16 chunks per B tile (token-major)
  c32("ibmi", BM); c32("ibni", BN); c32("ibki", BK);

  const std::string FUNC =
      "gpu.func @gemm(%x: memref<?x?xbf16>, %w: memref<?x?xbf16>, "
      "%y: !llvm.ptr, %m: i32, %k: i32, %n: i32) kernel {";

  const std::string tid = e.val("gpu.thread_id x");
  const std::string bx = e.val("gpu.block_id x");
  const std::string by = e.val("gpu.block_id y");
  const std::string smem = e.val("gpu.dynamic_shared_memory : memref<?xi8, "
                                 "#gpu.address_space<workgroup>>");
  const std::string asm_ =
      e.val("memref.view " + smem + "[" + C["i0"] +
            "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
            s(BM) + "x" + s(BK) +
            "xbf16, #gpu.address_space<workgroup>>");
  const std::string bsm =
      e.val("memref.view " + smem + "[" + C["ibsoff"] +
            "][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x" +
            s(BN) + "x" + s(BK) +
            "xbf16, #gpu.address_space<workgroup>>");
  const std::string tidi = e.val("arith.index_cast " + tid + " : index to i32");
  const std::string lane =
      e.val("arith.andi " + tidi + ", " + C["c31"] + " : i32");
  const std::string warp =
      e.val("arith.shrui " + tidi + ", " + C["c5"] + " : i32");
  const std::string bxi = e.val("arith.index_cast " + bx + " : index to i32");
  const std::string byi = e.val("arith.index_cast " + by + " : index to i32");
  const std::string n0 =
      e.val("arith.muli " + bxi + ", " + C["ibni"] + " : i32");
  const std::string m0 =
      e.val("arith.muli " + byi + ", " + C["ibmi"] + " : i32");

  // One staged tile of `nrows` rows x BK columns, 8 bf16 (16 B) per lane per
  // iteration. `src` is the global operand, `dst` the shared double buffer,
  // `base`/`bound` the row origin and its runtime bound, `kt` the K step, and
  // `buf` the double-buffer slot. Out-of-range chunks request a zero fill.
  const auto stage_body = [&](const char *src, const std::string &dst,
                              const std::string &buf, const std::string &kt,
                              const std::string &base, const char *bound,
                              int nrows, const char *chunks, const char *tag) {
    const std::string it = "%ci_" + std::string(tag);
    e.op("scf.for " + it + " = " + tid + " to " + C[chunks] + " step " +
         C["ith"] + " : index {");
    const std::string el =
        e.val("arith.muli " + it + ", " + C["i8"] + " : index");
    const std::string row = e.val("arith.divui " + el + ", " + C["ibk"] +
                                 " : index");
    const std::string col = e.val("arith.remui " + el + ", " + C["ibk"] +
                                 " : index");
    const std::string row32 =
        e.val("arith.index_cast " + row + " : index to i32");
    const std::string col32 =
        e.val("arith.index_cast " + col + " : index to i32");
    const std::string ktcol =
        e.val("arith.muli " + kt + ", " + C["ibki"] + " : i32");
    const std::string grow =
        e.val("arith.addi " + base + ", " + row32 + " : i32");
    const std::string gcol =
        e.val("arith.addi " + ktcol + ", " + col32 + " : i32");
    const std::string rok =
        e.val("arith.cmpi slt, " + grow + ", " + bound + " : i32");
    const std::string cok = e.val("arith.cmpi slt, " + gcol + ", %k : i32");
    const std::string ok =
        e.val("arith.andi " + rok + ", " + cok + " : i1");
    const std::string sr =
        e.val("arith.select " + ok + ", " + grow + ", " + C["c0"] + " : i32");
    const std::string sc =
        e.val("arith.select " + ok + ", " + gcol + ", " + C["c0"] + " : i32");
    const std::string sri =
        e.val("arith.index_cast " + sr + " : i32 to index");
    const std::string sci =
        e.val("arith.index_cast " + sc + " : i32 to index");
    const std::string nel =
        e.val("arith.select " + ok + ", " + C["i8"] + ", " + C["i0"] +
              " : index");
    e.val(std::string("nvgpu.device_async_copy %") + src + "[" + sri + ", " +
          sci + "], " + dst + "[" + buf + ", " + row + ", " + col + "], 8, " +
          nel + " : memref<?x?xbf16> to memref<2x" + s(nrows) + "x" + s(BK) +
          "xbf16, #gpu.address_space<workgroup>>");
    e.op("}");
  };

  const std::string kidx = e.val("arith.index_cast %k : i32 to index");
  const std::string lane15 =
      e.val("arith.andi " + lane + ", " + C["c15"] + " : i32");
  const std::string lsh16 =
      e.val("arith.shrui " + lane + ", " + C["c4"] + " : i32");
  const std::string l7 =
      e.val("arith.andi " + lane + ", " + C["c7"] + " : i32");
  const std::string lsh3 =
      e.val("arith.shrui " + lane + ", " + C["c3"] + " : i32");
  const std::string lsh3m =
      e.val("arith.andi " + lsh3 + ", " + C["c1"] + " : i32");
  const std::string w16 =
      e.val("arith.muli " + warp + ", " + C["c16"] + " : i32");
  const std::string arow32 =
      e.val("arith.addi " + w16 + ", " + lane15 + " : i32");
  const std::string arowi =
      e.val("arith.index_cast " + arow32 + " : i32 to index");
  const std::string acol32 =
      e.val("arith.muli " + lsh16 + ", " + C["c8"] + " : i32");
  const std::string bcoloff =
      e.val("arith.muli " + lsh3m + ", " + C["c8"] + " : i32");
  const std::string ktmax =
      e.val("arith.ceildivui " + kidx + ", " + C["ibk"] + " : index");
  const std::string ktmaxi =
      e.val("arith.index_cast " + ktmax + " : index to i32");

  // Prologue: stage K step 0 into slot 0 and commit it as its own group.
  stage_body("w", asm_, C["i0"], C["c0"], m0, "%m", BM, "icha", "a0");
  stage_body("x", bsm, C["i0"], C["c0"], n0, "%n", BN, "ichb", "b0");
  const std::string gtok =
      e.val("\"nvgpu.device_async_create_group\"() : () -> "
            "!nvgpu.device.async.token");

  std::vector<std::string> accs, names;
  for (int i = 0; i < NT; ++i) {
    accs.push_back(
        e.val("arith.constant dense<0.0> : vector<2x2xf32>"));
    names.push_back("%o" + std::to_string(i));
  }
  std::string ia = "iter_args(";
  for (int i = 0; i < NT; ++i)
    ia += (i ? ", " : "") + names[i] + " = " + accs[i];
  ia += ")";
  std::string ty = " -> (";
  std::string res;
  for (int i = 0; i < NT; ++i) {
    ty += (i ? ", " : "") + std::string("vector<2x2xf32>");
    res += (i ? ", " : "") + std::string("%r") + std::to_string(i);
  }
  ty += ")";

  e.op(res + " = scf.for %kt = " + C["i0"] + " to " + ktmax + " step " +
       C["i1"] + " " + ia + ty + " {");
  const std::string kti = e.val("arith.index_cast %kt : index to i32");
  std::vector<std::string> cur = names;
  const std::string nxt =
      e.val("arith.addi " + kti + ", " + C["c1"] + " : i32");
  const std::string has =
      e.val("arith.cmpi slt, " + nxt + ", " + ktmaxi + " : i32");
  // Issue the next stage while this one computes; the wait below keeps at most
  // one group in flight, so the last iteration waits for completion.
  e.op("scf.if " + has + " {");
  const std::string bufn =
      e.val("arith.andi " + nxt + ", " + C["c1"] + " : i32");
  const std::string bufi =
      e.val("arith.index_cast " + bufn + " : i32 to index");
  stage_body("w", asm_, bufi, nxt, m0, "%m", BM, "icha", "a1");
  stage_body("x", bsm, bufi, nxt, n0, "%n", BN, "ichb", "b1");
  e.val("\"nvgpu.device_async_create_group\"() : () -> "
        "!nvgpu.device.async.token");
  e.op("}");
  e.op("scf.if " + has + " {");
  e.op("nvgpu.device_async_wait " + gtok + " {numGroups = 1 : i32}");
  e.op("} else {");
  e.op("nvgpu.device_async_wait " + gtok + " {numGroups = 0 : i32}");
  e.op("}");
  e.op("gpu.barrier");
  const std::string buf =
      e.val("arith.andi " + kti + ", " + C["c1"] + " : i32");
  const std::string buf_i =
      e.val("arith.index_cast " + buf + " : i32 to index");

  for (int kb = 0; kb < KB; ++kb) {
    std::string acol = acol32, bcol = bcoloff;
    if (kb != 0) {
      const std::string kc = e.val("arith.constant " + s(kb * 16) + " : i32");
      acol = e.val("arith.addi " + acol32 + ", " + kc + " : i32");
      bcol = e.val("arith.addi " + bcoloff + ", " + kc + " : i32");
    }
    const std::string aci =
        e.val("arith.index_cast " + acol + " : i32 to index");
    const std::string a =
        e.val("nvgpu.ldmatrix " + asm_ + "[" + buf_i + ", " + arowi + ", " +
              aci + "] {numTiles = 4 : i32, transpose = false} : memref<2x" +
              s(BM) + "x" + s(BK) +
              "xbf16, #gpu.address_space<workgroup>> -> vector<4x2xbf16>");
    const std::string bci =
        e.val("arith.index_cast " + bcol + " : i32 to index");
    for (int nt = 0; nt < NT; ++nt) {
      std::string brow32 = l7;
      if (nt != 0) {
        const std::string bc =
            e.val("arith.constant " + s(nt * 8) + " : i32");
        brow32 = e.val("arith.addi " + l7 + ", " + bc + " : i32");
      }
      const std::string browi =
          e.val("arith.index_cast " + brow32 + " : i32 to index");
      const std::string b =
          e.val("nvgpu.ldmatrix " + bsm + "[" + buf_i + ", " + browi + ", " +
                bci + "] {numTiles = 2 : i32, transpose = false} : memref<2x" +
                s(BN) + "x" + s(BK) +
                "xbf16, #gpu.address_space<workgroup>> -> vector<2x2xbf16>");
      cur[nt] = e.val("nvgpu.mma.sync (" + a + ", " + b + ", " + cur[nt] +
                      ") {mmaShape = [16, 8, 16]} : (vector<4x2xbf16>, "
                      "vector<2x2xbf16>, vector<2x2xf32>) -> "
                      "vector<2x2xf32>");
    }
  }
  e.op("gpu.barrier");
  std::string yields, ytypes;
  for (int i = 0; i < NT; ++i) {
    yields += (i ? ", " : "") + cur[i];
    ytypes += (i ? ", " : "") + std::string("vector<2x2xf32>");
  }
  e.op("scf.yield " + yields + " : " + ytypes);
  e.op("}");

  // ---- store: Y[token][row] (token-major), f32 ----
  const std::string l2c =
      e.val("arith.andi " + lane + ", " + C["c3"] + " : i32");
  const std::string l2x =
      e.val("arith.muli " + l2c + ", " + C["c2"] + " : i32");
  const std::string r4 =
      e.val("arith.shrui " + lane + ", " + C["c2"] + " : i32");
  const std::string mrow_base =
      e.val("arith.addi " + w16 + ", " + r4 + " : i32");
  const std::string m64 = e.val("arith.extsi %m : i32 to i64");
  for (int nt = 0; nt < NT; ++nt) {
    const std::string o = "%r" + std::to_string(nt);
    std::string tok = n0;
    if (nt != 0) {
      const std::string tc = e.val("arith.constant " + s(nt * 8) + " : i32");
      tok = e.val("arith.addi " + n0 + ", " + tc + " : i32");
    }
    tok = e.val("arith.addi " + tok + ", " + l2x + " : i32");
    for (int rr = 0; rr < 2; ++rr) {
      std::string mrow = mrow_base;
      if (rr != 0)
        mrow = e.val("arith.addi " + mrow_base + ", " + C["c8"] + " : i32");
      const std::string gmrow =
          e.val("arith.addi " + m0 + ", " + mrow + " : i32");
      const std::string mok =
          e.val("arith.cmpi slt, " + gmrow + ", %m : i32");
      for (int cc = 0; cc < 2; ++cc) {
        const std::string tokc =
            cc != 0 ? e.val("arith.addi " + tok + ", " + C["c1"] + " : i32")
                    : tok;
        const std::string tok_ok =
            e.val("arith.cmpi slt, " + tokc + ", %n : i32");
        const std::string ok =
            e.val("arith.andi " + mok + ", " + tok_ok + " : i1");
        e.op("scf.if " + ok + " {");
        const std::string tok64 = e.val("arith.extsi " + tokc + " : i32 to i64");
        const std::string off0 =
            e.val("arith.muli " + tok64 + ", " + m64 + " : i64");
        const std::string mrow64 =
            e.val("arith.extsi " + gmrow + " : i32 to i64");
        const std::string off =
            e.val("arith.addi " + off0 + ", " + mrow64 + " : i64");
        const std::string p =
            e.val("llvm.getelementptr %y[" + off +
                  "] : (!llvm.ptr, i64) -> !llvm.ptr, f32");
        const std::string v =
            e.val("vector.extract " + o + "[" + s(rr) + ", " + s(cc) +
                  "] : f32 from vector<2x2xf32>");
        e.op("llvm.store " + v + ", " + p + " : f32, !llvm.ptr");
        e.op("}");
      }
    }
  }
  e.op("gpu.return");
  e.op("}");

  std::string text = "module attributes {gpu.container_module} {\n";
  text += "  gpu.module @gemm [#nvvm.target<chip = \"sm_" + s(sm_arch) +
          "\">] {\n";
  text += "      " + FUNC + "\n" + e.body + "\n";
  text += "  }\n}\n";
  return text;
}

} // namespace

bool dense_gemm_shape_supported(const DenseGemmShape &shape) {
  if (shape.bm <= 0 || shape.bn <= 0 || shape.bk <= 0) return false;
  if (shape.bm % 16 != 0 || shape.bn % 8 != 0 || shape.bk % 16 != 0)
    return false;
  if (shape.bm / 16 > 8) return false;  // at most 256 threads
  if (shape.bn / 8 > 16) return false;  // at most 16 f32 accumulators/lane
  // Up to 96 KiB of dynamic shared memory; the loader raises the per-function
  // limit above 48 KiB.
  return dense_gemm_shared_bytes(shape) <= 96 * 1024;
}

int dense_gemm_block_threads(const DenseGemmShape &shape) {
  return (shape.bm / 16) * 32;
}

int dense_gemm_shared_bytes(const DenseGemmShape &shape) {
  return 2 * (shape.bm * shape.bk + shape.bn * shape.bk) * 2;
}

std::optional<std::string> generate_dense_gemm_ptx(const DenseGemmShape &shape,
                                                   int sm_arch) {
  if (!dense_gemm_shape_supported(shape)) return std::nullopt;
  const std::string text = gemm_region_text(shape, sm_arch);
  return lower_gpu_module_to_ptx(text, /*link_libdevice=*/false, "dense-gemm");
}

std::string dense_gemm_codegen_version() {
  // gemm1: first generated dense GEMM (mma m16n8k16, cp.async double buffer).
  return std::string("llvm-") + LLVM_VERSION_STRING + "-gemm1";
}

} // namespace sonicboom::compiler
