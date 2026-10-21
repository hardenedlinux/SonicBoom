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

// Q3 prefill attention autotune.
//
// Model-free: the shape-class key (head_dim, seq_len bucket, query-head
// bucket), the (br, bc, as) grid around a production config (production
// first), the tunable head_dims, and the gate/cache behaviour of
// attn_prefill_choice.
//
// On a GPU: the generated d64/gqa7 prefill kernel is tuned in-process under
// SONICBOOM_AUTOTUNE=1; the emitted output must be bit-identical to the
// production (64,16,1,as2) pinned reference, the winner must land in the
// per-SM cache under "attn_prefill", and three cold runs (memo + cache
// cleared, AUTOTUNE=force) must pick the same config. Skips the GPU part
// cleanly when no device is present.

#include "../../core/src/nn/cuda_resident.h"
#include "../../core/src/planner/mlir_autotune.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace mt = sonicboom::mlir_tune;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::cout << "FAIL: " << what << "\n";
  }
}

void set_env(const char *name, const char *value) {
  if (value)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
}

bool have_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool in_grid(const std::vector<mt::TileConfig> &grid,
             const mt::TileConfig &c) {
  for (const mt::TileConfig &g : grid)
    if (g == c) return true;
  return false;
}

} // namespace

int main() {
  // Every cache operation goes to a throwaway directory.
  char tmpl[] = "/tmp/sb-q3-test-XXXXXX";
  const char *dir = ::mkdtemp(tmpl);
  if (!dir) {
    std::cout << "FAIL: mkdtemp\n";
    return 1;
  }
  set_env("XDG_CACHE_HOME", dir);
  set_env("SONICBOOM_AUTOTUNE", "0");
  // The GPU section forces several cold retunes; lift the production cold
  // budget so they are not cut short.
  set_env("SONICBOOM_AUTOTUNE_BUDGET_MS", "0");
  mt::AutotuneCache::instance().reset_for_test();
  mt::attn_prefill_reset_for_test();

  // --- the shape-class key -----------------------------------------------
  {
    const struct {
      int seq_len;
      const char *want;
    } seq[] = {{1, "1-63"},     {63, "1-63"},   {64, "64-127"},
               {127, "64-127"}, {128, "128-255"}, {255, "128-255"},
               {256, "256-511"}, {511, "256-511"}, {512, "512-1023"},
               {1023, "512-1023"}, {1024, "1024+"}, {8192, "1024+"}};
    for (const auto &c : seq)
      check(mt::attn_prefill_seq_bucket(c.seq_len) == c.want,
            "seq bucket for " + std::to_string(c.seq_len) + " is " + c.want);

    check(mt::attn_prefill_heads_bucket(1) == "1-7" &&
              mt::attn_prefill_heads_bucket(7) == "1-7",
          "1..7 query heads bucket to 1-7");
    check(mt::attn_prefill_heads_bucket(8) == "8-15" &&
              mt::attn_prefill_heads_bucket(15) == "8-15",
          "8..15 query heads bucket to 8-15");
    check(mt::attn_prefill_heads_bucket(16) == "16+" &&
              mt::attn_prefill_heads_bucket(32) == "16+",
          "16+ query heads bucket to 16+");

    check(mt::attn_prefill_shape_class(64, 512, 14) == "64/512-1023/8-15",
          "shape class is <head_dim>/<seq bucket>/<heads bucket>");
  }

  // --- candidates: production first, real (br, bc, as) neighbours --------
  {
    const mt::TileConfig prod{64, 64, 16, 2, 1};  // Qwen d64: 64,16,1,as2
    const std::vector<mt::TileConfig> grid =
        mt::attn_prefill_candidates(64, prod);
    check(!grid.empty() && grid.front() == prod,
          "the first candidate is the production config");
    for (int br : {32, 64})
      for (int bc : {16, 32})
        for (int as : {0, 2})
          check(in_grid(grid, mt::TileConfig{br, 64, bc, as, 1}),
                "candidate (br=" + std::to_string(br) + ",bc=" +
                    std::to_string(bc) + ",as=" + std::to_string(as) +
                    ") is present");
    check(!in_grid(grid, mt::TileConfig{64, 32, 16, 2, 1}),
          "dps is not varied (dc stays at head_dim/dps)");

    const mt::TileConfig gemma{64, 64, 16, 0, 1};  // d256: 64,16,dps4,as0
    const std::vector<mt::TileConfig> ggrid =
        mt::attn_prefill_candidates(256, gemma);
    check(ggrid.front() == gemma, "d256 production is the reference");
    check(in_grid(ggrid, mt::TileConfig{32, 64, 16, 2, 1}),
          "d256 grid spans br/as");

    const mt::TileConfig global{32, 64, 16, 0, 1};  // d512: 32,16,dps8,as0
    const std::vector<mt::TileConfig> wgrid =
        mt::attn_prefill_candidates(512, global);
    check(wgrid.front() == global && in_grid(wgrid, mt::TileConfig{16, 64, 16, 2, 1}),
          "d512 grid includes br=16 and production first");
  }

  // --- which head dims the grid covers -----------------------------------
  {
    check(mt::attn_prefill_tunable(64) && mt::attn_prefill_tunable(128) &&
              mt::attn_prefill_tunable(32) && mt::attn_prefill_tunable(256) &&
              mt::attn_prefill_tunable(512),
          "head_dim 32/64/128/256/512 have Q3 classes");
    check(!mt::attn_prefill_tunable(96) && !mt::attn_prefill_tunable(0),
          "unsupported head dims have no Q3 class");
  }

  // --- gate and cache ----------------------------------------------------
  {
    mt::AttnPrefillChoice off = mt::attn_prefill_choice(128, 200, 14);
    check(!off.resolved(), "gate off: the class is unresolved");

    set_env("SONICBOOM_AUTOTUNE", "1");
    mt::attn_prefill_reset_for_test();
    mt::AttnPrefillChoice cold = mt::attn_prefill_choice(128, 200, 14);
    check(!cold.resolved(), "gate on, cold cache: the class is unresolved");

    const mt::TileConfig winner{64, 64, 32, 2, 1};
    check(mt::attn_prefill_choice_store(128, 200, 14, winner),
          "the winner is persisted");
    mt::attn_prefill_reset_for_test();
    mt::AttnPrefillChoice warm = mt::attn_prefill_choice(128, 200, 14);
    check(warm.resolved() && warm.cfg == winner &&
              warm.source == mt::AttnPrefillSource::Cache,
          "a second process reads the winner from the per-SM cache");
    const std::string cls = mt::attn_prefill_shape_class(128, 200, 14);
    check(mt::AutotuneCache::instance().lookup("attn_prefill", cls) == winner,
          "the cache record lives under op \"attn_prefill\"");
    set_env("SONICBOOM_AUTOTUNE", nullptr);
  }

  // --- the GPU end-to-end: tuned output is the (16,16) reference ---------
  if (have_cuda()) {
    constexpr int D = 64;
    constexpr int GQA = 7;
    constexpr int NHKV = 2;
    constexpr int NHQ = NHKV * GQA;
    constexpr int N = 128;
    constexpr uint64_t NSLOTS = 256;
    constexpr float SCALE = 1.0f / 8.0f;  // 1/sqrt(64)

    const size_t q_n = size_t(N) * NHQ * D;
    const size_t kv_n = size_t(N) * NHKV * D;
    const size_t c_n = size_t(NSLOTS) * NHKV * D;
    std::vector<float> hq(q_n), hk(kv_n), hv(kv_n);
    uint32_t seed = 12345;
    auto next = [&] {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      return float(seed % 2000) / 1000.0f - 1.0f;
    };
    for (float &x : hq) x = next();
    for (float &x : hk) x = next();
    for (float &x : hv) x = next();

    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dkc = nullptr,
          *dvc = nullptr, *dout = nullptr;
    uint64_t *dstart = nullptr;
    bool ok = cudaMalloc(reinterpret_cast<void **>(&dq), q_n * 4) == cudaSuccess;
    ok = ok &&
         cudaMalloc(reinterpret_cast<void **>(&dk), kv_n * 4) == cudaSuccess;
    ok = ok &&
         cudaMalloc(reinterpret_cast<void **>(&dv), kv_n * 4) == cudaSuccess;
    ok = ok &&
         cudaMalloc(reinterpret_cast<void **>(&dkc), c_n * 4) == cudaSuccess;
    ok = ok &&
         cudaMalloc(reinterpret_cast<void **>(&dvc), c_n * 4) == cudaSuccess;
    ok = ok &&
         cudaMalloc(reinterpret_cast<void **>(&dout), q_n * 4) == cudaSuccess;
    ok = ok && cudaMalloc(reinterpret_cast<void **>(&dstart),
                          sizeof(uint64_t)) == cudaSuccess;
    check(ok, "GPU buffers allocate");
    if (ok) {
      cudaMemcpy(dq, hq.data(), q_n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dk, hk.data(), kv_n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dv, hv.data(), kv_n * 4, cudaMemcpyHostToDevice);
      cudaMemset(dkc, 0, c_n * 4);
      cudaMemset(dvc, 0, c_n * 4);
      const uint64_t zero = 0;
      cudaMemcpy(dstart, &zero, sizeof(zero), cudaMemcpyHostToDevice);

      auto run = [&](std::vector<float> &out) {
        const bool ran = sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
            dq, dk, dv, dkc, dvc, dstart, NSLOTS, N, NHQ, NHKV, D,
            /*sliding_window=*/0, SCALE, dout);
        cudaDeviceSynchronize();
        out.resize(q_n);
        cudaMemcpy(out.data(), dout, q_n * 4, cudaMemcpyDeviceToHost);
        return ran;
      };

      // Reference: the production (br,bc,dps,as) = (64,16,1,2) pinned; Q3 is
      // bypassed by the pin. The tuner's reference is the same config.
      set_env("SONICBOOM_ATTN_CODEGEN", "64,16,1,2");
      set_env("SONICBOOM_ATTN_IMPL", nullptr);
      set_env("SONICBOOM_AUTOTUNE", "0");
      std::vector<float> ref;
      const bool ref_ok = run(ref);
      check(ref_ok, "the production-config reference runs");

      // Q3: the generated kernel with the autotuner choosing the tile.
      set_env("SONICBOOM_ATTN_CODEGEN", nullptr);
      set_env("SONICBOOM_ATTN_IMPL", "codegen");
      set_env("SONICBOOM_AUTOTUNE", "1");
      set_env("SONICBOOM_AUTOTUNE_VERBOSE", "1");
      mt::AutotuneCache::instance().reset_for_test();
      mt::attn_prefill_reset_for_test();
      std::vector<float> tuned;
      const bool tuned_ok = run(tuned);
      check(tuned_ok, "the Q3-tuned generated kernel runs");
      check(ref_ok && tuned_ok && !ref.empty() && !tuned.empty() &&
                std::memcmp(ref.data(), tuned.data(), q_n * 4) == 0,
            "the Q3 pick is bit-identical to the production reference");

      const std::string cls = mt::attn_prefill_shape_class(D, N, NHQ);
      const std::vector<mt::TileConfig> grid = mt::attn_prefill_candidates(
          D, mt::TileConfig{64, 64, 16, 2, 1});
      std::optional<mt::TileConfig> rec =
          mt::AutotuneCache::instance().lookup("attn_prefill", cls);
      check(rec.has_value() && in_grid(grid, *rec),
            "the winner is a grid config in the attn_prefill class " + cls);

      // Warm cache: three cold runs (memo and disk cache cleared, forced
      // retune) pick the same config.
      std::vector<mt::TileConfig> picks;
      for (int i = 0; i < 3; ++i) {
        set_env("SONICBOOM_AUTOTUNE", "force");
        mt::AutotuneCache::instance().reset_for_test();
        mt::attn_prefill_reset_for_test();
        std::vector<float> out;
        const bool ran = run(out);
        check(ran, "cold run " + std::to_string(i) + " runs");
        check(!out.empty() && !ref.empty() &&
                  std::memcmp(ref.data(), out.data(), q_n * 4) == 0,
              "cold run " + std::to_string(i) +
                  " stays bit-identical to the reference");
        const auto r = mt::AutotuneCache::instance().lookup("attn_prefill", cls);
        check(r.has_value(), "cold run " + std::to_string(i) + " stores a pick");
        if (r) picks.push_back(*r);
      }
      bool same = picks.size() == 3;
      for (size_t i = 1; same && i < picks.size(); ++i)
        same = picks[i] == picks[0];
      check(same, "three cold runs pick the same (br, bc, as)");
      if (!picks.empty())
        std::printf("  Q3 winner %s: br=%d dc=%d bc=%d as=%d\n", cls.c_str(),
                    picks[0].tile_m, picks[0].tile_n, picks[0].tile_k,
                    picks[0].unroll);
    }

    for (float *p : {dq, dk, dv, dkc, dvc, dout}) cudaFree(p);
    cudaFree(dstart);
  } else {
    std::cout << "  no CUDA device: the tuned-kernel half is skipped\n";
  }

  set_env("SONICBOOM_AUTOTUNE", nullptr);
  set_env("SONICBOOM_AUTOTUNE_VERBOSE", nullptr);
  set_env("SONICBOOM_ATTN_IMPL", nullptr);
  set_env("SONICBOOM_ATTN_CODEGEN", nullptr);
  set_env("SONICBOOM_ATTN_PREFILL", nullptr);

  if (failures == 0) {
    std::cout << "test_attn_prefill_autotune OK\n";
    return 0;
  }
  std::cout << "test_attn_prefill_autotune FAILED: " << failures
            << " check(s)\n";
  return 1;
}
