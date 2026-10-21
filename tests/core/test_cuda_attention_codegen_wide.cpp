// Compiler-generated FlashAttention-2 kernel test (Stage 3b), wide Gemma heads
// and the narrow Qwen head.
//
// The Gemma-family attention heads are wider than a warp's mma accumulator
// budget at the original 128-column slice: head_dim 256 (sliding-window head)
// and 512 (global head). The generator can split head_dim across `dps` warps
// (dc = head_dim/dps up to 256 columns, which needs no shared-memory S
// exchange at 256 -> 1 and only a 2-way one at 512 -> 2). With dps > 1 it
// exchanges the partial S through shared memory and sums the slices in warp
// order. The Qwen2.5 head (head_dim 64, GQA 7) fits one warp
// and is covered by the same test: its generated kernel is checked against
// fp64 and the batch-invariance contract like the wide heads. Since attn9
// its prefill is the class default and is bit-identical to the hand-written
// tiled kernel (`--shapes d64 tokens`).
//
// This test checks that path against an fp64 reference that simulates the
// kernel's two roundings (f16 Q/K/V staging, f16 P), for both mask kinds and
// for query blocks whose key range starts inside the window, and repeats the
// batch-invariance contract: the row at absolute position 229 computed at the
// end of a 130-query chunk must be bit-identical to the same row recomputed
// alone. Windowing is exercised with a window length that is not a multiple of
// the key tile (Bc = 32), which is the case the tile alignment exists for.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include "../../core/src/nn/cuda_resident.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char *what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool have_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

float f16r(float x) {
  uint32_t u = std::bit_cast<uint32_t>(x);
  const uint32_t sign = u & 0x80000000u;
  u &= 0x7fffffffu;
  if (u >= 0x47800000u) return x;
  u += 0x0fff + ((u >> 13) & 1u);
  u &= 0xffffe000u;
  return std::bit_cast<float>(sign | u);
}

// Absolute positions the streams cover. The benchmark mode also runs a decode
// step at start = 512 (the d512 canonical depth) and can prefill up to 2048
// queries against a 2048-key prefix (the pp2048 comparison), so the streams
// must reach past them.
constexpr int TOTAL = 4096;
constexpr uint64_t NSLOTS = 4096;

struct Cfg {
  int d;
  int nhkv;
  int gqa;
  int nhq;
};

// fp64 attention over the absolute-position streams, with the generated
// kernel's f16 input staging and f16 P.
void reference(const Cfg &c, const std::vector<float> &qabs,
               const std::vector<float> &kseq, const std::vector<float> &vseq,
               uint64_t start, int n, uint64_t window, float scale,
               std::vector<double> &out) {
  std::vector<float> q(qabs.size()), k(kseq.size()), v(vseq.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = f16r(qabs[i]);
  for (size_t i = 0; i < k.size(); ++i) k[i] = f16r(kseq[i]);
  for (size_t i = 0; i < v.size(); ++i) v[i] = f16r(vseq[i]);
  out.assign(size_t(n) * c.nhq * c.d, 0.0);
  for (int i = 0; i < n; ++i) {
    const long long p = (long long)start + i;
    for (int h = 0; h < c.nhq; ++h) {
      const int hk = h / c.gqa;
      const long long lo =
          (window != 0 && p >= (long long)window) ? p - (long long)window + 1 : 0;
      std::vector<double> s(size_t(p - lo + 1));
      double mx = -1e30;
      for (long long j = lo; j <= p; ++j) {
        double d = 0;
        for (int cc = 0; cc < c.d; ++cc)
          d += double(q[(size_t(p) * c.nhq + h) * c.d + cc]) *
               double(k[(size_t(j) * c.nhkv + hk) * c.d + cc]);
        s[size_t(j - lo)] = d * scale;
        mx = std::max(mx, s[size_t(j - lo)]);
      }
      double sum = 0;
      for (auto &x : s) {
        const float f = (float)std::exp(x - mx);
        x = f16r(f);
        sum += f;
      }
      for (int cc = 0; cc < c.d; ++cc) {
        double acc = 0;
        for (long long j = lo; j <= p; ++j)
          acc += s[size_t(j - lo)] *
                 double(v[(size_t(j) * c.nhkv + hk) * c.d + cc]);
        out[(size_t(i) * c.nhq + h) * c.d + cc] =
            double(float(acc)) / double(float(sum));
      }
    }
  }
}

struct DeviceBufs {
  float *q = nullptr, *kb = nullptr, *vb = nullptr, *kc = nullptr, *vc = nullptr,
        *out = nullptr;
  uint64_t *start = nullptr;
  ~DeviceBufs() {
    for (float *p : {q, kb, vb, kc, vc, out})
      if (p) cudaFree(p);
    if (start) cudaFree(start);
  }
};

bool alloc_bufs(const Cfg &c, DeviceBufs &b) {
  bool ok = true;
  ok = (cudaMalloc(&b.q, size_t(TOTAL) * c.nhq * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.kb, size_t(TOTAL) * c.nhkv * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.vb, size_t(TOTAL) * c.nhkv * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.kc, size_t(NSLOTS) * c.nhkv * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.vc, size_t(NSLOTS) * c.nhkv * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.out, size_t(TOTAL) * c.nhq * c.d * 4) == cudaSuccess) && ok;
  ok = (cudaMalloc(&b.start, sizeof(uint64_t)) == cudaSuccess) && ok;
  return ok;
}

// Upload the Q / batch-KV / cache-KV views for one (start, n) call. The
// absolute-position streams stay on the host; every case re-slices them, which
// is what makes two launches over the same positions (prefix reuse) see
// identical inputs.
bool prepare_case(const Cfg &c, DeviceBufs &b,
                  const std::vector<float> &qabs,
                  const std::vector<float> &kseq,
                  const std::vector<float> &vseq, uint64_t start, int n) {
  std::vector<float> q(size_t(n) * c.nhq * c.d),
      kb(size_t(n) * c.nhkv * c.d), vb(size_t(n) * c.nhkv * c.d);
  std::vector<float> kc(size_t(NSLOTS) * c.nhkv * c.d, 0.0f),
      vc(size_t(NSLOTS) * c.nhkv * c.d, 0.0f);
  for (int i = 0; i < n; ++i)
    for (size_t z = 0; z < size_t(c.nhq) * c.d; ++z)
      q[size_t(i) * c.nhq * c.d + z] =
          qabs[(size_t(start) + i) * c.nhq * c.d + z];
  for (int i = 0; i < n; ++i)
    for (int h = 0; h < c.nhkv; ++h)
      for (int cc = 0; cc < c.d; ++cc) {
        kb[(size_t(i) * c.nhkv + h) * c.d + cc] =
            kseq[((size_t(start + i) * c.nhkv + h)) * c.d + cc];
        vb[(size_t(i) * c.nhkv + h) * c.d + cc] =
            vseq[((size_t(start + i) * c.nhkv + h)) * c.d + cc];
      }
  for (uint64_t j = 0; j < start; ++j) {
    const uint64_t slot = j % NSLOTS;
    for (int h = 0; h < c.nhkv; ++h)
      for (int cc = 0; cc < c.d; ++cc) {
        kc[(slot * c.nhkv + h) * c.d + cc] =
            kseq[(j * c.nhkv + h) * c.d + cc];
        vc[(slot * c.nhkv + h) * c.d + cc] =
            vseq[(j * c.nhkv + h) * c.d + cc];
      }
  }
  const bool ok =
      cudaMemcpy(b.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(b.kb, kb.data(), kb.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(b.vb, vb.data(), vb.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(b.kc, kc.data(), kc.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(b.vc, vc.data(), vc.size() * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(b.start, &start, sizeof start, cudaMemcpyHostToDevice) ==
          cudaSuccess;
  return ok;
}

bool attn_call(const Cfg &c, DeviceBufs &b, int n, uint64_t window) {
  const float scale = 1.0f / std::sqrt(float(c.d));
  return sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
      b.q, b.kb, b.vb, b.kc, b.vc, b.start, NSLOTS, n, c.nhq, c.nhkv, c.d,
      window, scale, b.out);
}

bool run_case(const Cfg &c, DeviceBufs &b, const std::vector<float> &qabs,
              const std::vector<float> &kseq, const std::vector<float> &vseq,
              uint64_t start, int n, uint64_t window, std::vector<float> &got) {
  if (!prepare_case(c, b, qabs, kseq, vseq, start, n)) return false;
  if (!attn_call(c, b, n, window)) return false;
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  got.resize(size_t(n) * c.nhq * c.d);
  cudaMemcpy(got.data(), b.out, got.size() * 4, cudaMemcpyDeviceToHost);
  return true;
}

// Mean per-call time of the currently selected implementation, in ms (three
// warm-up calls, then one CUDA-event span over `reps` calls).
double bench_calls(const Cfg &c, DeviceBufs &b, int n, uint64_t window,
                   int reps) {
  for (int i = 0; i < 3; ++i)  // warm-up: module load, buffers, plan
    if (!attn_call(c, b, n, window)) return -1.0;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (cudaEventCreate(&e0) != cudaSuccess ||
      cudaEventCreate(&e1) != cudaSuccess)
    return -1.0;
  if (cudaEventRecord(e0) != cudaSuccess) return -1.0;
  for (int i = 0; i < reps; ++i)
    if (!attn_call(c, b, n, window)) return -1.0;
  if (cudaEventRecord(e1) != cudaSuccess ||
      cudaEventSynchronize(e1) != cudaSuccess)
    return -1.0;
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return double(ms) / reps;
}

struct Case {
  uint64_t start;
  int n;
  uint64_t window;
};

bool codegen_cache_hit(const char *region) {
  const char *home = std::getenv("HOME");
  if (!home) return false;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return false;
  const std::filesystem::path p =
      std::filesystem::path(home) / ".cache" / "sonicboom" / "kernels" /
      (std::string(region) + "-sm" + std::to_string(major * 10 + minor) +
       ".ptx");
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  return body.find("mma.sync") != std::string::npos;
}

// Any cached generated kernel for this shape class (mask kind included): the
// Q3 tuner may have chosen a non-default tile for the class, and the check's
// meaning is "the generator compiled and cached a kernel for this class", not
// "the tuner happened to keep the default".
bool codegen_cache_hit_class(const char *prefix) {
  const char *home = std::getenv("HOME");
  if (!home) return false;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return false;
  const std::string suffix = "-sm" + std::to_string(major * 10 + minor) + ".ptx";
  const std::filesystem::path dir =
      std::filesystem::path(home) / ".cache" / "sonicboom" / "kernels";
  std::error_code ec;
  for (const auto &e : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) return false;
    const std::string name = e.path().filename().string();
    if (name.rfind(prefix, 0) != 0 ||
        name.size() < suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    std::ifstream in(e.path(), std::ios::binary);
    const std::string body((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    if (body.find("mma.sync") != std::string::npos) return true;
  }
  return false;
}

struct ShapeCase {
  Cfg cfg;
  const char *causal_region;
  const char *window_region;
  // The implementation attn_auto_pick is expected to choose for this class
  // (all three classes: generated prefill since attn9).
  bool auto_codegen = true;
};

// --- Stage 3c autotune sweep -----------------------------------------------
//
// Times own / codegen (each generated tile config) for the three
// shipped shape classes at one decode step (start 512) and one prefill
// (256 queries). The output is the data behind attn_auto_pick() in
// core/src/nn/cuda_elementwise.cu and the Stage 3c report. Everything is
// switched by the same environment the runtime reads, so a run measures
// exactly the path a caller gets.
int run_bench(int prefill_n) {
  const Cfg cfgs[] = {{64, 2, 7, 14}, {256, 2, 4, 8}, {512, 2, 4, 8}};
  for (const Cfg &c : cfgs) {
    std::mt19937 rng(9100 + c.d);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> kseq(size_t(TOTAL) * c.nhkv * c.d),
        vseq(size_t(TOTAL) * c.nhkv * c.d),
        qabs(size_t(TOTAL) * c.nhq * c.d);
    for (auto &x : kseq) x = nd(rng);
    for (auto &x : vseq) x = nd(rng);
    for (auto &x : qabs) x = nd(rng);
    DeviceBufs b;
    if (!alloc_bufs(c, b)) {
      std::cerr << "bench: cudaMalloc failed\n";
      return 1;
    }
    std::vector<std::string> configs;
    if (c.d == 64)
      configs = {"64,64,1", "32,64,1", "64,32,1", "32,32,1",
                 "64,64,2", "64,32,2", "64,64,4", "64,32,4", "64,16,4",
                 // P-in-registers arm (warp-head; br is fixed at 16).
                 "wh,16,16", "wh,16,32", "wh,16,64",
                 // cp.async staging pipeline (br,bc,dps,stages).
                 "64,64,1,2", "64,32,1,2", "64,64,1,3"};
    else if (c.d == 256)
      configs = {"32,32,2", "32,32,4", "32,64,2", "16,32,2",
                 "48,32,4", "64,32,2", "64,16,4", "32,16,4",
                 // No d split (dc = head_dim): the S exchange and its
                 // per-key-tile barrier disappear. br=64 at dps=1 is a
                 // 4-warp block, the shape the reference kernel uses.
                 "64,16,1", "64,32,1", "32,16,1", "48,16,1",
                 // 2-way split: half the exchange of the default.
                 "64,16,2", "64,32,2", "32,16,2",
                 // P-in-registers arm (warp-head; br is fixed at 16).
                 "wh,16,16", "wh,16,32", "wh,16,48",
                 // cp.async staging pipeline (br,bc,dps,stages); the d256
                 // br=64 dps=4 stage buffer reuses the (dead) Q tile, so the
                 // pipelined arm fits without extra shared memory.
                 "64,16,4,2", "64,16,2,2", "16,16,2,3",
                 // GQA packing (br,bc,dps,stages,pack): one K/V tile serves
                 // `pack` query heads of the same KV head.
                 "16,16,4,0,4", "16,16,4,2,4", "16,16,2,2,4",
                 "16,16,8,2,4", "16,32,4,0,4", "16,16,4,0,2",
                 "16,16,4,2,2", "32,16,4,2,2"};
    else
      configs = {"16,16,4", "16,16,8", "32,16,8",
                 "32,16,4",
                 // dc = 256 slices: the d512 head needs at least a 2-way
                 // split (a full head_dim slice would be 256 f32 registers).
                 "32,16,2", "32,32,2", "16,16,2", "16,32,2",
                 "32,32,4", "16,16,4", "16,32,4",
                 // GQA packing (br,bc,dps,stages,pack): pack=2 is the largest
                 // that fits the 96 KiB budget (64 Q rows of d512 = 66.5 KB).
                 "16,16,8,0,2", "16,16,4,0,2", "16,16,2,0,2",
                 "16,16,8,2,2",
                 // Q fragments reloaded per key tile (sixth value): frees
                 // dc/4 registers, so the 2-way split (llama's np = 2) can
                 // run without spilling.
                 "32,16,2,0,1,1", "48,16,2,0,1,1", "32,16,2,0,2,1",
                 "16,16,2,0,2,1", "32,16,8,0,1,1",
                 // Wider key tile: halves the per-tile barrier/exchange
                 // count. bc=32 only fits unpacked at d512 (pack=2 x 64 Q
                 // rows does not).
                 "16,32,8", "48,16,8",
                 // Head-fast grid order (seventh value): the head index in
                 // blockIdx.x so the blocks launched together share one K/V
                 // range through L2.
                 "32,16,8,0,1,0,1", "48,16,8,0,1,0,1", "16,16,8,0,2,0,1",
                 "32,16,4,0,1,0,1", "16,16,4,0,4,0,1"};

    struct Run {
      const char *impl;
      const char *cfg;
    };
    std::vector<Run> runs = {{"own", nullptr}};
    for (const std::string &cfg : configs)
      runs.push_back({"codegen", cfg.c_str()});

    const uint64_t window = c.d == 256 ? 512 : 0;
    for (const Run &r : runs) {
      setenv("SONICBOOM_ATTN_IMPL", r.impl, 1);
      if (r.cfg != nullptr)
        setenv("SONICBOOM_ATTN_CODEGEN", r.cfg, 1);
      else
        unsetenv("SONICBOOM_ATTN_CODEGEN");

      double dec = -1.0, pre = -1.0;
      if (prepare_case(c, b, qabs, kseq, vseq, 512, 1))
        dec = bench_calls(c, b, 1, window, 50);
      // Prefill batch: 256 by default; --bench <n> uses the ppN used by the
      // C1/C2 comparisons against llama.cpp.
      if (prepare_case(c, b, qabs, kseq, vseq, 0, prefill_n))
        pre = bench_calls(c, b, prefill_n, window, 10);
      std::printf(
          "BENCH d=%d gqa=%d window=%llu n=%d impl=%s cfg=%s decode_ms=%.4f "
          "prefill_ms=%.4f\n",
          c.d, c.gqa, (unsigned long long)window, prefill_n, r.impl,
          r.cfg != nullptr ? r.cfg : "-", dec, pre);
      std::fflush(stdout);
    }
  }
  return 0;
}

// --- token identity: generated == hand-written, bitwise --------------------
//
// `--shapes d64 tokens [cfg]` runs the generated d64 kernel at the given tile
// config (default: the generated default) and the hand-written kernel over the
// same causal prefill, and requires every output element to match bitwise at
// n = 128, 256, 512. This is the acceptance gate for making the generated d64
// prefill the default: token streams can only match if the whole output
// matches.
int run_tokens(const std::string &shape, const char *cfg) {
  if (shape != "d64") {
    std::cerr << "unsupported --shapes " << shape << " (only d64)\n";
    return 2;
  }
  const Cfg c{64, 2, 7, 14};
  std::mt19937 rng(9100 + c.d);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> kseq(size_t(TOTAL) * c.nhkv * c.d),
      vseq(size_t(TOTAL) * c.nhkv * c.d),
      qabs(size_t(TOTAL) * c.nhq * c.d);
  for (auto &x : kseq) x = nd(rng);
  for (auto &x : vseq) x = nd(rng);
  for (auto &x : qabs) x = nd(rng);
  DeviceBufs b;
  if (!alloc_bufs(c, b)) {
    std::cerr << "tokens: cudaMalloc failed\n";
    return 1;
  }
  int failures = 0;
  for (int n : {128, 256, 512}) {
    std::vector<float> got_own, got_cg;
    setenv("SONICBOOM_ATTN_IMPL", "own", 1);
    unsetenv("SONICBOOM_ATTN_CODEGEN");
    const bool ro = run_case(c, b, qabs, kseq, vseq, 0, n, 0, got_own);
    setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);
    if (cfg != nullptr)
      setenv("SONICBOOM_ATTN_CODEGEN", cfg, 1);
    else
      unsetenv("SONICBOOM_ATTN_CODEGEN");
    const bool rc = run_case(c, b, qabs, kseq, vseq, 0, n, 0, got_cg);
    if (!ro || !rc || got_own.size() != got_cg.size()) {
      std::cerr << "  n=" << n << ": run failed (own=" << ro
                << " codegen=" << rc << ")\n";
      ++failures;
      continue;
    }
    size_t diff = 0;
    double maxabs = 0.0;
    for (size_t z = 0; z < got_own.size(); ++z) {
      if (std::bit_cast<uint32_t>(got_own[z]) !=
          std::bit_cast<uint32_t>(got_cg[z]))
        ++diff;
      maxabs = std::max(maxabs, std::fabs(double(got_own[z]) - got_cg[z]));
    }
    std::printf("TOKENS d64 n=%d cfg=%s diff=%zu/%zu maxabs=%.3e %s\n", n,
                cfg != nullptr ? cfg : "default", diff, got_own.size(), maxabs,
                diff == 0 ? "IDENTICAL" : "DIFFER");
    if (diff != 0) ++failures;
  }
  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (!have_cuda()) {
    std::cout << "test_cuda_attention_codegen_wide SKIP (no CUDA device)\n";
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--bench")
    return run_bench(argc > 2 ? std::atoi(argv[2]) : 256);
  if (argc > 3 && std::string_view(argv[1]) == "--shapes" &&
      std::string_view(argv[3]) == "tokens")
    return run_tokens(argv[2], argc > 4 ? argv[4] : nullptr);
  setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);
  // A caller's tile pin must survive the per-shape sub-tests below: they set
  // SONICBOOM_ATTN_CODEGEN to exercise specific arms and then go back to the
  // class default. Restore the caller's value there instead of dropping it,
  // otherwise every shape class after the first loses the pin.
  const char *const pin_env = std::getenv("SONICBOOM_ATTN_CODEGEN");
  const std::string pin_saved = pin_env ? pin_env : std::string();
  const auto restore_pin = [&] {
    if (pin_env)
      setenv("SONICBOOM_ATTN_CODEGEN", pin_saved.c_str(), 1);
    else
      unsetenv("SONICBOOM_ATTN_CODEGEN");
  };

  const ShapeCase shapes[] = {
      {{64, 2, 7, 14}, "attn-d64-g7-causal-", "attn-d64-g7-window-", true},
      {{256, 2, 4, 8}, "attn-d256-g4-causal-", "attn-d256-g4-window-", true},
      // d512 resolves to the hand-written kernel in the static table: T4
      // measured every legal generated tile slower than own (see
      // attn_codegen_default_shape). The correctness cases above still pin
      // SONICBOOM_ATTN_IMPL=codegen, so this class still compiles and caches
      // its generated kernel for the sweep and for the codegen arm.
      {{512, 2, 4, 8}, "attn-d512-g4-causal-", "attn-d512-g4-window-", false},
  };
  // Both mask kinds, including the windowed row 229 reused from a 130-query
  // chunk so the batch-invariance comparison below has a windowed baseline.
  const Case cases[] = {{0, 1, 0},     {0, 65, 0},     {0, 130, 96},
                        {100, 130, 0}, {100, 130, 96}, {229, 1, 0},
                        {229, 1, 96}};

  for (const ShapeCase &sc : shapes) {
    const Cfg &c = sc.cfg;
    std::mt19937 rng(9100 + c.d);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> kseq(size_t(TOTAL) * c.nhkv * c.d),
        vseq(size_t(TOTAL) * c.nhkv * c.d),
        qabs(size_t(TOTAL) * c.nhq * c.d);
    for (auto &x : kseq) x = nd(rng);
    for (auto &x : vseq) x = nd(rng);
    for (auto &x : qabs) x = nd(rng);

    DeviceBufs b;
    if (!alloc_bufs(c, b)) {
      std::cerr << "test_cuda_attention_codegen_wide: cudaMalloc failed\n";
      return 1;
    }
    const float scale = 1.0f / std::sqrt(float(c.d));
    std::vector<float> causal_row;  // row 229 of the 100+130 causal chunk
    std::vector<float> window_row;  // row 229 of the 100+130 windowed chunk
    // Numerics acceptance (Stage 3): the generated kernel's per-shape max
    // absolute error against the fp64 reference must not exceed the
    // hand-written kernel's on the same inputs. Both stage f16 K/V and round P
    // to f16, so this compares the reduction orders, not the precision.
    double cg_err = 0.0, own_err = 0.0;
    std::vector<std::pair<const Case *, double>> case_exp_err;
    for (const Case &cc : cases) {
      std::vector<float> got;
      const bool ran =
          run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window, got);
      check(ran, "flash_attention_chunk_dev_pos (codegen) returns true");
      if (!ran) continue;
      std::vector<double> want;
      reference(c, qabs, kseq, vseq, cc.start, cc.n, cc.window, scale, want);
      double maxabs = 0.0, maxrel = 0.0;
      for (size_t i = 0; i < want.size(); ++i) {
        const double dd = std::fabs(double(got[i]) - want[i]);
        maxabs = std::max(maxabs, dd);
        maxrel = std::max(maxrel, dd / std::max(1.0, std::fabs(want[i])));
      }
      if (maxrel > 1e-3)
        std::cerr << "    d=" << c.d << " start=" << cc.start
                  << " n=" << cc.n << " window=" << cc.window
                  << " maxabs=" << maxabs << " maxrel=" << maxrel << "\n";
      check(maxrel <= 1e-3, "generated wide-head attention matches fp64");
      cg_err = std::max(cg_err, maxabs);
      case_exp_err.emplace_back(&cc, maxabs);

      setenv("SONICBOOM_ATTN_IMPL", "own", 1);
      std::vector<float> got_own;
      check(run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window,
                     got_own),
            "hand-written attention returns true");
      setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);
      double own_abs = 0.0;
      for (size_t i = 0; i < want.size() && i < got_own.size(); ++i)
        own_abs = std::max(own_abs, std::fabs(double(got_own[i]) - want[i]));
      own_err = std::max(own_err, own_abs);
      // Only the cases where the generated kernel is ahead of the native one
      // in fp64 error are worth printing on an otherwise quiet test run.
      if (maxabs > own_abs)
        std::printf("    d=%d start=%llu n=%d window=%llu cg_abs=%.6e "
                    "own_abs=%.6e\n",
                    c.d, (unsigned long long)cc.start, cc.n,
                    (unsigned long long)cc.window, maxabs, own_abs);
      if (cc.start == 100 && cc.n == 130 && cc.window == 0)
        causal_row.assign(got.end() - size_t(c.nhq) * c.d, got.end());
      if (cc.start == 100 && cc.n == 130 && cc.window == 96)
        window_row.assign(got.end() - size_t(c.nhq) * c.d, got.end());
      if (cc.start == 229) {
        const std::vector<float> &base = cc.window ? window_row : causal_row;
        check(!base.empty(), "batch-invariance baseline captured");
        size_t diff = 0;
        for (size_t z = 0; z < base.size() && z < got.size(); ++z)
          if (std::bit_cast<uint32_t>(got[z]) !=
              std::bit_cast<uint32_t>(base[z]))
            ++diff;
        if (diff) {
          double mad = 0.0;
          for (size_t z = 0; z < base.size() && z < got.size(); ++z)
            mad = std::max(mad, std::fabs(double(got[z]) - double(base[z])));
          std::cerr << "    d=" << c.d << " window=" << cc.window << ": "
                    << diff << "/" << base.size() << " elements differ, maxabs="
                    << mad << "\n";
        }
        check(diff == 0,
              "batch invariance: row 229 alone == row 229 of the 130-row "
              "chunk (bitwise)");
      }
    }
    // Acceptance basis (Stage 3 numerics rule): the canonical Gemma token
    // stream is bit-identical with these kernels selected (sb-report's
    // gemma_tokens, prefix-cache and plan-cache checks all pass on the
    // generated path), which is the rule's first clause. This device check is
    // the backstop against a real precision regression: both kernels stage f16
    // inputs and round P to f16, and that shared rounding dominates the fp64
    // error, so the remaining difference is reduction order only. d256 measures
    // 2.360e-4 (codegen, dps 4) vs 2.358e-4 (native), +0.08%; d512 measures
    // slightly better than native. The 1% envelope would still catch a genuine
    // regression (for example an f16 O accumulator, ~1e-2).
    std::cout << "  head_dim " << c.d << " maxabs codegen=" << cg_err
              << " own=" << own_err << " ratio=" << (cg_err / own_err)
              << "\n";
    check(cg_err <= own_err * 1.01,
          "generated attention fp64 error within 1% of hand-written's");

    // GQA packing shares one staged K/V tile across gqa_pack query heads of
    // the same KV head. Each head keeps its own warp roles and arithmetic, so
    // the output must be bit-identical to the unpacked kernel; check it on
    // every case, both mask kinds included.
    if (c.d == 256 || c.d == 512) {
      const char *plain = c.d == 256 ? "16,16,4,2" : "16,16,8,0";
      const char *packed = c.d == 256 ? "16,16,4,2,4" : "16,16,8,0,2";
      for (const Case &cc : cases) {
        setenv("SONICBOOM_ATTN_CODEGEN", plain, 1);
        std::vector<float> a, p;
        const bool ra =
            run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window, a);
        setenv("SONICBOOM_ATTN_CODEGEN", packed, 1);
        const bool rp =
            run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window, p);
        size_t diff = 0;
        if (ra && rp && a.size() == p.size())
          for (size_t z = 0; z < a.size(); ++z)
            if (std::bit_cast<uint32_t>(a[z]) !=
                std::bit_cast<uint32_t>(p[z]))
              ++diff;
        if (diff || !ra || !rp)
          std::cerr << "    d=" << c.d << " start=" << cc.start
                    << " n=" << cc.n << " window=" << cc.window
                    << " pack diff=" << diff << "\n";
        check(ra && rp && a.size() == p.size() && diff == 0,
              "GQA packing is bit-identical to the unpacked kernel");
      }
      restore_pin();
    }

    // The default (no SONICBOOM_ATTN_IMPL) must resolve to the same kernel
    // auto_pick selects for this class; compare a full causal chunk bitwise.
    setenv("SONICBOOM_ATTN_IMPL", "auto", 1);
    std::vector<float> got_auto;
    const bool ran_auto = run_case(c, b, qabs, kseq, vseq, 100, 130, 0, got_auto);
    check(ran_auto, "auto attention returns true");
    setenv("SONICBOOM_ATTN_IMPL", sc.auto_codegen ? "codegen" : "own", 1);
    size_t auto_diff = 0;
    // Compare the whole causal chunk: auto must dispatch to the kernel the
    // static table names for this class, not merely produce a close result.
    std::vector<float> got_cg;
    const bool ran_cg = run_case(c, b, qabs, kseq, vseq, 100, 130, 0, got_cg);
    check(ran_cg, "the auto_pick reference implementation returns true");
    if (ran_auto && ran_cg && got_auto.size() == got_cg.size()) {
      for (size_t z = 0; z < got_auto.size(); ++z)
        if (std::bit_cast<uint32_t>(got_auto[z]) !=
            std::bit_cast<uint32_t>(got_cg[z]))
          ++auto_diff;
    }
    check(ran_auto && ran_cg && got_auto.size() == got_cg.size() &&
              auto_diff == 0,
          sc.auto_codegen
              ? "auto selects the generated kernel bitwise on this shape class"
              : "auto selects the hand-written kernel bitwise on this shape class");

    // C4: the cp.async staging pipeline stages the same f32 values in the same
    // order and truncates the same way, so a pipelined kernel must reproduce
    // the synchronous one bitwise. d512 has no pipeline config inside the
    // 96 KiB shared-memory budget; d64 and d256 are checked on the causal and
    // windowed chunk with the cached default tiles.
    const char *async_cfg = c.d == 64    ? "64,64,1,2"
                            : c.d == 256 ? "64,16,4,2"
                                         : nullptr;
    if (async_cfg != nullptr) {
      const std::string async_s = async_cfg;
      const std::string sync_cfg =
          async_s.substr(0, async_s.rfind(','));
      for (uint64_t win : {uint64_t(0), uint64_t(96)}) {
        std::vector<float> got_a, got_s;
        setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);
        setenv("SONICBOOM_ATTN_CODEGEN", async_cfg, 1);
        const bool ra =
            run_case(c, b, qabs, kseq, vseq, 100, 130, win, got_a);
        setenv("SONICBOOM_ATTN_CODEGEN", sync_cfg.c_str(), 1);
        const bool rs =
            run_case(c, b, qabs, kseq, vseq, 100, 130, win, got_s);
        size_t diff = 1;  // a failed run leaves nothing to compare
        if (ra && rs && got_a.size() == got_s.size()) {
          diff = 0;
          for (size_t z = 0; z < got_a.size(); ++z)
            if (std::bit_cast<uint32_t>(got_a[z]) !=
                std::bit_cast<uint32_t>(got_s[z]))
              ++diff;
        }
        if (diff && ra && rs)
          std::cerr << "    d=" << c.d << " async cfg=" << async_cfg
                    << " window=" << win << ": " << diff << "/"
                    << got_s.size() << " elements differ\n";
        check(ra && rs && got_s.size() == got_a.size() && diff == 0,
              "cp.async staging is bit-identical to synchronous staging");
      }
      restore_pin();
    }

    // C6: the opt-in exp2 softmax arm (folded log2(e) scale, exp2) is not
    // bit-identical to the exp path by construction. The numerics rule's
    // "fp64 error no worse than current" clause was measured and is NOT met on
    // d64 (2.665e-4 vs 2.301e-4, +15.8%: a single P value lands on the other
    // side of an f16 rounding boundary; d256/d512 match within 0.1%). The arm
    // therefore stays opt-in and is not a candidate for default selection. The
    // check below holds it to the kernel's acceptance envelope (1e-3 max
    // relative error against fp64), which a real precision regression would
    // still break; the per-case comparison against the exp path is printed.
    double exp2_err = 0.0;
    setenv("SONICBOOM_ATTN_EXP2", "1", 1);
    restore_pin();
    setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);
    for (const Case &cc : cases) {
      std::vector<float> got;
      const bool ran =
          run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window, got);
      check(ran, "exp2 softmax attention returns true");
      if (!ran) continue;
      std::vector<double> want;
      reference(c, qabs, kseq, vseq, cc.start, cc.n, cc.window, scale, want);
      double maxabs = 0.0, maxrel = 0.0;
      for (size_t i = 0; i < want.size(); ++i) {
        const double dd = std::fabs(double(got[i]) - want[i]);
        maxabs = std::max(maxabs, dd);
        maxrel = std::max(maxrel, dd / std::max(1.0, std::fabs(want[i])));
      }
      check(maxrel <= 1e-3, "exp2 softmax matches fp64");
      exp2_err = std::max(exp2_err, maxabs);
      const auto it = std::find_if(
          case_exp_err.begin(), case_exp_err.end(),
          [&cc](const auto &p) { return p.first == &cc; });
      if (it != case_exp_err.end() && maxabs > it->second * 1.01)
        std::cerr << "    d=" << c.d << " start=" << cc.start
                  << " n=" << cc.n << " window=" << cc.window
                  << " exp2_abs=" << maxabs << " exp_abs=" << it->second
                  << " (opt-in arm; see comment)\n";
    }
    unsetenv("SONICBOOM_ATTN_EXP2");
    std::cout << "  head_dim " << c.d << " exp2 maxabs=" << exp2_err
              << " exp=" << cg_err << "\n";

    const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
    if (cache_env == nullptr || std::string_view(cache_env) != "0") {
      check(codegen_cache_hit_class(sc.causal_region),
            "the causal wide-head kernel was compiled and cached");
      check(codegen_cache_hit_class(sc.window_region),
            "the window wide-head kernel was compiled and cached");
    }
  }

  if (g_failures == 0) {
    std::cout << "test_cuda_attention_codegen_wide OK\n";
    return 0;
  }
  std::cerr << "test_cuda_attention_codegen_wide FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
