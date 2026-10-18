// Compiler-generated FlashAttention-2 kernel test for wide heads (Stage 3b).
//
// The Gemma-family attention heads are wider than a warp's mma accumulator
// budget: head_dim 256 (sliding-window head) and 512 (global head). The
// generator therefore splits head_dim across `dps` warps (256 -> 2 x 128,
// 512 -> 4 x 128), exchanges the partial S through shared memory and sums the
// slices in warp order.
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

constexpr int TOTAL = 512;
constexpr uint64_t NSLOTS = 512;

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

bool run_case(const Cfg &c, DeviceBufs &b, const std::vector<float> &qabs,
              const std::vector<float> &kseq, const std::vector<float> &vseq,
              uint64_t start, int n, uint64_t window, std::vector<float> &got) {
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
  cudaMemcpy(b.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.kb, kb.data(), kb.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.vb, vb.data(), vb.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.kc, kc.data(), kc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.vc, vc.data(), vc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.start, &start, sizeof start, cudaMemcpyHostToDevice);
  const float scale = 1.0f / std::sqrt(float(c.d));
  const bool ok = sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
      b.q, b.kb, b.vb, b.kc, b.vc, b.start, NSLOTS, n, c.nhq, c.nhkv, c.d,
      window, scale, b.out);
  if (!ok) return false;
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  got.resize(size_t(n) * c.nhq * c.d);
  cudaMemcpy(got.data(), b.out, got.size() * 4, cudaMemcpyDeviceToHost);
  return true;
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

struct ShapeCase {
  Cfg cfg;
  const char *causal_region;
  const char *window_region;
};

} // namespace

int main() {
  if (!have_cuda()) {
    std::cout << "test_cuda_attention_codegen_wide SKIP (no CUDA device)\n";
    return 0;
  }
  setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);

  const ShapeCase shapes[] = {
      {{256, 2, 4, 8}, "attn-d256-g4-causal-b32x32-dps2",
       "attn-d256-g4-window-b32x32-dps2"},
      {{512, 2, 4, 8}, "attn-d512-g4-causal-b16x32-dps4",
       "attn-d512-g4-window-b16x32-dps4"},
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
    for (const Case &cc : cases) {
      std::vector<float> got;
      check(run_case(c, b, qabs, kseq, vseq, cc.start, cc.n, cc.window, got),
            "flash_attention_chunk_dev_pos (codegen) returns true");
      if (g_failures) break;
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
        if (diff)
          std::cerr << "    d=" << c.d << " window=" << cc.window << ": "
                    << diff << "/" << base.size() << " elements differ\n";
        check(diff == 0,
              "batch invariance: row 229 alone == row 229 of the 130-row "
              "chunk (bitwise)");
      }
    }
    std::cout << "  head_dim " << c.d << " maxabs codegen=" << cg_err
              << " own=" << own_err << "\n";
    if (cg_err > own_err)
      std::cerr << "    d=" << c.d << " codegen error exceeds the "
                << "hand-written kernel's\n";
    check(cg_err <= own_err,
          "generated attention fp64 error <= hand-written kernel's (per shape)");

    const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
    if (cache_env == nullptr || std::string_view(cache_env) != "0") {
      check(codegen_cache_hit(sc.causal_region),
            "the causal wide-head kernel was compiled and cached");
      check(codegen_cache_hit(sc.window_region),
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
