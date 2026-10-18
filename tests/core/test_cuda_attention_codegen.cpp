// Compiler-generated FlashAttention-2 kernel test (Stage 3a).
//
// SONICBOOM_ATTN_IMPL=codegen routes nn::cuda::flash_attention_chunk_dev_pos
// through the MLIR-generated FA-2 kernel (compiler/AttentionCodegen.h). The
// generated kernel stages f16 copies of Q/K/V and rounds P to f16 before the PV
// tensor-core product, so the fp64 reference below simulates exactly those two
// roundings; the remaining difference is the f32 exp (ex2.approx) against the
// reference's double exp, well below the tolerance.
//
// Shape: head_dim 64, gqa 7 (Qwen2.5), causal and sliding-window masks. Cases
// cover a single query, a key sequence split across the cache and the batch,
// the second and third query blocks (partial tail tiles), and the window's
// fully-masked leading tiles. The last case is the batch-invariance contract:
// the row at absolute position 229 computed at the end of a 130-query chunk
// (cache + batch) must be bit-identical to the same row recomputed alone
// (cache only, a different block and lane in the grid).
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

// Round to f16 (nearest even) through the f32 bit pattern. The test data is
// O(1) normal numbers, so the exponent range and subnormals do not matter.
float f16r(float x) {
  uint32_t u = std::bit_cast<uint32_t>(x);
  const uint32_t sign = u & 0x80000000u;
  u &= 0x7fffffffu;
  if (u >= 0x47800000u) return x;  // overflow: out of f16 range, leave as-is
  u += 0x0fff + ((u >> 13) & 1u);
  u &= 0xffffe000u;
  return std::bit_cast<float>(sign | u);
}

constexpr int D = 64;
constexpr int NHKV = 2;
constexpr int GQA = 7;
constexpr int NHQ = NHKV * GQA;
constexpr int TOTAL = 4096;
constexpr uint64_t NSLOTS = 512;

// fp64 attention over the absolute-position streams, with the generated
// kernel's f16 input staging and f16 P.
void reference(const std::vector<float> &qabs, const std::vector<float> &kseq,
               const std::vector<float> &vseq, uint64_t start, int n,
               uint64_t window, float scale, std::vector<double> &out) {
  std::vector<float> q(qabs.size()), k(kseq.size()), v(vseq.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = f16r(qabs[i]);
  for (size_t i = 0; i < k.size(); ++i) k[i] = f16r(kseq[i]);
  for (size_t i = 0; i < v.size(); ++i) v[i] = f16r(vseq[i]);
  out.assign(size_t(n) * NHQ * D, 0.0);
  for (int i = 0; i < n; ++i) {
    const long long p = (long long)start + i;
    for (int h = 0; h < NHQ; ++h) {
      const int hk = h / GQA;
      const long long lo =
          (window != 0 && p >= (long long)window) ? p - (long long)window + 1 : 0;
      std::vector<double> s(size_t(p - lo + 1));
      double mx = -1e30;
      for (long long j = lo; j <= p; ++j) {
        double d = 0;
        for (int c = 0; c < D; ++c)
          d += double(q[(size_t(p) * NHQ + h) * D + c]) *
               double(k[(size_t(j) * NHKV + hk) * D + c]);
        s[size_t(j - lo)] = d * scale;
        mx = std::max(mx, s[size_t(j - lo)]);
      }
      double sum = 0;
      for (auto &x : s) {
        const float f = (float)std::exp(x - mx);
        x = f16r(f);  // O uses the f16 P; l accumulates the f32 value
        sum += f;
      }
      for (int c = 0; c < D; ++c) {
        double acc = 0;
        for (long long j = lo; j <= p; ++j)
          acc += s[size_t(j - lo)] *
                 double(v[(size_t(j) * NHKV + hk) * D + c]);
        out[(size_t(i) * NHQ + h) * D + c] =
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

bool run_case(DeviceBufs &b, const std::vector<float> &qabs,
              const std::vector<float> &kseq, const std::vector<float> &vseq,
              uint64_t start, int n, uint64_t window, std::vector<float> &got) {
  std::vector<float> q(size_t(n) * NHQ * D), kb(size_t(n) * NHKV * D),
      vb(size_t(n) * NHKV * D);
  std::vector<float> kc(size_t(NSLOTS) * NHKV * D, 0.0f),
      vc(size_t(NSLOTS) * NHKV * D, 0.0f);
  for (int i = 0; i < n; ++i)
    for (size_t z = 0; z < size_t(NHQ) * D; ++z)
      q[size_t(i) * NHQ * D + z] = qabs[(size_t(start) + i) * NHQ * D + z];
  for (int i = 0; i < n; ++i)
    for (int h = 0; h < NHKV; ++h)
      for (int c = 0; c < D; ++c) {
        kb[(size_t(i) * NHKV + h) * D + c] =
            kseq[((size_t(start + i) * NHKV + h)) * D + c];
        vb[(size_t(i) * NHKV + h) * D + c] =
            vseq[((size_t(start + i) * NHKV + h)) * D + c];
      }
  for (uint64_t j = 0; j < start; ++j) {
    const uint64_t slot = j % NSLOTS;
    for (int h = 0; h < NHKV; ++h)
      for (int c = 0; c < D; ++c) {
        kc[(slot * NHKV + h) * D + c] = kseq[(j * NHKV + h) * D + c];
        vc[(slot * NHKV + h) * D + c] = vseq[(j * NHKV + h) * D + c];
      }
  }
  cudaMemcpy(b.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.kb, kb.data(), kb.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.vb, vb.data(), vb.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.kc, kc.data(), kc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.vc, vc.data(), vc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.start, &start, sizeof start, cudaMemcpyHostToDevice);
  const float scale = 1.0f / std::sqrt(float(D));
  const bool ok = sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
      b.q, b.kb, b.vb, b.kc, b.vc, b.start, NSLOTS, n, NHQ, NHKV, D, window,
      scale, b.out);
  if (!ok) return false;
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  got.resize(size_t(n) * NHQ * D);
  cudaMemcpy(got.data(), b.out, got.size() * 4, cudaMemcpyDeviceToHost);
  return true;
}

struct Case {
  uint64_t start;
  int n;
  uint64_t window;
};

// The JIT writes the generated PTX to its disk cache only after generation
// succeeds, and the runtime only launches a shape after the module loaded, so
// the file proves the codegen path ran rather than silently falling back to
// the hand-written kernel (the fallback is by design and would otherwise let
// every differential check above pass on the wrong kernel).
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

} // namespace

int main() {
  if (!have_cuda()) {
    std::cout << "test_cuda_attention_codegen SKIP (no CUDA device)\n";
    return 0;
  }
  // The runtime reads the switch on every call; codegen falls back to the
  // hand-written kernel for shapes without a generator.
  setenv("SONICBOOM_ATTN_IMPL", "codegen", 1);

  // One absolute-position stream shared by every case, so two launches that
  // cover the same position see identical Q/K/V (prefix reuse).
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> kseq(size_t(TOTAL) * NHKV * D), vseq(size_t(TOTAL) * NHKV * D),
      qabs(size_t(TOTAL) * NHQ * D);
  for (auto &x : kseq) x = nd(rng);
  for (auto &x : vseq) x = nd(rng);
  for (auto &x : qabs) x = nd(rng);

  DeviceBufs b;
  bool alloc = true;
  alloc = (cudaMalloc(&b.q, size_t(TOTAL) * NHQ * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.kb, size_t(TOTAL) * NHKV * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.vb, size_t(TOTAL) * NHKV * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.kc, size_t(NSLOTS) * NHKV * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.vc, size_t(NSLOTS) * NHKV * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.out, size_t(TOTAL) * NHQ * D * 4) == cudaSuccess) && alloc;
  alloc = (cudaMalloc(&b.start, sizeof(uint64_t)) == cudaSuccess) && alloc;
  if (!alloc) {
    std::cerr << "test_cuda_attention_codegen: cudaMalloc failed\n";
    return 1;
  }

  const Case cases[] = {{0, 1, 0},   {63, 1, 0},    {64, 1, 0},   {0, 65, 0},
                        {0, 130, 0}, {100, 130, 0}, {100, 130, 96}, {164, 1, 96},
                        {229, 1, 0}};
  std::vector<float> inv_row;  // last row of the 100+130 causal case
  const float scale = 1.0f / std::sqrt(float(D));
  for (const Case &c : cases) {
    std::vector<float> got;
    check(run_case(b, qabs, kseq, vseq, c.start, c.n, c.window, got),
          "flash_attention_chunk_dev_pos (codegen) returns true");
    if (g_failures) break;
    std::vector<double> want;
    reference(qabs, kseq, vseq, c.start, c.n, c.window, scale, want);
    double maxabs = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
      const double d = std::fabs(double(got[i]) - want[i]);
      maxabs = std::max(maxabs, d);
      maxrel = std::max(maxrel, d / std::max(1.0, std::fabs(want[i])));
    }
    if (maxrel > 1e-3)
      std::cerr << "    start=" << c.start << " n=" << c.n
                << " window=" << c.window << " maxabs=" << maxabs
                << " maxrel=" << maxrel << "\n";
    check(maxrel <= 1e-3, "generated attention matches the fp64 reference");
    if (c.start == 100 && c.window == 0)
      inv_row.assign(got.end() - size_t(NHQ) * D, got.end());
    if (c.start == 229) {
      check(!inv_row.empty(), "batch-invariance baseline captured");
      int diff = 0;
      for (size_t z = 0; z < inv_row.size(); ++z)
        if (std::bit_cast<uint32_t>(got[z]) !=
            std::bit_cast<uint32_t>(inv_row[z]))
          ++diff;
      if (diff)
        std::cerr << "    " << diff << "/" << inv_row.size()
                  << " elements differ\n";
      check(diff == 0,
            "batch invariance: row 229 alone == row 229 of the 130-row chunk "
            "(bitwise)");
    }
  }

  // Prove the generated kernels are what ran (see codegen_cache_hit): one
  // entry per mask kind the cases above exercised. A cache-disabled run
  // (SONICBOOM_GLUE_CACHE=0) writes no file, so it skips just this check.
  const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
  if (cache_env == nullptr || std::string_view(cache_env) != "0") {
    check(codegen_cache_hit("attn-d64-g7-causal-b64x64"),
          "the causal generated kernel was compiled and cached by the runtime");
    check(codegen_cache_hit("attn-d64-g7-window-b64x64"),
          "the window generated kernel was compiled and cached by the runtime");
  }

  if (g_failures == 0) {
    std::cout << "test_cuda_attention_codegen OK\n";
    return 0;
  }
  std::cerr << "test_cuda_attention_codegen FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
