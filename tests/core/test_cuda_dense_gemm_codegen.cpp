// Compiler-generated dense bf16 GEMM test (v1 Stage 6, step 10a).
//
// The generated kernel computes Y[N][M] = X[N][K] * W[M][K]^T with the strided
// dense projection's layouts: X token-major bf16, W row-major bf16, Y
// token-major f32 (y[token * M + row]). It is checked here against an fp64
// reference that consumes the same bf16-rounded operands, for a full
// (64, 64, 64) tile and for a tail shape where every dimension misses a tile
// multiple (M = 1000, K = 1000, N = 7).
//
// The batch-invariance contract is the same as the attention kernels': a
// token's output row must be bit-identical whether it is computed alone
// (N = 1), in a 7-token segment, in a 64-token segment, or inside a full
// 512-token prefill — and also when the same token sits at a different
// (block, position-in-block) coordinate, which is checked by re-running the
// tail of the stream through the kernel with the base pointer shifted by 64
// tokens. That is what lets the planner drop the fixed 128-column cuBLASLt
// blocking.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include "../../core/src/planner/glue_jit.h"

#include <cuda_runtime.h>

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

// Round to bfloat16 (round-to-nearest-even), as the kernel's operands are.
float bf16r(float x) {
  uint32_t u = std::bit_cast<uint32_t>(x);
  const uint32_t sign = u & 0x80000000u;
  u &= 0x7fffffffu;
  if (u >= 0x7f800000u) return x;  // inf / nan pass through
  u += 0x7fff + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  return std::bit_cast<float>(sign | u);
}

uint16_t bf16_bits(float x) {
  return uint16_t(std::bit_cast<uint32_t>(bf16r(x)) >> 16);
}

// fp64 reference over the bf16-rounded operands.
void reference(const std::vector<uint16_t> &x, const std::vector<uint16_t> &w,
               int m, int k, int n, std::vector<double> &out) {
  out.assign(size_t(n) * m, 0.0);
  std::vector<double> xf(size_t(n) * k), wf(size_t(m) * k);
  for (size_t i = 0; i < xf.size(); ++i)
    xf[i] = double(std::bit_cast<float>(uint32_t(x[i]) << 16));
  for (size_t i = 0; i < wf.size(); ++i)
    wf[i] = double(std::bit_cast<float>(uint32_t(w[i]) << 16));
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < m; ++r) {
      double acc = 0.0;
      for (int i = 0; i < k; ++i) acc += xf[size_t(t) * k + i] * wf[size_t(r) * k + i];
      out[size_t(t) * m + r] = acc;
    }
}

struct DeviceBufs {
  uint16_t *x = nullptr;
  uint16_t *w = nullptr;
  float *y = nullptr;
  ~DeviceBufs() {
    if (x) cudaFree(x);
    if (w) cudaFree(w);
    if (y) cudaFree(y);
  }
};

bool alloc_bufs(int m, int k, int nmax, DeviceBufs &b) {
  return cudaMalloc(&b.x, size_t(nmax) * k * 2) == cudaSuccess &&
         cudaMalloc(&b.w, size_t(m) * k * 2) == cudaSuccess &&
         cudaMalloc(&b.y, size_t(nmax) * m * 4) == cudaSuccess;
}

bool run(const sonicboom::compiler::DenseGemmShape &shape, DeviceBufs &b,
         int m, int k, int n, size_t x_off, size_t y_off) {
  return sonicboom::planner::glue::launch_dense_gemm(
      shape, b.x + x_off, b.w, b.y + y_off, m, k, n);
}

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
  using sonicboom::compiler::DenseGemmShape;
  if (!have_cuda()) {
    std::cout << "test_cuda_dense_gemm_codegen SKIP (no CUDA device)\n";
    return 0;
  }
  const DenseGemmShape shape{64, 64, 64};

  // ---- 1. full tiles: Y = X W^T against fp64 ----
  const int M = 1024, K = 1024, NMAX = 512;
  DeviceBufs b;
  if (!alloc_bufs(M, K, NMAX, b)) {
    std::cerr << "test_cuda_dense_gemm_codegen: cudaMalloc failed\n";
    return 1;
  }
  std::mt19937 rng(6006);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> xh(size_t(NMAX) * K), wh(size_t(M) * K);
  for (auto &v : xh) v = bf16_bits(nd(rng));
  for (auto &v : wh) v = bf16_bits(nd(rng) / std::sqrt(float(K)));
  check(cudaMemcpy(b.x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess &&
        cudaMemcpy(b.w, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess,
        "operand upload");

  std::vector<float> got(size_t(NMAX) * M);
  const auto compare = [&](int n, const char *what) {
    std::vector<double> want;
    reference(xh, wh, M, K, n, want);
    check(cudaMemcpy(got.data(), b.y, size_t(n) * M * 4,
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "result download");
    double maxabs = 0.0, maxrel = 0.0;
    for (int t = 0; t < n; ++t)
      for (int r = 0; r < M; ++r) {
        const double d = std::fabs(double(got[size_t(t) * M + r]) -
                                   want[size_t(t) * M + r]);
        maxabs = std::max(maxabs, d);
        maxrel = std::max(maxrel,
                          d / std::max(1.0, std::fabs(want[size_t(t) * M + r])));
      }
    if (maxrel > 1e-3)
      std::cerr << "    n=" << n << " maxabs=" << maxabs
                << " maxrel=" << maxrel << "\n";
    check(maxrel <= 1e-3, what);
    std::printf("  n=%d maxabs=%.3e maxrel=%.3e\n", n, maxabs, maxrel);
  };

  check(run(shape, b, M, K, 64, 0, 0), "generated dense GEMM launches");
  check(cudaDeviceSynchronize() == cudaSuccess, "kernel completes");
  compare(64, "generated GEMM matches fp64 on full tiles");

  // ---- 2. tail shape: M, K and N all miss the tile multiples ----
  const int MT = 1000, KT = 1000, NT = 7;
  DeviceBufs bt;
  if (!alloc_bufs(MT, KT, NT, bt)) {
    std::cerr << "test_cuda_dense_gemm_codegen: cudaMalloc failed\n";
    return 1;
  }
  std::vector<uint16_t> xt(size_t(NT) * KT), wt(size_t(MT) * KT);
  for (auto &v : xt) v = bf16_bits(nd(rng));
  for (auto &v : wt) v = bf16_bits(nd(rng) / std::sqrt(float(KT)));
  check(cudaMemcpy(bt.x, xt.data(), xt.size() * 2, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(bt.w, wt.data(), wt.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess,
        "tail operand upload");
  {
    const DenseGemmShape tail{32, 64, 32};
    check(run(tail, bt, MT, KT, NT, 0, 0), "tail-shape GEMM launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "tail kernel completes");
    std::vector<double> want;
    reference(xt, wt, MT, KT, NT, want);
    std::vector<float> g(size_t(NT) * MT);
    check(cudaMemcpy(g.data(), bt.y, g.size() * 4, cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "tail result download");
    double maxrel = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
      maxrel = std::max(maxrel, std::fabs(double(g[i]) - want[i]) /
                                    std::max(1.0, std::fabs(want[i])));
    if (maxrel > 1e-3) std::cerr << "    tail maxrel=" << maxrel << "\n";
    check(maxrel <= 1e-3, "generated GEMM matches fp64 on a tail shape");
    std::printf("  tail m=%d k=%d n=%d maxrel=%.3e\n", MT, KT, NT, maxrel);
  }

  // ---- 3. batch invariance (bitwise) ----
  // Token 0 through the same kernel at N = 1, 7, 64 and 512.
  std::vector<uint32_t> row0;
  for (int n : {1, 7, 64, NMAX}) {
    check(run(shape, b, M, K, n, 0, 0), "invariance launch");
    check(cudaDeviceSynchronize() == cudaSuccess, "invariance kernel completes");
    std::vector<float> y(size_t(M), 0.0f);
    check(cudaMemcpy(y.data(), b.y, y.size() * 4,
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "invariance download");
    if (n == 1) {
      row0.resize(M);
      for (int r = 0; r < M; ++r) row0[r] = std::bit_cast<uint32_t>(y[r]);
      continue;
    }
    size_t diff = 0;
    for (int r = 0; r < M; ++r)
      if (std::bit_cast<uint32_t>(y[r]) != row0[r]) ++diff;
    if (diff) std::cerr << "    n=" << n << ": " << diff << "/" << M
                        << " row-0 elements differ from n=1\n";
    check(diff == 0, "token 0 row is bitwise independent of N");
  }
  // Tokens 0..6 between the 7-token and the 512-token call.
  check(run(shape, b, M, K, 7, 0, 0), "7-token launch");
  check(cudaDeviceSynchronize() == cudaSuccess, "7-token kernel completes");
  std::vector<float> y7(size_t(7) * M);
  check(cudaMemcpy(y7.data(), b.y, y7.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess,
        "7-token download");
  check(run(shape, b, M, K, NMAX, 0, 0), "full-stream launch");
  check(cudaDeviceSynchronize() == cudaSuccess, "full-stream kernel completes");
  std::vector<float> yfull(size_t(NMAX) * M);
  check(cudaMemcpy(yfull.data(), b.y, yfull.size() * 4,
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "full-stream download");
  {
    size_t diff = 0;
    for (size_t i = 0; i < y7.size(); ++i)
      if (std::bit_cast<uint32_t>(y7[i]) != std::bit_cast<uint32_t>(yfull[i]))
        ++diff;
    check(diff == 0, "7-token rows == first 7 rows of the 512-token call");
  }
  // Tokens 64..511 through the kernel with the base pointer shifted by 64
  // tokens: same values, different (block, position-in-block) coordinates.
  {
    check(run(shape, b, M, K, NMAX - 64, size_t(64) * K, size_t(64) * M),
          "shifted-base launch");
    check(cudaDeviceSynchronize() == cudaSuccess, "shifted kernel completes");
    std::vector<float> yshift(size_t(NMAX - 64) * M);
    check(cudaMemcpy(yshift.data(), b.y + size_t(64) * M, yshift.size() * 4,
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "shifted download");
    size_t diff = 0;
    for (size_t i = 0; i < yshift.size(); ++i)
      if (std::bit_cast<uint32_t>(yshift[i]) !=
          std::bit_cast<uint32_t>(yfull[size_t(64) * M + i]))
        ++diff;
    if (diff) std::cerr << "    shifted: " << diff << "/" << yshift.size()
                        << " elements differ\n";
    check(diff == 0, "token row does not depend on its block coordinate");
  }

  // ---- 4. the JIT kept the generated kernel on disk ----
  const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
  if (cache_env == nullptr || std::string_view(cache_env) != "0") {
    check(codegen_cache_hit("gemm-m64n64k64"),
          "the generated dense GEMM was compiled and cached");
    check(codegen_cache_hit("gemm-m32n64k32"),
          "the tail-shape dense GEMM was compiled and cached");
  }

  if (g_failures == 0) {
    std::cout << "test_cuda_dense_gemm_codegen OK\n";
    return 0;
  }
  std::cerr << "test_cuda_dense_gemm_codegen FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
