// Compiler-generated q8_1 x q8_1 int8 tensor-core matmul (MMQ1).
//
// The generated kernel computes Y[N][M] = X[N][K] * W[M][K]^T from the q8_1
// operands the prefill path already produces: the activation in the
// block_q8_1_mmq layout ([K/128][N] 144-byte blocks, ds=0 float scales or ds=1
// half2 scales), the weights as int8 rows with per-32 f32 scales. It is checked
// here against an fp64 reference that evaluates the kernel's arithmetic — an
// exact integer dot over each 32-value group, scaled by the group's activation
// and weight scales — for full tiles, for a tail shape where M and N miss the
// tile multiples, and for the bitwise batch-invariance contract.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include "../../core/src/planner/glue_jit.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kBlockBytes = 144;  // block_q8_1_mmq: 16 scale bytes + 128 int8
constexpr int kQsOffset = 16;

struct Q8Weight {
  std::vector<int8_t> q;   // [m][k]
  std::vector<float> s;    // [m][k/32]
};

// Random weights in the converter's layout: each 32-value group quantized with
// its own scale from a random f32 row (the same arithmetic the host-side
// converter performs, without the dequantize-from-K-quant step).
Q8Weight make_weight(int m, int k, std::mt19937 &rng) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Q8Weight w;
  w.q.resize(size_t(m) * k);
  w.s.resize(size_t(m) * (k / 32));
  std::vector<float> row(k);
  for (int r = 0; r < m; ++r) {
    for (float &v : row) v = nd(rng) / std::sqrt(float(k));
    for (int g = 0; g < k / 32; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(row[g * 32 + j]));
      const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
      w.s[size_t(r) * (k / 32) + g] = d;
      for (int j = 0; j < 32; ++j) {
        float v = d > 0.0f ? std::round(row[g * 32 + j] / d) : 0.0f;
        v = std::max(-127.0f, std::min(127.0f, v));
        w.q[size_t(r) * k + g * 32 + j] = int8_t(v);
      }
    }
  }
  return w;
}

// The activation in block_q8_1_mmq layout, [k/128][n] blocks. `ds` selects
// float (0) or half2 (1) scales; values are random int8 with random positive
// scales. Blocks are generated for a fixed token count (`nmax`) so a smaller
// launch's packed buffer can be derived from the same per-token content.
struct Q8Act {
  std::vector<uint8_t> bytes;
  int nmax = 0;
};

// Repack the first `n` tokens' blocks into the [k/128][n] layout a launch of n
// tokens indexes.
Q8Act pack_tokens(const Q8Act &full, int k, int n) {
  const int kb = (k + 127) / 128;
  Q8Act a;
  a.nmax = n;
  a.bytes.assign(size_t(kb) * n * kBlockBytes, 0);
  for (int b = 0; b < kb; ++b)
    for (int t = 0; t < n; ++t)
      std::memcpy(a.bytes.data() + (size_t(b) * n + t) * kBlockBytes,
                  full.bytes.data() + (size_t(b) * full.nmax + t) * kBlockBytes,
                  kBlockBytes);
  return a;
}

Q8Act make_activation(int k, int n, int ds, std::mt19937 &rng) {
  std::uniform_int_distribution<int> qi(-127, 127);
  std::uniform_real_distribution<float> sd(0.001f, 0.05f);
  const int kb = (k + 127) / 128;
  Q8Act a;
  a.nmax = n;
  a.bytes.assign(size_t(kb) * n * kBlockBytes, 0);
  for (int b = 0; b < kb; ++b)
    for (int t = 0; t < n; ++t) {
      uint8_t *p = a.bytes.data() + (size_t(b) * n + t) * kBlockBytes;
      for (int s = 0; s < 4; ++s) {
        const float d = sd(rng);
        if (ds == 1) {
          const uint16_t h = [] (float f) {
            const uint32_t u = std::bit_cast<uint32_t>(f);
            const uint32_t sign = (u >> 16) & 0x8000u;
            const uint32_t abs = u & 0x7fffffffu;
            if (abs >= 0x47800000u) return uint16_t(sign | 0x7c00u);
            if (abs < 0x38800000u) return uint16_t(sign);  // below f16 normal
            const uint32_t e = (abs >> 23) - 112;
            const uint32_t m = abs & 0x7fffffu;
            return uint16_t(sign | ((e << 10) +
                                    ((m + 0x1000u + ((m >> 13) & 1u)) >> 13)));
          }(d);
          uint16_t *hp = reinterpret_cast<uint16_t *>(p + s * 4);
          hp[0] = h;
          hp[1] = h;
        } else {
          std::memcpy(p + s * 4, &d, 4);
        }
      }
      for (int j = 0; j < 128; ++j) p[kQsOffset + j] = uint8_t(int8_t(qi(rng)));
    }
  return a;
}

// fp64 reference: sum_g (sa[g] * sw[r][g]) * (integer dot over group g), with
// out-of-range K values (K not a multiple of 128) zeroed from the activation
// buffer, exactly as the kernel masks them.
void reference(const Q8Act &a, const Q8Weight &w, int m, int k, int n, int ds,
               std::vector<double> &out) {
  out.assign(size_t(n) * m, 0.0);
  const int kb = (k + 127) / 128;
  std::vector<int8_t> act(size_t(n) * kb * 128, 0);
  std::vector<float> ascl(size_t(n) * kb * 4, 0.0f);
  for (int b = 0; b < kb; ++b)
    for (int t = 0; t < n; ++t) {
      const uint8_t *p = a.bytes.data() + (size_t(b) * n + t) * kBlockBytes;
      for (int s = 0; s < 4; ++s) {
        float d;
        if (ds == 1) {
          const uint16_t *hp = reinterpret_cast<const uint16_t *>(p + s * 4);
          d = [] (uint16_t h) {
            const uint32_t sign = uint32_t(h & 0x8000u) << 16;
            const uint32_t e = (h >> 10) & 0x1fu;
            uint32_t mant = h & 0x3ffu;
            uint32_t u;
            if (e == 0) {
              u = sign;
            } else if (e == 31) {
              u = sign | 0x7f800000u | (mant << 13);
            } else {
              u = sign | ((e - 15 + 127) << 23) | (mant << 13);
            }
            return std::bit_cast<float>(u);
          }(hp[0]);
        } else {
          std::memcpy(&d, p + s * 4, 4);
        }
        ascl[size_t(t) * kb * 4 + b * 4 + s] = d;
      }
      for (int j = 0; j < 128; ++j)
        act[size_t(t) * kb * 128 + b * 128 + j] = int8_t(p[kQsOffset + j]);
    }
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < m; ++r) {
      double acc = 0.0;
      for (int g = 0; g < k / 32; ++g) {
        int32_t isum = 0;
        for (int j = 0; j < 32; ++j) {
          const int kk = g * 32 + j;
          const size_t ai = size_t(t) * kb * 128 + (kk / 128) * 128 + (kk % 128);
          isum += int32_t(act[ai]) * int32_t(w.q[size_t(r) * k + kk]);
        }
        const int b = g / 4, s = g % 4;
        const double sa = ascl[size_t(t) * kb * 4 + b * 4 + s];
        const double sw = w.s[size_t(r) * (k / 32) + g];
        acc += (sa * sw) * double(isum);
      }
      out[size_t(t) * m + r] = acc;
    }
}

struct DeviceBufs {
  uint8_t *x = nullptr;
  int8_t *wq = nullptr;
  float *ws = nullptr;
  float *y = nullptr;
  ~DeviceBufs() {
    if (x) cudaFree(x);
    if (wq) cudaFree(wq);
    if (ws) cudaFree(ws);
    if (y) cudaFree(y);
  }
};

bool alloc_bufs(int m, int k, int nmax, int ds, DeviceBufs &b) {
  const int kb = (k + 127) / 128;
  return cudaMalloc(&b.x, size_t(kb) * nmax * kBlockBytes) == cudaSuccess &&
         cudaMalloc(&b.wq, size_t(m) * k) == cudaSuccess &&
         cudaMalloc(&b.ws, size_t(m) * (k / 32) * 4) == cudaSuccess &&
         cudaMalloc(&b.y, size_t(nmax) * m * 4) == cudaSuccess;
}

bool run(const sonicboom::compiler::MmqTcShape &shape, DeviceBufs &b, int m,
         int k, int n) {
  return sonicboom::planner::glue::launch_mmq_tc(shape, b.x, b.wq, b.ws, b.y, m,
                                                 k, n);
}

bool cache_hit(const char *region) {
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

}  // namespace

int main() {
  using sonicboom::compiler::MmqTcShape;
  if (!have_cuda()) {
    std::cout << "test_cuda_mmq_tc_codegen SKIP (no CUDA device)\n";
    return 0;
  }
  std::mt19937 rng(4141);
  const int M = 256, K = 512, NMAX = 512;
  const MmqTcShape shape{64, 64, 0};
  DeviceBufs b;
  if (!alloc_bufs(M, K, NMAX, 0, b)) {
    std::cerr << "test_cuda_mmq_tc_codegen: cudaMalloc failed\n";
    return 1;
  }
  const Q8Weight w = make_weight(M, K, rng);
  const Q8Act act_full = make_activation(K, NMAX, 0, rng);
  const auto upload = [&](const Q8Act &a) {
    return cudaMemcpy(b.x, a.bytes.data(), a.bytes.size(),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  };
  check(upload(pack_tokens(act_full, K, NMAX)) &&
            cudaMemcpy(b.wq, w.q.data(), w.q.size(), cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(b.ws, w.s.data(), w.s.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess,
        "operand upload");

  std::vector<float> got(size_t(NMAX) * M);
  const auto compare = [&](int n, int ds, const Q8Act &a, const Q8Weight &ww,
                           const char *what) {
    std::vector<double> want;
    reference(a, ww, M, K, n, ds, want);
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
    if (maxrel > 1e-6)
      std::cerr << "    n=" << n << " maxabs=" << maxabs
                << " maxrel=" << maxrel << "\n";
    check(maxrel <= 1e-6, what);
    std::printf("  n=%d maxabs=%.3e maxrel=%.3e\n", n, maxabs, maxrel);
  };

  const Q8Act act64 = pack_tokens(act_full, K, 64);
  check(upload(act64), "activation upload");
  check(run(shape, b, M, K, 64), "generated mmq-tc launches");
  check(cudaDeviceSynchronize() == cudaSuccess, "kernel completes");
  compare(64, 0, act64, w, "generated mmq-tc matches the fp64 reference");

  // ---- ds=1 (half2 activation scales) ----
  {
    const MmqTcShape hshape{64, 64, 1};
    const Q8Act hact_full = make_activation(K, NMAX, 1, rng);
    const Q8Act hact = pack_tokens(hact_full, K, 64);
    check(upload(hact), "half-scale activation upload");
    check(run(hshape, b, M, K, 64), "ds=1 mmq-tc launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "ds=1 kernel completes");
    compare(64, 1, hact, w, "ds=1 mmq-tc matches the fp64 reference");
  }

  // ---- tail shape: M and N miss the tile multiples, K misses 128 ----
  {
    const int MT = 100, KT = 96, NT = 7;
    DeviceBufs bt;
    if (!alloc_bufs(MT, KT, NT, 0, bt)) {
      std::cerr << "test_cuda_mmq_tc_codegen: tail cudaMalloc failed\n";
      return 1;
    }
    const Q8Weight wt = make_weight(MT, KT, rng);
    const Q8Act at = make_activation(KT, NT, 0, rng);
    check(cudaMemcpy(bt.x, at.bytes.data(), at.bytes.size(),
                     cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(bt.wq, wt.q.data(), wt.q.size(),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(bt.ws, wt.s.data(), wt.s.size() * 4,
                         cudaMemcpyHostToDevice) == cudaSuccess,
          "tail operand upload");
    const MmqTcShape tail{32, 64, 0};
    check(run(tail, bt, MT, KT, NT), "tail-shape mmq-tc launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "tail kernel completes");
    std::vector<double> want;
    reference(at, wt, MT, KT, NT, 0, want);
    std::vector<float> g(size_t(NT) * MT);
    check(cudaMemcpy(g.data(), bt.y, g.size() * 4, cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "tail result download");
    double maxrel = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
      maxrel = std::max(maxrel, std::fabs(double(g[i]) - want[i]) /
                                    std::max(1.0, std::fabs(want[i])));
    if (maxrel > 1e-6) std::cerr << "    tail maxrel=" << maxrel << "\n";
    check(maxrel <= 1e-6, "mmq-tc matches fp64 on a tail shape");
    std::printf("  tail m=%d k=%d n=%d maxrel=%.3e\n", MT, KT, NT, maxrel);
  }

  // ---- batch invariance (bitwise) ----
  std::vector<uint32_t> row0;
  for (int n : {1, 7, 64, NMAX}) {
    check(upload(pack_tokens(act_full, K, n)), "invariance activation upload");
    check(run(shape, b, M, K, n), "invariance launch");
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

  // ---- tile invariance (bitwise): every MMQ2 candidate must equal the
  // smallest config. The autotuner accepts a tuned tile only on this contract,
  // so the choice never changes numerics. ----
  {
    check(upload(act64), "tile-invariance activation upload");
    std::vector<float> y(size_t(64) * M);
    std::vector<uint32_t> want(y.size());
    const MmqTcShape smallest{16, 16, 0};
    check(run(smallest, b, M, K, 64), "smallest tile launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "smallest tile completes");
    check(cudaMemcpy(y.data(), b.y, y.size() * 4, cudaMemcpyDeviceToHost) ==
              cudaSuccess,
          "smallest tile download");
    for (size_t i = 0; i < y.size(); ++i) want[i] = std::bit_cast<uint32_t>(y[i]);
    for (int bm : {16, 32, 64, 128})
      for (int bn : {16, 32, 64}) {
        if (bm == 16 && bn == 16) continue;
        const MmqTcShape s{bm, bn, 0};
        check(run(s, b, M, K, 64), "tile-invariance launch");
        check(cudaDeviceSynchronize() == cudaSuccess,
              "tile-invariance kernel completes");
        check(cudaMemcpy(y.data(), b.y, y.size() * 4,
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "tile-invariance download");
        size_t diff = 0;
        for (size_t i = 0; i < y.size(); ++i)
          if (std::bit_cast<uint32_t>(y[i]) != want[i]) ++diff;
        if (diff)
          std::cerr << "    tile " << bm << "x" << bn << ": " << diff
                    << " elements differ from 16x16\n";
        check(diff == 0, "tile output is bitwise equal to the 16x16 config");
      }
    std::printf("  tiles 16x16..128x64: bitwise identical\n");
  }

  // ---- the JIT kept the generated kernel on disk ----
  const char *cache_env = std::getenv("SONICBOOM_GLUE_CACHE");
  if (cache_env == nullptr || std::string_view(cache_env) != "0") {
    check(cache_hit("mmqtc-m64n64d0"),
          "the generated mmq-tc kernel was compiled and cached");
    check(cache_hit("mmqtc-m64n64d1"),
          "the ds=1 mmq-tc kernel was compiled and cached");
  }

  if (g_failures == 0) {
    std::cout << "test_cuda_mmq_tc_codegen OK\n";
    return 0;
  }
  std::cerr << "test_cuda_mmq_tc_codegen FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
