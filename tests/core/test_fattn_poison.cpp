// Prefill attention must be a function of the K/V rows it may read. For one
// shape the call is repeated with the same valid inputs and three different
// fills of everything else:
//   - control: cache rows past the valid length zeroed, batch/Q pad zeroed;
//   - poison NaN: every cache row the call must not read (the tail past
//     start+n and the batch-owned positions [start, start+n), which have to
//     come from the batch) and the batch/Q pad filled with NaN;
//   - poison large: the same with a large finite value.
// The output is compared byte for byte. The output buffer is poisoned before
// every call, so the comparison also proves every output element is written.
//
// The d128/GQA-7 prefill drives either the row-owning FA2 family or the
// query-resident staged family; both are exercised (SONICBOOM_FATTN_FA2 1/0)
// over f16, f32 and q8 caches, with starts at chunk boundaries and off them,
// n not a multiple of the query tile, and rings larger than the span (a real
// prefill moves through a ring sized by the context, so rows past the span
// always exist).

#include <sonicboom/nn/attention.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../core/src/nn/cuda_resident.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kD = 128, kHq = 28, kHkv = 4;
constexpr float kScale = 0.08838834764831845f;
constexpr int kPad = 64;  // batch/Q pad rows past the span

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

double urand(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return double((s >> 11) & ((1ull << 53) - 1)) / double(1ull << 53);
}

enum class Fill { Random, Zero, Nan, Big };

float pick(Fill f, uint64_t& s) {
  switch (f) {
    case Fill::Random: return float((urand(s) - 0.5) * 2.0);
    case Fill::Zero: return 0.f;
    case Fill::Nan: return NAN;
    case Fill::Big: return 60000.f;
  }
  return 0.f;
}

struct Ctx {
  uint64_t start = 0;
  int n = 0;
  uint64_t n_slots = 0;
  int dtype = 0;  // 0 f32, 1 f16, 2 q8
  bool f16() const { return dtype == 1; }
  bool q8() const { return dtype == 2; }

  std::vector<float> q, kb, vb;
  std::vector<__half> kc16, vc16;
  std::vector<float> kc32, vc32;
  std::vector<int8_t> kc8, vc8;

  float *dq = nullptr, *dkb = nullptr, *dvb = nullptr, *dout = nullptr;
  void *dkc = nullptr, *dvc = nullptr;
  uint64_t* dstart = nullptr;

  void fill_cache_pos(uint64_t pos, Fill f) {
    for (int h = 0; h < kHkv; ++h) {
      // Position-deterministic: the valid rows must be identical every call.
      uint64_t s = 0x243f6a8885a308d3ull ^ (pos * 0x9e3779b97f4a7c15ull) ^
                   (uint64_t(h) << 32);
      const size_t r = size_t(pos) * kHkv + size_t(h);
      for (int pass = 0; pass < 2; ++pass) {
        if (dtype == 1) {
          __half* row = (pass == 0 ? kc16 : vc16).data() + r * kD;
          for (int d = 0; d < kD; ++d) row[d] = __float2half_rn(pick(f, s));
        } else if (dtype == 2) {
          int8_t* row = (pass == 0 ? kc8.data() : vc8.data()) + r * (kD + 4);
          for (int d = 0; d < kD; ++d) {
            const float v = pick(f, s);
            row[d] = int8_t(int(std::fmax(-127.f, std::fmin(127.f, v * 127.f))));
          }
          float sc = 0.01f;
          if (f == Fill::Nan) sc = NAN;
          else if (f == Fill::Big) sc = 60000.f;
          std::memcpy(row + kD, &sc, 4);
        } else {
          float* row = (pass == 0 ? kc32.data() : vc32.data()) + r * kD;
          for (int d = 0; d < kD; ++d) row[d] = pick(f, s);
        }
      }
    }
  }

  bool run(Fill f, std::vector<float>& out) {
    const size_t pad_v = f == Fill::Nan ? 1 : f == Fill::Big ? 2 : 0;
    std::vector<float> qq = q, kk = kb, vv = vb;
    const float pv = pad_v == 1 ? NAN : pad_v == 2 ? 60000.f : 0.f;
    for (int pad = 0; pad < kPad; ++pad) {
      const size_t r = size_t(n) + size_t(pad);
      for (int h = 0; h < kHq; ++h)
        for (int d = 0; d < kD; ++d)
          qq[(r * kHq + size_t(h)) * kD + size_t(d)] = pv;
      for (int h = 0; h < kHkv; ++h)
        for (int d = 0; d < kD; ++d) {
          kk[(r * kHkv + size_t(h)) * kD + size_t(d)] = pv;
          vv[(r * kHkv + size_t(h)) * kD + size_t(d)] = pv;
        }
    }
    cudaMemcpy(dq, qq.data(), qq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkb, kk.data(), kk.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dvb, vv.data(), vv.size() * 4, cudaMemcpyHostToDevice);
    for (uint64_t pos = 0; pos < n_slots; ++pos) {
      Fill fp;
      if (pos >= start + uint64_t(n)) fp = f;
      else if (pos >= start) fp = f == Fill::Zero ? Fill::Random : f;
      else fp = Fill::Random;
      fill_cache_pos(pos, fp);
    }
    const size_t cbytes = dtype == 1 ? kc16.size() * 2
                                     : dtype == 2 ? kc8.size()
                                                  : kc32.size() * 4;
    if (dtype == 1) {
      cudaMemcpy(dkc, kc16.data(), cbytes, cudaMemcpyHostToDevice);
      cudaMemcpy(dvc, vc16.data(), cbytes, cudaMemcpyHostToDevice);
    } else if (dtype == 2) {
      cudaMemcpy(dkc, kc8.data(), cbytes, cudaMemcpyHostToDevice);
      cudaMemcpy(dvc, vc8.data(), cbytes, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(dkc, kc32.data(), cbytes, cudaMemcpyHostToDevice);
      cudaMemcpy(dvc, vc32.data(), cbytes, cudaMemcpyHostToDevice);
    }
    out.assign(size_t(n) * kHq * kD, NAN);  // unwritten output cannot pass
    cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dstart, &start, 8, cudaMemcpyHostToDevice);
    if (!sonicboom::nn::cuda::flash_attention_chunk_dev_pos(
            dq, dkb, dvb, static_cast<const float*>(dkc),
            static_cast<const float*>(dvc), dstart, n_slots, n, kHq, kHkv, kD,
            0, kScale, dout, f16(), q8())) {
      std::fprintf(stderr, "declined\n");
      return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    return cudaMemcpy(out.data(), dout, out.size() * 4,
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  bool alloc() {
    uint64_t st = 0x9e3779b97f4a7c15ull ^ (start * 1315423911ull) ^ uint64_t(n);
    q.assign(size_t(n + kPad) * kHq * kD, 0.f);
    kb.assign(size_t(n + kPad) * kHkv * kD, 0.f);
    vb.assign(size_t(n + kPad) * kHkv * kD, 0.f);
    for (auto& x : q) x = float((urand(st) - 0.5) * 2.0);
    for (auto& x : kb) x = float((urand(st) - 0.5) * 2.0);
    for (auto& x : vb) x = float((urand(st) - 0.5) * 2.0);
    if (dtype == 1) {
      kc16.resize(size_t(n_slots) * kHkv * kD);
      vc16.resize(size_t(n_slots) * kHkv * kD);
    } else if (dtype == 2) {
      kc8.resize(size_t(n_slots) * kHkv * (kD + 4));
      vc8.resize(size_t(n_slots) * kHkv * (kD + 4));
    } else {
      kc32.resize(size_t(n_slots) * kHkv * kD);
      vc32.resize(size_t(n_slots) * kHkv * kD);
    }
    cudaMalloc(&dq, q.size() * 4);
    cudaMalloc(&dkb, kb.size() * 4);
    cudaMalloc(&dvb, vb.size() * 4);
    cudaMalloc(&dout, size_t(n) * kHq * kD * 4);
    cudaMalloc(&dstart, 8);
    const size_t cbytes = dtype == 1 ? kc16.size() * 2
                                     : dtype == 2 ? kc8.size()
                                                  : kc32.size() * 4;
    cudaMalloc(&dkc, cbytes);
    cudaMalloc(&dvc, cbytes);
    return dq && dkb && dvb && dout && dstart && dkc && dvc;
  }
};

bool run_shape(uint64_t start, int n, uint64_t n_slots, int dtype,
               const char* fa2) {
  setenv("SONICBOOM_FATTN_FA2", fa2, 1);
  setenv("SONICBOOM_FATTN_REPORT", "0", 1);
  Ctx c;
  c.start = start;
  c.n = n;
  c.n_slots = n_slots;
  c.dtype = dtype;
  if (!c.alloc()) return false;
  std::vector<float> ref, out;
  bool ok = c.run(Fill::Zero, ref);
  if (ok) {
    static const Fill kPoisons[] = {Fill::Nan, Fill::Big};
    for (Fill f : kPoisons) {
      ok = c.run(f, out) &&
           std::memcmp(ref.data(), out.data(), ref.size() * 4) == 0;
      if (!ok) break;
    }
  }
  // A second control proves the call is a function of its inputs at all.
  if (ok) ok = c.run(Fill::Zero, out) &&
               std::memcmp(ref.data(), out.data(), ref.size() * 4) == 0;
  cudaFree(c.dq); cudaFree(c.dkb); cudaFree(c.dvb); cudaFree(c.dout);
  cudaFree(c.dkc); cudaFree(c.dvc); cudaFree(c.dstart);
  return ok;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("fattn poison: no CUDA device, skipped\n");
    return 0;
  }
  struct Shape { uint64_t start; int n; uint64_t n_slots; };
  static const Shape kShapes[] = {
      {0, 512, 512},      // no cache reads
      {3584, 512, 4096},  // production last segment, ring ends at the span
      {3584, 512, 3584},  // tight ring: the cache ends at the first query
      {3584, 512, 4160},  // ring larger than the span: tail rows exist
      {100, 37, 300},     // non-tile N, unaligned start, tail
      {0, 513, 577},      // non-tile N from position 0
      {300, 7, 400},      // tiny n off a chunk boundary
  };
  static const char* kDtypes[] = {"f32", "f16", "q8"};
  for (const char* fa2 : {"1", "0"}) {
    for (int d = 0; d < 3; ++d) {
      for (const Shape& s : kShapes) {
        char label[160];
        std::snprintf(label, sizeof(label),
                      "attn start=%llu n=%d n_slots=%llu kv=%s fa2=%s "
                      "unused rows change the output",
                      (unsigned long long)s.start, s.n,
                      (unsigned long long)s.n_slots, kDtypes[d], fa2);
        check(run_shape(s.start, s.n, s.n_slots, d, fa2), label);
      }
    }
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "fattn poison: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("fattn poison: all shapes byte-identical to the control\n");
  return 0;
}
