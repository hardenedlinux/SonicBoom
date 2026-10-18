// Compiler-generated flash-decoding (split-KV) test (Stage 5).
//
// SONICBOOM_ATTN_DECODE=codegen routes nn::cuda::decode_attention_dev through
// the MLIR-generated pair compiler/AttentionCodegen.h emits for decode shapes:
// `fa_decode` (one warp per (query head, key chunk), online softmax in f32)
// and `fa_combine` (one block per query head, ascending chunk order). The
// generated kernels read the f32 KV cache directly (no f16 staging), so the
// fp64 reference below is the plain softmax over the same cache slots; the
// remaining difference is the f32 summation order and ex2.approx.
//
// Shapes: head_dim 256 (Gemma sliding-window and causal) and 512 (Gemma
// global causal), plus head_dim 64 with the GQA ratio of Qwen2.5 (the forced
// codegen case — `auto` keeps d64 on the hand-written kernel). Cases cover a
// short context (most chunks empty), a full window, a causal context larger
// than the slot count (slot wrap-around), and two chunk lengths: the shape
// class fixes the chunk, so the result must match the reference for both.
//
// The hand-written split-KV kernel is run on the same inputs: the generated
// path must not be worse against the fp64 reference than the kernel it
// replaces. A second codegen run must be bit-identical (determinism).
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include "../../core/src/nn/cuda_resident.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
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

struct Case {
  const char *name;
  int d;
  int nhkv;
  int gqa;
  uint64_t pos;
  uint64_t sliding;   // 0 = causal
  uint64_t sizing;    // fixed max key count the launch config is sized for
  uint64_t nslots;
  long chunk;         // 0 = runtime default
};

// The JIT writes the generated PTX to its disk cache only after generation and
// a successful module load. The file proves the codegen path ran rather than
// falling back to the hand-written kernel (the fallback is by design and would
// otherwise let a differential check pass on the wrong kernel).
bool codegen_cache_hit(const Case &c) {
  const char *home = std::getenv("HOME");
  if (!home) return false;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return false;
  const std::string mask = c.sliding != 0 ? "-window" : "-causal";
  const std::string region = "attn-dec-d" + std::to_string(c.d) + "-g" +
                             std::to_string(c.gqa) + mask;
  const std::filesystem::path p =
      std::filesystem::path(home) / ".cache" / "sonicboom" / "kernels" /
      (region + "-sm" + std::to_string(major * 10 + minor) + ".ptx");
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  return body.find(".entry fa_decode") != std::string::npos &&
         body.find(".entry fa_combine") != std::string::npos;
}

// fp64 softmax over the cache slots for one decode step. Keys are the
// absolute positions [lo..pos]; key j lives in slot j % nslots.
void reference(const std::vector<float> &q, const std::vector<float> &kc,
               const std::vector<float> &vc, const Case &c, float scale,
               std::vector<double> &out) {
  const int nhq = c.nhkv * c.gqa;
  out.assign(size_t(nhq) * c.d, 0.0);
  const long long p = (long long)c.pos;
  const long long lo =
      (c.sliding != 0 && p >= (long long)c.sliding)
          ? p - (long long)c.sliding + 1
          : 0;
  for (int h = 0; h < nhq; ++h) {
    const int hk = h / c.gqa;
    std::vector<double> s(size_t(p - lo + 1));
    double mx = -1e30;
    for (long long j = lo; j <= p; ++j) {
      const uint64_t slot = uint64_t(j) % c.nslots;
      const float *krow = kc.data() + (slot * c.nhkv + hk) * size_t(c.d);
      double acc = 0;
      for (int i = 0; i < c.d; ++i)
        acc += double(q[size_t(h) * c.d + i]) * double(krow[i]);
      s[size_t(j - lo)] = acc * scale;
      mx = std::max(mx, s[size_t(j - lo)]);
    }
    double sum = 0;
    for (auto &x : s) {
      x = std::exp(x - mx);
      sum += x;
    }
    for (int i = 0; i < c.d; ++i) {
      double acc = 0;
      for (long long j = lo; j <= p; ++j) {
        const uint64_t slot = uint64_t(j) % c.nslots;
        const float *vrow = vc.data() + (slot * c.nhkv + hk) * size_t(c.d);
        acc += s[size_t(j - lo)] * double(vrow[i]);
      }
      out[size_t(h) * c.d + i] = acc / sum;
    }
  }
}

double max_abs(const std::vector<float> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    worst = std::max(worst, std::abs(double(got[i]) - ref[i]));
  return worst;
}

double max_rel(const std::vector<float> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    // Relative for normal values, absolute below 1e-3: outputs near zero
    // (cancellation in the V-weighted sum) would otherwise turn a ~1e-8
    // absolute f32 difference into an arbitrary relative one.
    const double denom = std::abs(ref[i]) + 1e-3;
    worst = std::max(worst, std::abs(double(got[i]) - ref[i]) / denom);
  }
  return worst;
}

struct DeviceBufs {
  float *q = nullptr, *kc = nullptr, *vc = nullptr, *out = nullptr;
  uint64_t *pos = nullptr;
  ~DeviceBufs() {
    for (float *p : {q, kc, vc, out})
      if (p) cudaFree(p);
    if (pos) cudaFree(pos);
  }
};

bool run(const Case &c, const std::vector<float> &q, const std::vector<float> &kc,
         const std::vector<float> &vc, const char *impl, long chunk,
         std::vector<float> &got) {
  const int nhq = c.nhkv * c.gqa;
  DeviceBufs b;
  if (cudaMalloc(&b.q, q.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&b.kc, kc.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&b.vc, vc.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&b.out, size_t(nhq) * c.d * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&b.pos, sizeof(uint64_t)) != cudaSuccess)
    return false;
  cudaMemcpy(b.q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.kc, kc.data(), kc.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.vc, vc.data(), vc.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.pos, &c.pos, sizeof(uint64_t), cudaMemcpyHostToDevice);

  setenv("SONICBOOM_ATTN_DECODE", impl, 1);
  if (chunk > 0)
    setenv("SONICBOOM_ATTN_DECODE_CHUNK", std::to_string(chunk).c_str(), 1);
  else
    unsetenv("SONICBOOM_ATTN_DECODE_CHUNK");
  const float scale = 1.0f / std::sqrt(float(c.d));
  const bool ok = sonicboom::nn::cuda::decode_attention_dev(
      b.q, b.kc, b.vc, b.pos, c.sizing, c.nslots, nhq, c.nhkv, c.d, c.sliding,
      scale, b.out);
  if (!ok) return false;
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  got.assign(size_t(nhq) * c.d, 0.0f);
  cudaMemcpy(got.data(), b.out, got.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return true;
}

} // namespace

int main() {
  if (!have_cuda()) {
    std::cout << "SKIP: no CUDA device\n";
    return 0;
  }

  const Case cases[] = {
      // Gemma sliding-window head: short context, most chunks empty.
      {"d256-window-short", 256, 2, 4, 37, 512, 512, 2048, 0},
      // Gemma sliding-window head at a full window.
      {"d256-window-full", 256, 2, 4, 511, 512, 512, 2048, 0},
      // Gemma global head: causal, chunk count sized from n_ctx.
      {"d256-causal", 256, 2, 4, 300, 0, 1024, 2048, 0},
      // Slot wrap-around: more keys than cache slots (the cache holds the
      // most recent nslots, which is exactly what the kernels index).
      {"d256-causal-wrap", 256, 2, 4, 300, 0, 1024, 128, 0},
      // Gemma global head at head_dim 512.
      {"d512-causal", 512, 2, 4, 700, 0, 1024, 2048, 0},
      // Qwen2.5 head shape, forced through the generator.
      {"d64-causal-gqa7", 64, 2, 7, 200, 0, 512, 2048, 0},
  };

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (const Case &c : cases) {
    std::cout << c.name << ":\n";
    const int nhq = c.nhkv * c.gqa;
    std::vector<float> q(size_t(nhq) * c.d), kc(size_t(c.nslots) * c.nhkv * c.d),
        vc(size_t(c.nslots) * c.nhkv * c.d);
    for (auto &x : q) x = dist(rng);
    for (auto &x : kc) x = dist(rng);
    for (auto &x : vc) x = dist(rng);

    std::vector<double> ref;
    reference(q, kc, vc, c, 1.0f / std::sqrt(float(c.d)), ref);

    std::vector<float> own, cg, cg2;
    check(run(c, q, kc, vc, "own", 0, own), "own kernel ran");
    check(run(c, q, kc, vc, "codegen", 0, cg), "codegen kernel ran");
    check(run(c, q, kc, vc, "codegen", 0, cg2), "codegen kernel ran (repeat)");
    check(codegen_cache_hit(c), "generated PTX was written to the disk cache");

    if (own.size() == ref.size() && cg.size() == ref.size()) {
      const double own_err = max_rel(own, ref);
      const double cg_err = max_rel(cg, ref);
      std::cout << "  max rel error: own=" << own_err
                << " codegen=" << cg_err << "  (abs: own=" << max_abs(own, ref)
                << " codegen=" << max_abs(cg, ref) << ")\n";
      if (cg_err > 1e-3) {
        for (int h = 0; h < 1; ++h)
          for (int i = 0; i < 6; ++i)
            std::cerr << "    [" << h << "][" << i << "] ref=" << ref[size_t(h) * c.d + i]
                      << " own=" << own[size_t(h) * c.d + i]
                      << " cg=" << cg[size_t(h) * c.d + i] << "\n";
      }
      check(cg_err <= 1e-3, "codegen within fp64 tolerance");
      // The generated pair may reorder the f32 sums, but must not be worse
      // than the kernel it replaces (small slack for run-to-run comparison).
      check(cg_err <= std::max(2.0 * own_err, 1e-4),
            "codegen error not worse than the hand-written split-KV");
    }
    check(cg.size() == cg2.size() &&
              std::equal(cg.begin(), cg.end(), cg2.begin()),
          "codegen is deterministic (bit-identical repeat)");

    if (cg.size() == ref.size()) {
      // A different chunk length is a different reduction order for the same
      // shape class; it still has to match the reference.
      std::vector<float> alt;
      if (run(c, q, kc, vc, "codegen", 16, alt) && alt.size() == ref.size()) {
        const double alt_err = max_rel(alt, ref);
        std::cout << "  max rel error: codegen(chunk=16)=" << alt_err << "\n";
        check(alt_err <= 1e-3, "codegen at chunk=16 within tolerance");
      } else {
        check(false, "codegen at chunk=16 ran");
      }
    }
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "OK\n";
  return 0;
}
