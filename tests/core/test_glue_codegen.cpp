// Compiler-generated glue kernel test (Stage 5).
//
// The MLIR -> NVVM -> PTX pipeline (compiler::generate_glue_ptx) and the driver
// JIT (planner::glue::launch) must reproduce the hand-written kernel the region
// replaces, byte for byte.
//
// Decode (5a): RMS-norm + q8_1 quantize. For random x, w the generated kernel's
// (d, qs[32]) blocks must memcmp-equal quant::cuda::rms_norm_quantize_q8_1_dev's
// output — d included, so the reduction order and the quantize sequence both
// have to match, not just the int8 quants.
//
// Prefill (5b): residual add + RMS-norm + mmq q8_1 quantize. The generated
// kernel's residual sum and 144-byte mmq blocks (d, optional ds half2, qs) must
// memcmp-equal nn::cuda::add_dev followed by
// quant::cuda::rms_norm_quantize_mmq_q8_1_dev over the same rows.
//
// Prefill (5c): RMS-norm + residual add. The generated kernel's row must
// memcmp-equal rms_norm_heads_dev followed by add_dev over the same rows: the
// inverse RMS is the norm kernel's reduction and the apply is `residual +
// x * inv * w` in that order.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include <sonicboom/compiler/GlueCodegen.h>

#include "../../core/src/nn/cuda_resident.h"
#include "../../core/src/planner/glue_jit.h"
#include "../../core/src/quant/cuda_resident.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool have_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int device_arch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return 0;
  return major * 10 + minor;
}

// A norm-weight buffer that stays alive for the whole run. The reference
// kernels upload their weight through upload_bytes, which caches by (host
// pointer, byte size); a local std::vector whose address gets recycled by the
// allocator would make a later call silently reuse an earlier test's weights.
std::vector<float>& persistent_weight(size_t n) {
  static std::vector<std::unique_ptr<std::vector<float>>> keep;
  keep.push_back(std::make_unique<std::vector<float>>(n));
  return *keep.back();
}

// One region launch, compared byte for byte with the hand-written kernel.
void test_rms_norm_quantize_q8_1(std::mt19937& rng, int n) {
  const std::string tag = "rms_norm_quantize_q8_1 n=" + std::to_string(n);

  std::vector<float> x(n);
  std::vector<float>& w = persistent_weight(n);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  const float eps = 1e-6f;

  const int nblocks = n / 32;
  const size_t yq_bytes =
      size_t(nblocks) * sizeof(sonicboom::quant::cuda::BlockQ8_1);

  float *xd = nullptr, *wd = nullptr;
  void *yref = nullptr, *ygen = nullptr;
  check(cudaMalloc(&xd, n * sizeof(float)) == cudaSuccess, "cudaMalloc x");
  check(cudaMalloc(&wd, n * sizeof(float)) == cudaSuccess, "cudaMalloc w");
  check(cudaMalloc(&yref, yq_bytes) == cudaSuccess, "cudaMalloc yref");
  check(cudaMalloc(&ygen, yq_bytes) == cudaSuccess, "cudaMalloc ygen");
  if (g_failures) return;

  cudaMemcpy(xd, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(yref, 0, yq_bytes);
  cudaMemset(ygen, 0, yq_bytes);

  // The reference: the kernel the region replaces.
  check(sonicboom::quant::cuda::rms_norm_quantize_q8_1_dev(
            std::span<const float>(w), xd, eps,
            reinterpret_cast<sonicboom::quant::cuda::BlockQ8_1 *>(yref), n),
        (tag + ": reference kernel").c_str());

  // The generated kernel, loaded and launched through the JIT.
  void *xarg = xd;
  void *warg = wd;
  void *yarg = ygen;
  int narg = n;
  float earg = eps;
  void *args[] = {&xarg, &warg, &yarg, &narg, &earg};
  check(sonicboom::planner::glue::launch(
            "rms_norm_quantize_q8_1", args, 1,
            sonicboom::compiler::kGlueBlockThreads,
            sonicboom::compiler::kGlueDynamicSharedBytes),
        (tag + ": generated kernel launch").c_str());

  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<uint8_t> got(yq_bytes), ref(yq_bytes);
  cudaMemcpy(got.data(), ygen, yq_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), yref, yq_bytes, cudaMemcpyDeviceToHost);

  const bool equal = std::memcmp(got.data(), ref.data(), yq_bytes) == 0;
  if (!equal) {
    // Report the first differing block and whether it is the scale or a quant,
    // so a reduction-order and a quantization mismatch are distinguishable.
    for (int b = 0; b < nblocks; ++b) {
      const size_t off = size_t(b) * sizeof(sonicboom::quant::cuda::BlockQ8_1);
      if (std::memcmp(got.data() + off, ref.data() + off,
                      sizeof(sonicboom::quant::cuda::BlockQ8_1)) == 0)
        continue;
      float dg = 0, dr = 0;
      std::memcpy(&dg, got.data() + off, sizeof(float));
      std::memcpy(&dr, ref.data() + off, sizeof(float));
      std::cerr << "    first differing block " << b << ": d got=" << dg
                << " ref=" << dr << "\n";
      break;
    }
  }
  check(equal, (tag + ": byte-identical output").c_str());

  cudaFree(xd);
  cudaFree(wd);
  cudaFree(yref);
  cudaFree(ygen);
}

// One prefill add+norm+quantize region launch, compared byte for byte with the
// reference pair. `ds` is the mmq scale layout (0 = D4, 1 = DS4 half2).
void test_add_rms_norm_quantize_mmq_q8_1(std::mt19937& rng, int tokens, int dim,
                                         int ds) {
  const std::string tag = "add_rms_norm_quantize_mmq_q8_1 tokens=" +
                          std::to_string(tokens) + " dim=" +
                          std::to_string(dim) + " ds=" + std::to_string(ds);

  const size_t elems = size_t(tokens) * size_t(dim);
  std::vector<float> a(elems), b(elems);
  std::vector<float>& w = persistent_weight(dim);
  for (auto& v : a) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : b) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  const float eps = 1e-6f;

  const int nblocks = dim / 128;
  const size_t yq_bytes =
      size_t(tokens) * size_t(nblocks) *
      sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq);
  const size_t xy_bytes = elems * sizeof(float);

  float *ad = nullptr, *bd = nullptr, *wd = nullptr;
  float *resref = nullptr, *resgen = nullptr;
  void *yref = nullptr, *ygen = nullptr;
  check(cudaMalloc(&ad, xy_bytes) == cudaSuccess, "cudaMalloc a");
  check(cudaMalloc(&bd, xy_bytes) == cudaSuccess, "cudaMalloc b");
  check(cudaMalloc(&wd, dim * sizeof(float)) == cudaSuccess, "cudaMalloc w");
  check(cudaMalloc(&resref, xy_bytes) == cudaSuccess, "cudaMalloc resref");
  check(cudaMalloc(&resgen, xy_bytes) == cudaSuccess, "cudaMalloc resgen");
  check(cudaMalloc(&yref, yq_bytes) == cudaSuccess, "cudaMalloc yref");
  check(cudaMalloc(&ygen, yq_bytes) == cudaSuccess, "cudaMalloc ygen");
  if (g_failures) return;

  cudaMemcpy(ad, a.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(resref, 0, xy_bytes);
  cudaMemset(resgen, 0, xy_bytes);
  cudaMemset(yref, 0, yq_bytes);
  cudaMemset(ygen, 0, yq_bytes);

  // The reference: the fused pair the region replaces.
  auto* yq_ref = reinterpret_cast<sonicboom::quant::cuda::BlockQ8_1Mmq*>(yref);
  check(sonicboom::nn::cuda::add_dev(ad, bd, resref, int(elems)),
        (tag + ": reference add").c_str());
  check(sonicboom::quant::cuda::rms_norm_quantize_mmq_q8_1_dev(
            std::span<const float>(w), resref, eps, yq_ref, dim, tokens, ds),
        (tag + ": reference quantize").c_str());

  // The generated kernel, loaded and launched through the JIT: one 256-thread
  // block per token row.
  void *aarg = ad;
  void *barg = bd;
  void *warg = wd;
  void *yarg = ygen;
  void *rarg = resgen;
  int dimarg = dim;
  int nbarg = tokens;
  int dsarg = ds;
  float earg = eps;
  void *args[] = {&aarg,  &barg, &warg,  &yarg,     &rarg,
                  &dimarg, &nbarg, &dsarg, &earg};
  check(sonicboom::planner::glue::launch(
            "add_rms_norm_quantize_mmq_q8_1", args, tokens,
            sonicboom::compiler::kAddRmsNormBlockThreads,
            sonicboom::compiler::kAddRmsNormSharedBytes),
        (tag + ": generated kernel launch").c_str());

  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<uint8_t> got(yq_bytes), ref(yq_bytes);
  cudaMemcpy(got.data(), ygen, yq_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), yref, yq_bytes, cudaMemcpyDeviceToHost);
  check(std::memcmp(got.data(), ref.data(), yq_bytes) == 0,
        (tag + ": byte-identical mmq blocks").c_str());
  if (std::memcmp(got.data(), ref.data(), yq_bytes) != 0) {
    for (int t = 0; t < tokens; ++t) {
      for (int blk = 0; blk < nblocks; ++blk) {
        const size_t off = (size_t(t) * size_t(nblocks) + size_t(blk)) *
                           sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq);
        if (std::memcmp(got.data() + off, ref.data() + off,
                        sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq)) == 0)
          continue;
        float dg = 0, dr = 0;
        std::memcpy(&dg, got.data() + off, sizeof(float));
        std::memcpy(&dr, ref.data() + off, sizeof(float));
        std::cerr << "    first differing block t=" << t << " blk=" << blk
                  << ": d got=" << dg << " ref=" << dr << "\n";
        break;
      }
    }
  }

  std::vector<float> rgot(elems), rref(elems);
  cudaMemcpy(rgot.data(), resgen, xy_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(rref.data(), resref, xy_bytes, cudaMemcpyDeviceToHost);
  const bool res_equal =
      std::memcmp(rgot.data(), rref.data(), xy_bytes) == 0;
  if (!res_equal) {
    for (size_t i = 0; i < elems; ++i) {
      if (std::memcmp(&rgot[i], &rref[i], sizeof(float)) != 0) {
        std::cerr << "    first differing residual " << i
                  << ": got=" << rgot[i] << " ref=" << rref[i] << "\n";
        break;
      }
    }
  }
  check(res_equal, (tag + ": byte-identical residual sum").c_str());

  cudaFree(ad);
  cudaFree(bd);
  cudaFree(wd);
  cudaFree(resref);
  cudaFree(resgen);
  cudaFree(yref);
  cudaFree(ygen);
}

// One prefill norm+add region launch, compared byte for byte with the
// reference pair (rms_norm_heads_dev then add_dev).
void test_rms_norm_add(std::mt19937& rng, int tokens, int dim) {
  const std::string tag = "rms_norm_add tokens=" + std::to_string(tokens) +
                          " dim=" + std::to_string(dim);

  const size_t elems = size_t(tokens) * size_t(dim);
  const size_t xy_bytes = elems * sizeof(float);
  std::vector<float> x(elems), r(elems);
  std::vector<float>& w = persistent_weight(dim);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : r) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  const float eps = 1e-6f;

  float *xd = nullptr, *rd = nullptr, *wd = nullptr, *nd = nullptr;
  float *yref = nullptr, *ygen = nullptr;
  check(cudaMalloc(&xd, xy_bytes) == cudaSuccess, "cudaMalloc x");
  check(cudaMalloc(&rd, xy_bytes) == cudaSuccess, "cudaMalloc r");
  check(cudaMalloc(&wd, dim * sizeof(float)) == cudaSuccess, "cudaMalloc w");
  check(cudaMalloc(&nd, xy_bytes) == cudaSuccess, "cudaMalloc n");
  check(cudaMalloc(&yref, xy_bytes) == cudaSuccess, "cudaMalloc yref");
  check(cudaMalloc(&ygen, xy_bytes) == cudaSuccess, "cudaMalloc ygen");
  if (g_failures) return;

  cudaMemcpy(xd, x.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(rd, r.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(yref, 0, xy_bytes);
  cudaMemset(ygen, 0, xy_bytes);

  // The reference: the two kernels the region replaces.
  check(sonicboom::nn::cuda::rms_norm_heads_dev(std::span<const float>(w), xd,
                                                eps, nd, dim, tokens),
        (tag + ": reference norm").c_str());
  check(sonicboom::nn::cuda::add_dev(rd, nd, yref, int(elems)),
        (tag + ": reference add").c_str());

  // The generated kernel, loaded and launched through the JIT: one 256-thread
  // block per row.
  void *xarg = xd;
  void *rarg = rd;
  void *warg = wd;
  void *yarg = ygen;
  int dimarg = dim;
  float earg = eps;
  void *args[] = {&xarg, &rarg, &warg, &yarg, &dimarg, &earg};
  check(sonicboom::planner::glue::launch(
            "rms_norm_add", args, tokens,
            sonicboom::compiler::kRmsNormAddBlockThreads,
            sonicboom::compiler::kRmsNormAddSharedBytes),
        (tag + ": generated kernel launch").c_str());

  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<float> got(elems), ref(elems);
  cudaMemcpy(got.data(), ygen, xy_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), yref, xy_bytes, cudaMemcpyDeviceToHost);
  const bool equal = std::memcmp(got.data(), ref.data(), xy_bytes) == 0;
  if (!equal) {
    for (size_t i = 0; i < elems; ++i) {
      if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
        std::cerr << "    first differing element " << i << ": got=" << got[i]
                  << " ref=" << ref[i] << "\n";
        break;
      }
    }
  }
  check(equal, (tag + ": byte-identical output").c_str());

  cudaFree(xd);
  cudaFree(rd);
  cudaFree(wd);
  cudaFree(nd);
  cudaFree(yref);
  cudaFree(ygen);
}

// One attention-prep region launch (Stage 5d): per-head norm + optional RoPE +
// optional fp16 rounding.
//
//   rope=0, cast=1 (V): byte-identical to rms_norm_heads_dev + cast_fp16_dev —
//                       no trig is involved, so the reduction, the apply and
//                       the fp16 roundtrip must all match exactly.
//   rope=1          (Q/K): the rotation calls libdevice through MLIR's LLVM,
//                       which is not the nvcc that built the reference kernel;
//                       the two round trig differently by up to 1 ulp. Check
//                       the rotation against a host fp64 evaluation of the
//                       kernel's formula, and (cast=1) require every element
//                       that still differs from the reference to be exactly
//                       one fp16 ulp away — the fp16 rounding itself is exact.
void test_heads_norm_rope_cast(std::mt19937& rng, int heads, int dim,
                               int positions, bool do_rope, bool do_cast,
                               bool with_weight, bool with_ff) {
  const int hpp = heads / positions;
  const std::string tag =
      "heads_norm_rope_cast heads=" + std::to_string(heads) + " dim=" +
      std::to_string(dim) + " pos=" + std::to_string(positions) +
      " rope=" + std::to_string(do_rope ? 1 : 0) +
      " cast=" + std::to_string(do_cast ? 1 : 0) +
      " w=" + std::to_string(with_weight ? 1 : 0) +
      " ff=" + std::to_string(with_ff ? 1 : 0);

  const size_t elems = size_t(heads) * size_t(dim);
  const size_t xy_bytes = elems * sizeof(float);
  const int half = dim / 2;
  std::vector<float> x(elems);
  std::vector<float>& w = persistent_weight(dim);
  std::vector<float>& ff = persistent_weight(half);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  for (auto& v : ff) v = std::uniform_real_distribution<float>(0.5f, 2.0f)(rng);
  const float eps = 1e-6f;
  const float base = 10000.0f;
  const float log_base = std::log(base);
  const uint64_t start_pos = 37;

  float *xd = nullptr, *wd = nullptr, *ref = nullptr, *gen = nullptr;
  float *gen_pre = nullptr, *nrm = nullptr;
  uint64_t* dpos = nullptr;
  float* dff = nullptr;
  check(cudaMalloc(&xd, xy_bytes) == cudaSuccess, "cudaMalloc x");
  check(cudaMalloc(&wd, with_weight ? dim * sizeof(float) : 1) == cudaSuccess,
        "cudaMalloc w");
  check(cudaMalloc(&ref, xy_bytes) == cudaSuccess, "cudaMalloc ref");
  check(cudaMalloc(&gen, xy_bytes) == cudaSuccess, "cudaMalloc gen");
  check(cudaMalloc(&gen_pre, xy_bytes) == cudaSuccess, "cudaMalloc gen_pre");
  check(cudaMalloc(&nrm, xy_bytes) == cudaSuccess, "cudaMalloc nrm");
  check(cudaMalloc(&dpos, sizeof(uint64_t)) == cudaSuccess, "cudaMalloc pos");
  if (with_ff)
    check(cudaMalloc(&dff, half * sizeof(float)) == cudaSuccess, "cudaMalloc ff");
  if (g_failures) return;

  cudaMemcpy(xd, x.data(), xy_bytes, cudaMemcpyHostToDevice);
  if (with_weight)
    cudaMemcpy(wd, w.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dpos, &start_pos, sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (with_ff) cudaMemcpy(dff, ff.data(), half * sizeof(float),
                          cudaMemcpyHostToDevice);

  // The reference chain: norm, optional in-place rope (batched for prefill),
  // optional cast.
  const std::span<const float> wspan =
      with_weight ? std::span<const float>(w) : std::span<const float>();
  const std::span<const float> ffspan =
      with_ff ? std::span<const float>(ff) : std::span<const float>();
  check(sonicboom::nn::cuda::rms_norm_heads_dev(wspan, xd, eps, nrm, dim,
                                                heads),
        (tag + ": reference norm").c_str());
  cudaMemcpy(ref, nrm, xy_bytes, cudaMemcpyDeviceToDevice);
  if (do_rope) {
    const bool ok =
        positions == 1
            ? sonicboom::nn::cuda::rope_neox_heads_dev_pos(ref, dim, heads,
                                                           dpos, base, 1.0f,
                                                           ffspan)
            : sonicboom::nn::cuda::rope_neox_heads_batched_dev_pos(
                  ref, dim, hpp, positions, dpos, base, 1.0f, ffspan);
    check(ok, (tag + ": reference rope").c_str());
  }
  if (do_cast)
    check(sonicboom::nn::cuda::cast_fp16_dev(ref, int(elems)),
          (tag + ": reference cast").c_str());

  auto launch = [&](float* dst, int cast_flag) {
    // Kernel ABI: (x, w, has_w, y, pos, head_dim, eps, freq_scale, log_base,
    // ff, heads_per_pos, do_rope, do_cast).
    void* xarg = xd;
    void* warg = with_weight ? static_cast<void*>(wd) : nullptr;
    int haswarg = with_weight ? 1 : 0;
    void* yarg = dst;
    void* posarg = dpos;
    int dimarg = dim;
    float earg = eps;
    float fsarg = 1.0f;
    float lbarg = log_base;
    void* ffarg = with_ff ? static_cast<void*>(dff) : nullptr;
    int hpparg = hpp;
    int ropearg = do_rope ? 1 : 0;
    int castarg = cast_flag;
    void* args[] = {&xarg,   &warg,  &haswarg, &yarg,   &posarg, &dimarg,
                    &earg,   &fsarg, &lbarg,   &ffarg,  &hpparg, &ropearg,
                    &castarg};
    return sonicboom::planner::glue::launch(
        "heads_norm_rope_cast", args, heads,
        sonicboom::compiler::kHeadsNormRopeCastBlockThreads,
        sonicboom::compiler::kHeadsNormRopeCastSharedBytes);
  };

  check(launch(gen, do_cast ? 1 : 0), (tag + ": generated launch").c_str());
  // The unrounded launch is what the fp64 check compares: with cast on, the
  // generated value is already fp16-rounded.
  if (do_rope && do_cast)
    check(launch(gen_pre, 0), (tag + ": generated pre-cast launch").c_str());
  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<float> got(elems), refv(elems), pre(elems);
  cudaMemcpy(got.data(), gen, xy_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(refv.data(), ref, xy_bytes, cudaMemcpyDeviceToHost);

  if (!do_rope) {
    // V chain: the whole chain is trig-free; byte identity is required.
    const bool equal = std::memcmp(got.data(), refv.data(), xy_bytes) == 0;
    if (!equal) {
      for (size_t i = 0; i < elems; ++i) {
        if (std::memcmp(&got[i], &refv[i], sizeof(float)) != 0) {
          std::cerr << "    first differing element " << i << ": got=" << got[i]
                    << " ref=" << refv[i] << "\n";
          break;
        }
      }
    }
    check(equal, (tag + ": byte-identical output").c_str());
  } else {
    // The fp64 check: the kernel's formula, evaluated in double over the
    // reference norm output. theta's argument is rounded to float first (the
    // kernel computes it in float); exp/cos/sin are the double functions.
    std::vector<float> normv(elems);
    cudaMemcpy(normv.data(), nrm, xy_bytes, cudaMemcpyDeviceToHost);
    if (do_cast) cudaMemcpy(pre.data(), gen_pre, xy_bytes, cudaMemcpyDeviceToHost);
    const std::vector<float>& gp = do_cast ? pre : got;
    double max_abs = 0.0;
    size_t bad = 0;
    for (int h = 0; h < heads; ++h) {
      const float pos = float(start_pos + uint64_t(h / hpp));
      for (int off = 0; off < half; ++off) {
        // The kernel's theta, built at the same float rounding points, with
        // only exp/cos/sin evaluated in double: this isolates the rotation's
        // trig from any formula error.
        const float a1 = (-2.0f * float(off)) / float(dim);
        const float e = float(std::exp(double(a1 * log_base)));
        const float theta = 1.0f * pos * e;
        const float th = with_ff ? theta / ff[off] : theta;
        const double c = std::cos(double(th)), s = std::sin(double(th));
        const double x0 = normv[size_t(h) * dim + off];
        const double x1 = normv[size_t(h) * dim + off + half];
        const double e0 = x0 * c - x1 * s;
        const double e1 = x0 * s + x1 * c;
        const double g0 = gp[size_t(h) * dim + off];
        const double g1 = gp[size_t(h) * dim + off + half];
        max_abs = std::max({max_abs, std::abs(g0 - e0), std::abs(g1 - e1)});
        if (std::abs(g0 - e0) > 2e-5 || std::abs(g1 - e1) > 2e-5) ++bad;
      }
    }
    std::cout << "  " << tag << ": max |gen - fp64| = " << max_abs << "\n";
    check(bad == 0, (tag + ": rotation within 2e-5 of the fp64 formula").c_str());

    if (do_cast) {
      // Every remaining difference from the reference must be one fp16 ulp:
      // both values are fp16-roundtripped, so their bit patterns are halves
      // and adjacent halves differ by one. A wrong rounding or a larger value
      // error would not satisfy this.
      size_t differ = 0, far = 0;
      for (size_t i = 0; i < elems; ++i) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &got[i], sizeof(a));
        std::memcpy(&b, &refv[i], sizeof(b));
        if (a == b) continue;
        ++differ;
        const int raw_a = static_cast<int>(a);
        const int raw_b = static_cast<int>(b);
        if (std::abs(raw_a - raw_b) > 1) ++far;
      }
      std::cout << "  " << tag << ": " << differ << "/" << elems
                << " elements differ from the reference after fp16 rounding\n";
      check(far == 0,
            (tag + ": differences are at most one fp16 ulp").c_str());
    } else {
      size_t differ = 0;
      for (size_t i = 0; i < elems; ++i)
        if (std::memcmp(&got[i], &refv[i], sizeof(float)) != 0) ++differ;
      std::cout << "  " << tag << ": " << differ << "/" << elems
                << " elements differ from the reference rotation\n";
    }
  }

  cudaFree(xd);
  cudaFree(wd);
  cudaFree(ref);
  cudaFree(gen);
  cudaFree(gen_pre);
  cudaFree(nrm);
  cudaFree(dpos);
  if (with_ff) cudaFree(dff);
}

} // namespace

int main() {
  if (!have_cuda()) {
    std::cout << "no CUDA device: skipping\n";
    return 0;
  }
  const int arch = device_arch();
  std::cout << "glue codegen: sm_" << arch << "\n";

  // The PTX is generated once per region and cached on disk by glue_jit; this
  // call exercises the MLIR pipeline itself.
  auto ptx = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::RmsNormQuantizeQ8_1, arch);
  check(ptx.has_value(), "generate_glue_ptx returned PTX");
  if (ptx) {
    check(ptx->find(sonicboom::compiler::glue_kernel_entry(
              sonicboom::compiler::GlueRegionKind::RmsNormQuantizeQ8_1)) !=
              std::string::npos,
          "PTX contains the kernel entry symbol");
  }

  auto ptx_mmq = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::AddRmsNormQuantizeMmqQ8_1, arch);
  check(ptx_mmq.has_value(), "generate_glue_ptx returned prefill PTX");
  if (ptx_mmq) {
    check(
        ptx_mmq->find(sonicboom::compiler::glue_kernel_entry(
                  sonicboom::compiler::GlueRegionKind::
                      AddRmsNormQuantizeMmqQ8_1)) != std::string::npos,
        "prefill PTX contains the kernel entry symbol");
  }

  auto ptx_norm_rope = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::HeadsNormRopeCast, arch);
  check(ptx_norm_rope.has_value(),
        "generate_glue_ptx returned norm+rope+cast PTX");
  if (ptx_norm_rope) {
    check(ptx_norm_rope->find(sonicboom::compiler::glue_kernel_entry(
              sonicboom::compiler::GlueRegionKind::HeadsNormRopeCast)) !=
              std::string::npos,
          "norm+rope+cast PTX contains the kernel entry symbol");
    // The rotation's trig is libdevice; the link must have resolved it into
    // the PTX rather than leaving an undefined __nv_* reference.
    check(ptx_norm_rope->find("__nv_cosf") == std::string::npos,
          "norm+rope+cast PTX has no unresolved libdevice call");
  }

  auto ptx_norm_add = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::RmsNormAdd, arch);
  check(ptx_norm_add.has_value(), "generate_glue_ptx returned norm+add PTX");
  if (ptx_norm_add) {
    check(ptx_norm_add->find(sonicboom::compiler::glue_kernel_entry(
              sonicboom::compiler::GlueRegionKind::RmsNormAdd)) !=
              std::string::npos,
          "norm+add PTX contains the kernel entry symbol");
  }

  std::mt19937 rng(0x5eed5a);
  for (int n : {512, 2048, 2560}) test_rms_norm_quantize_q8_1(rng, n);
  // The generated prefill region over Gemma's and Qwen's hidden sizes, both mmq
  // scale layouts, and a single-block shape (only warp 0 has quantize work).
  test_add_rms_norm_quantize_mmq_q8_1(rng, 8, 2560, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 8, 2560, 1);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 5, 896, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 5, 896, 1);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 3, 128, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 3, 256, 1);
  // The prefill norm+add region over Gemma's hidden size, the PLE projection
  // width, a decode-sized row, and a single row (only one block's worth of
  // work).
  test_rms_norm_add(rng, 8, 2560);
  test_rms_norm_add(rng, 8, 256);
  test_rms_norm_add(rng, 5, 896);
  test_rms_norm_add(rng, 1, 2560);
  // The attention prep, one chain per shape the model graph emits: V
  // (norm -> cast), Q (norm -> rope), K (norm -> rope -> cast), decode (one
  // position) and prefill (batch 2, two positions). Gemma's local head_dim
  // 256, its global head_dim 512 (the full shared stage), and Qwen's 128;
  // with and without the norm weight and the freq factors.
  test_heads_norm_rope_cast(rng, 2, 256, 1, false, true, false, false);
  test_heads_norm_rope_cast(rng, 8, 256, 1, true, false, true, true);
  test_heads_norm_rope_cast(rng, 2, 256, 1, true, true, true, true);
  test_heads_norm_rope_cast(rng, 14, 128, 1, true, false, true, false);
  test_heads_norm_rope_cast(rng, 8, 512, 1, true, false, true, true);
  test_heads_norm_rope_cast(rng, 2, 512, 1, true, true, true, true);
  test_heads_norm_rope_cast(rng, 8, 256, 2, true, false, true, true);
  test_heads_norm_rope_cast(rng, 16, 256, 2, true, true, true, false);

  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "glue codegen: all checks passed\n";
  return 0;
}
